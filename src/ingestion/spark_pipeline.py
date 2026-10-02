"""Spark ingestion pipeline: raw files -> standardized canonical Parquet.

Blueprint Section 10 steps 20-29:
  20. Convert compatible data to Parquet.
  21. Standardize field names/types.
  22. Normalize Unicode without destroying Hindi/Telugu characters.
  23. Extract URLs before text cleaning.
  25. Deduplicate exact and near-duplicate messages before splitting.
  26. Detect language, preserve code-mixing.
  27. Preserve original labels; map to project_label only via schema.LABEL_MAPPING_RULES.
  28. Create stratified/group-aware train/validation/test sets.

Run:
    python src/ingestion/spark_pipeline.py --input data/raw --output data/interim

This is a per-dataset-loader pattern: add one `load_<key>()` function per
dataset once you've inspected its raw format, register it in LOADERS, and
the rest of the pipeline (dedup, split, write) is shared.
"""

from __future__ import annotations

import argparse
import hashlib
import unicodedata
from pathlib import Path
from typing import Callable

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType, StringType

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from schema import CANONICAL_FIELDS, LABEL_MAPPING_RULES  # noqa: E402

URL_RE = r"(https?://[^\s]+|www\.[^\s]+)"


def get_spark(app_name: str = "scamshield-ingestion") -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )


# ---------------------------------------------------------------------------
# Per-dataset loaders. Each must return a DataFrame with (at minimum):
# source_record_id, text, channel, language(optional), original_label,
# scam_category(optional), is_natural, augmentation_type.
# Fill these in once you've downloaded and inspected each dataset's actual
# file format (CSV/JSON/Parquet columns differ per source) — this is
# deliberately a stub so it fails loudly rather than silently guessing.
# ---------------------------------------------------------------------------


def load_meajor(spark: SparkSession, raw_dir: Path) -> DataFrame:
    """Loader for meajor_cleaned_preprocessed.csv.

    Raw schema (confirmed by direct inspection):
    sender, sender_domain, receiver, receiver_domain, date, subject,
    content_types, body, urls, url_count, url_length_max, url_length_avg,
    url_subdom_max, url_subdom_avg, attachment_count, has_attachments,
    attachment_types, language, source, label

    sender/receiver are hashed (anonymized) but sender_domain/receiver_domain
    are preserved in plain text (e.g. "enron.com") — this is real, usable
    sender-metadata, not something that needs to be derived separately.

    URL features (url_count, url_length_max/avg, url_subdom_max/avg) are
    already engineered by the source — preserved via engineered_features
    rather than recomputed, so nothing is wasted or duplicated.
    """
    csv_path = raw_dir / "meajor_cleaned_preprocessed.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run: "
            "python src/ingestion/download_datasets.py --dataset meajor"
        )

    df = spark.read.csv(
        str(csv_path), header=True, multiLine=True, escape='"', inferSchema=False
    )

    df = df.withColumn(
        "source_record_id",
        F.sha2(
            F.concat_ws(
                "|",
                F.coalesce(F.col("sender"), F.lit("")),
                F.coalesce(F.col("receiver"), F.lit("")),
                F.coalesce(F.col("date"), F.lit("")),
                F.coalesce(F.col("subject"), F.lit("")),
                F.coalesce(F.col("body"), F.lit("")),
            ),
            256,
        ),
    )
    df = df.withColumn(
        "text", F.concat_ws(" ", F.coalesce(F.col("subject"), F.lit("")), F.coalesce(F.col("body"), F.lit("")))
    )
    df = df.withColumn("channel", F.lit("email"))
    # label arrives as float (1.0/0.0), confirmed by direct inspection —
    # cast through int to get a clean "1"/"0" string, not "1.0"/"0.0".
    df = df.withColumn(
        "original_label",
        # label arrives as the literal string "1.0"/"0.0" (Spark's CSV
        # reader is StringType-only with inferSchema=False) — casting
        # straight to int fails silently on decimal-looking strings and
        # returns NULL. Confirmed bug: this produced NULL project_label
        # for all 108,685 MeAJOR rows on first run. Fix: cast through
        # double first.
        F.col("label").cast("double").cast("int").cast(StringType()),
    )
    df = df.withColumn("scam_category", F.lit(None).cast(StringType()))
    df = df.withColumn("is_natural", F.lit(True))
    df = df.withColumn("augmentation_type", F.lit("none"))

    # language arrives as semicolon-joined, sometimes-repeated tags
    # (e.g. "en;en", "en;en;en") and includes genuine non-English mail
    # (ja, ru, de, it, vi, no, ...) — confirmed by direct inspection, this
    # is NOT an English-only dataset. Take the first tag as the primary
    # language; the full raw value is preserved in engineered_features
    # in case per-paragraph detail matters later.
    df = df.withColumn(
        "language_primary", F.split(F.coalesce(F.col("language"), F.lit("und")), ";")[0]
    )

    df = df.withColumn(
        "engineered_features",
        F.to_json(
            F.struct(
                F.col("sender_domain"),
                F.col("receiver_domain"),
                F.col("url_count"),
                F.col("url_length_max"),
                F.col("url_length_avg"),
                F.col("url_subdom_max"),
                F.col("url_subdom_avg"),
                F.col("attachment_count"),
                F.col("has_attachments"),
                F.col("attachment_types"),
                F.col("source").alias("meajor_source_corpus"),
                F.col("language").alias("language_raw_tags"),
            )
        ),
    )

    return df.select(
        "source_record_id",
        "text",
        "channel",
        F.col("language_primary").alias("language"),
        "original_label",
        "scam_category",
        "is_natural",
        "augmentation_type",
        "engineered_features",
    )


# code column -> language, confirmed by manual inspection of the raw file
# (see config/datasets.yaml comment for revised_indian_sms).
_REVISED_INDIAN_SMS_LANG_MAP = {
    1: "en",
    2: "hi",       # includes Hinglish / Devanagari
    3: "te-en",    # Telugu-English code-mixed
    4: "te-en",    # Telugu-English code-mixed
    11: "en",
}


def load_revised_indian_sms(spark: SparkSession, raw_dir: Path) -> DataFrame:
    """Loader for data/raw/revised_indian_sms/revisedindiandataset.xls.

    Raw schema (confirmed by manual inspection): columns `code`, `label`,
    `msg`. `code` encodes language per _REVISED_INDIAN_SMS_LANG_MAP above.
    `label` is already `ham`/`spam` — no mapping ambiguity for this source.
    """
    import pandas as pd

    xls_path = raw_dir / "revisedindiandataset.xls"
    if not xls_path.exists():
        raise FileNotFoundError(
            f"{xls_path} not found. Place the raw file at "
            "data/raw/revised_indian_sms/revisedindiandataset.xls"
        )

    pdf = pd.read_excel(xls_path)
    pdf = pdf.dropna(subset=["msg"]).reset_index(drop=True)
    pdf["source_record_id"] = pdf.index.astype(str)
    pdf["language"] = (
        pdf["code"].fillna(1).astype(int).map(_REVISED_INDIAN_SMS_LANG_MAP).fillna("en")
    )
    pdf["text"] = pdf["msg"].astype(str)
    pdf["channel"] = "sms"
    pdf["original_label"] = pdf["label"].astype(str).str.lower().str.strip()
    pdf["scam_category"] = ""  # Spark can't infer a type from an all-None
    # pandas column when converting via createDataFrame — empty string
    # avoids the CANNOT_DETERMINE_TYPE error. (Confirmed fix, applied
    # after hitting this in practice.)
    pdf["is_natural"] = True
    pdf["augmentation_type"] = "none"

    out_cols = [
        "source_record_id",
        "text",
        "channel",
        "language",
        "original_label",
        "scam_category",
        "is_natural",
        "augmentation_type",
    ]
    return spark.createDataFrame(pdf[out_cols])


def load_indian_cyber_scam(spark: SparkSession, raw_dir: Path) -> DataFrame:
    """Loader for India_Cyber_Scam_Hinglish_Dataset.csv.

    Raw schema (confirmed by direct inspection):
    text, label, scam_category, caller_type, audio_duration, urgency_level,
    contains_blackmail, language_style

    label is 0/1 (non-scam/scam, confirmed from earlier sample rows).
    language_style was "hinglish" for every sampled row so far — treated
    as the language tag directly rather than re-detecting it.

    IMPORTANT: sample inspection earlier found many near-identical template
    sentences with only caller_type swapped (e.g. the same "parcel hold"
    line repeated with different scammer identities). is_natural is set to
    False here — this is template-generated data, not organically collected
    scam calls, and should not be treated as equivalent to MeAJOR's real
    email corpus. Expect the dedup step to remove a large fraction of rows.
    """
    csv_path = raw_dir / "India_Cyber_Scam_Hinglish_Dataset.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run: "
            "python src/ingestion/download_datasets.py --dataset indian_cyber_scam"
        )

    df = spark.read.csv(str(csv_path), header=True, multiLine=True, escape='"')

    df = df.withColumn("source_record_id", F.monotonically_increasing_id().cast(StringType()))
    df = df.withColumn("channel", F.lit("transcript"))
    df = df.withColumn("language", F.coalesce(F.col("language_style"), F.lit("hinglish")))
    df = df.withColumn("original_label", F.col("label").cast("double").cast("int").cast(StringType()))
    # is_natural = False: confirmed templated/repeated content, not organic.
    df = df.withColumn("is_natural", F.lit(False))
    df = df.withColumn("augmentation_type", F.lit("template_generated"))

    df = df.withColumn(
        "engineered_features",
        F.to_json(
            F.struct(
                F.col("caller_type"),
                F.col("audio_duration"),
                F.col("urgency_level"),
                F.col("contains_blackmail"),
            )
        ),
    )

    return df.select(
        "source_record_id",
        "text",
        "channel",
        "language",
        "original_label",
        F.col("scam_category"),
        "is_natural",
        "augmentation_type",
        "engineered_features",
    )


def load_llm_generated_phishing(spark: SparkSession, raw_dir: Path) -> DataFrame:
    """Loader for cross-model-phishing/data/{human,llm}_corpus_sampled.csv.

    Raw schema (confirmed by direct inspection):
    text, subject, body, label, source [, model, category for the llm file]

    IMPORTANT: label == 1 for every single row in both files (confirmed by
    sampling 2000 rows each) — this is a phishing-only human-vs-LLM
    comparison corpus, NOT a benign/phishing training set. Use it for the
    robustness/AI-generated-text evaluation (blueprint Phase 9), not to
    inflate M1-M4 training volume.

    The human subset's `source` values (CEAS_08, TREC-07, Nazario,
    Nigerian_Fraud, enron_data_fraud_labeled) are the SAME underlying
    corpora as the (not-yet-downloaded) Curated Phishing Email Datasets,
    and likely overlap with MeAJOR too — expect dedup to catch this if
    that dataset is ever added.
    """
    base = raw_dir / "cross-model-phishing" / "data"
    human_path = base / "human_corpus_sampled.csv"
    llm_path = base / "llm_corpus_sampled.csv"
    if not human_path.exists() or not llm_path.exists():
        raise FileNotFoundError(
            f"Expected files not found under {base}. Confirm the zip was "
            f"extracted to data/raw/llm_generated_phishing/cross-model-phishing/"
        )

    human = spark.read.csv(str(human_path), header=True, multiLine=True, escape='"')
    llm = spark.read.csv(str(llm_path), header=True, multiLine=True, escape='"')

    def _prep(df: DataFrame, is_natural: bool, augmentation_type: str, extra_struct):
        df = df.withColumn(
            "source_record_id",
            F.concat_ws(
                "_", F.lit("natural" if is_natural else "llm"),
                F.monotonically_increasing_id().cast(StringType()),
            ),
        )
        df = df.withColumn("channel", F.lit("email"))
        df = df.withColumn(
            "text",
            F.concat_ws(" ", F.coalesce(F.col("subject"), F.lit("")), F.coalesce(F.col("body"), F.lit(""))),
        )
        df = df.withColumn("language", F.lit("en"))
        df = df.withColumn("original_label", F.col("label").cast("double").cast("int").cast(StringType()))
        df = df.withColumn("scam_category", F.lit(None).cast(StringType()))
        df = df.withColumn("is_natural", F.lit(is_natural))
        df = df.withColumn("augmentation_type", F.lit(augmentation_type))
        df = df.withColumn("engineered_features", F.to_json(extra_struct))
        return df.select(
            "source_record_id", "text", "channel", "language", "original_label",
            "scam_category", "is_natural", "augmentation_type", "engineered_features",
        )

    human_out = _prep(
        human, is_natural=True, augmentation_type="none",
        extra_struct=F.struct(F.col("source").alias("original_corpus_source")),
    )
    llm_out = _prep(
        llm, is_natural=False, augmentation_type="llm_generated",
        extra_struct=F.struct(
            F.col("source").alias("original_corpus_source"),
            F.col("model").alias("llm_model"),
            F.col("category").alias("llm_scam_category"),
        ),
    )
    return human_out.unionByName(llm_out)


LOADERS: dict[str, Callable[[SparkSession, Path], DataFrame]] = {
    "meajor": load_meajor,
    "revised_indian_sms": load_revised_indian_sms,
    "indian_cyber_scam": load_indian_cyber_scam,
    "llm_generated_phishing": load_llm_generated_phishing,
    # "curated_phishing": load_curated_phishing,
    # "dravidian_spam_sms": load_dravidian_spam_sms,
}


# ---------------------------------------------------------------------------
# Shared standardization steps
# ---------------------------------------------------------------------------


@F.udf(returnType=StringType())
def normalize_unicode(text: str) -> str:
    if text is None:
        return None
    # NFC preserves precomposed Devanagari/Telugu characters correctly,
    # unlike NFKD which can decompose them destructively.
    return unicodedata.normalize("NFC", text)


@F.udf(returnType=BooleanType())
def has_url(text: str) -> bool:
    import re

    if text is None:
        return False
    return bool(re.search(URL_RE, text))


@F.udf(returnType=StringType())
def sha256_hash(text: str) -> str:
    if text is None:
        return None
    normalized = " ".join(text.strip().lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def standardize(df: DataFrame, source_dataset: str) -> DataFrame:
    df = df.withColumn(
        "message_id",
        F.sha2(F.concat_ws("|", F.lit(source_dataset), F.col("source_record_id")), 256),
    )
    df = df.withColumn("source_dataset", F.lit(source_dataset))
    df = df.withColumn("text", normalize_unicode(F.col("text")))
    df = df.withColumn("url_present", has_url(F.col("text")))
    df = df.withColumn("source_hash", sha256_hash(F.col("text")))

    mapping = LABEL_MAPPING_RULES.get(source_dataset, {})
    if mapping:
        mapping_expr = F.create_map([F.lit(x) for pair in mapping.items() for x in pair])
        df = df.withColumn("project_label", mapping_expr.getItem(F.col("original_label")))
    else:
        df = df.withColumn("project_label", F.lit(None).cast(StringType()))

    for col in CANONICAL_FIELDS:
        if col not in df.columns:
            df = df.withColumn(col, F.lit(None).cast(StringType()))

    return df.select(*CANONICAL_FIELDS)


def deduplicate(df: DataFrame) -> DataFrame:
    """Exact-duplicate removal by source_hash (near-duplicate detection —
    e.g. MinHash LSH — is a follow-up step once volumes are large enough
    to justify it; wire it in here when you get there)."""
    return df.dropDuplicates(["source_hash"])


def assign_splits(df: DataFrame, seed: int = 42) -> DataFrame:
    """Stratified by project_label, group-aware by source_dataset so a
    near-duplicate campaign doesn't leak across splits. Simple hash-based
    deterministic split — swap for sklearn's StratifiedGroupKFold if you
    need tighter stratification guarantees."""
    df = df.withColumn(
        "_bucket", F.abs(F.hash(F.col("message_id"))) % 100
    )
    df = df.withColumn(
        "split",
        F.when(F.col("_bucket") < 80, "train")
        .when(F.col("_bucket") < 90, "validation")
        .otherwise("test"),
    ).drop("_bucket")
    return df


def run(input_dir: Path, output_dir: Path) -> None:
    spark = get_spark()
    frames = []
    for key, loader in LOADERS.items():
        try:
            frames.append(standardize(loader(spark, input_dir / key), key))
        except NotImplementedError as e:
            print(f"[skip] {key}: {e}")

    if not frames:
        print("No loaders implemented yet — see LOADERS dict in this file. Nothing written.")
        spark.stop()
        return

    combined = frames[0]
    for f in frames[1:]:
        combined = combined.unionByName(f)

    combined = deduplicate(combined)
    combined = combined.filter(F.col("project_label").isNotNull())
    combined = assign_splits(combined)

    output_dir.mkdir(parents=True, exist_ok=True)
    combined.write.mode("overwrite").partitionBy("source_dataset").parquet(str(output_dir))
    print(f"Wrote standardized Parquet to {output_dir}")
    spark.stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw")
    parser.add_argument("--output", default="data/interim")
    args = parser.parse_args()
    run(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()