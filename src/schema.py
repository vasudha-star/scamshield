"""Canonical data schema for SCAMSHIELD (blueprint Section 9).

Every dataset gets standardized into this schema during Spark ingestion.
Import CANONICAL_FIELDS / CANONICAL_SCHEMA wherever you need to validate
or construct a DataFrame against the shared contract.
"""

from __future__ import annotations

from pyspark.sql.types import (
    BooleanType,
    StringType,
    StructField,
    StructType,
)

CANONICAL_FIELDS = [
    "message_id",       # internal stable identifier (uuid5 of source_dataset+source_record_id)
    "source_dataset",   # original corpus key, matches config/datasets.yaml
    "source_record_id", # original record identifier
    "text",             # message subject/body or SMS text
    "channel",          # email/sms/chat/transcript
    "language",         # en/hi/hinglish/te/etc.
    "original_label",   # source-provided label, verbatim — never overwritten
    "project_label",    # benign/spam/phishing/scam/unknown — only via documented mapping
    "scam_category",    # KYC/OTP/banking/etc., if supported by source
    "url_present",      # boolean
    "is_natural",       # natural vs synthetic/generated
    "augmentation_type",# none/translation/paraphrase/llm_generated
    "source_hash",      # sha256 of normalized text, for dedup
    "split",            # train/validation/test — assigned once, never reassigned
    "engineered_features", # JSON string of any dataset-native precomputed
                            # features (e.g. MeAJOR's url_count, sender_domain,
                            # url_length_avg) — preserved rather than dropped,
                            # since re-deriving them would waste work already
                            # done by the source. Null where a source has none.
]

CANONICAL_SCHEMA = StructType(
    [
        StructField("message_id", StringType(), nullable=False),
        StructField("source_dataset", StringType(), nullable=False),
        StructField("source_record_id", StringType(), nullable=True),
        StructField("text", StringType(), nullable=False),
        StructField("channel", StringType(), nullable=False),
        StructField("language", StringType(), nullable=True),
        StructField("original_label", StringType(), nullable=True),
        StructField("project_label", StringType(), nullable=True),
        StructField("scam_category", StringType(), nullable=True),
        StructField("url_present", BooleanType(), nullable=True),
        StructField("is_natural", BooleanType(), nullable=False),
        StructField("augmentation_type", StringType(), nullable=False),
        StructField("source_hash", StringType(), nullable=True),
        StructField("split", StringType(), nullable=True),
        StructField("engineered_features", StringType(), nullable=True),
    ]
)

# Label mapping rules must be added here explicitly and documented — never
# inferred implicitly per-dataset. See blueprint Section 5 (taxonomy rule)
# and Section 10 step 27: "preserve original labels; create project labels
# only through documented mapping."
LABEL_MAPPING_RULES: dict[str, dict[str, str]] = {
    # "meajor": {"phishing": "phishing", "benign": "benign"},
    # Fill in per-dataset as you inspect original_label values. Do not guess.
    # revised_indian_sms: confirmed by manual inspection, ham/spam only,
    # no phishing/scam-category distinction available in this source.
    "revised_indian_sms": {"ham": "benign", "spam": "spam"},
    # meajor: mapping confirmed from the dataset's own Zenodo description
    # ("label: 0 = benign; 1 = phishing"), not guessed. Both the raw 0/1
    # form and a possible string form are covered defensively — verify
    # actual distinct values with df.select("label").distinct().show()
    # before trusting this blindly; update this dict if the real values differ.
    "meajor": {
        "0": "benign", "1": "phishing",
        "benign": "benign", "phishing": "phishing",
    },
    # indian_cyber_scam: 0/1 confirmed from sample rows (non-scam/scam).
    # Mapped to "scam" not "phishing" — these are voice-call/transcript
    # scams (KYC, digital arrest, blackmail), not phishing emails; keeping
    # the taxonomy distinct rather than collapsing scam into phishing.
    "indian_cyber_scam": {"0": "benign", "1": "scam"},
    # llm_generated_phishing: confirmed label == 1 for every row in both
    # human and LLM subsets (it's a phishing-only comparison corpus, no
    # benign class present) — maps straight to "phishing".
    "llm_generated_phishing": {"1": "phishing"},
}

VALID_PROJECT_LABELS = {"benign", "spam", "phishing", "scam", "unknown"}
VALID_SPLITS = {"train", "validation", "test"}