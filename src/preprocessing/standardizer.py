"""Data Standardization and Deduplication Pipeline for ScamShield (Phase 3).

Standardizes heterogeneous raw datasets into the Canonical Data Schema:
- Extracts URLs before cleaning
- Performs Unicode normalization (NFKC) preserving Hindi, Telugu, and English
- Generates deterministic message_id (UUID5) and source_hash (SHA-256)
- Preserves verbatim original labels while applying documented project label mappings
- Executes exact deduplication BEFORE train/test splitting
- Saves data/processed/cleaned.parquet
- Generates reports/data_quality_report.json and reports/data_quality_report.md
"""

from __future__ import annotations

import hashlib
import json
import sys
import uuid
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.preprocessing.text_cleaner import clean_text
from src.utils.logger import get_logger

logger = get_logger("standardizer")

# Canonical Schema column list
CANONICAL_COLUMNS = [
    "message_id",
    "source_dataset",
    "source_record_id",
    "text",
    "channel",
    "language",
    "original_label",
    "project_label",
    "scam_category",
    "url_present",
    "extracted_urls",
    "url_features",
    "intent_features",
    "is_natural",
    "augmentation_type",
    "source_hash",
    "split",
]


def generate_record_id(source_dataset: str, source_record_id: str, source_hash: str) -> str:
    """Generates a stable, reproducible UUID5 from dataset provenance and content hash."""
    seed_str = f"{source_dataset}::{source_record_id}::{source_hash}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, seed_str))


def compute_sha256(text: str) -> str:
    """Computes SHA-256 content hash of normalized text for exact deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def process_meajor(raw_path: Path) -> pd.DataFrame:
    logger.info("Processing MeAJOR dataset...")
    df_raw = pd.read_parquet(raw_path)
    records = []

    for idx, row in df_raw.iterrows():
        # Text extraction: combine subject + body if text is not explicit
        body = str(row.get("body", "")) if pd.notnull(row.get("body")) else ""
        subject = str(row.get("subject", "")) if pd.notnull(row.get("subject")) else ""
        raw_text = f"{subject}\n\n{body}" if subject else body
        if not raw_text.strip():
            raw_text = str(row.get("text", ""))

        cleaned, urls, url_present = clean_text(raw_text)
        if not cleaned:
            continue

        h = compute_sha256(cleaned)
        orig_lbl = str(row.get("label", ""))
        # Documented mapping: 0 -> benign, 1 -> phishing
        proj_lbl = "benign" if orig_lbl in ("0", "0.0") else ("phishing" if orig_lbl in ("1", "1.0") else "unknown")

        records.append({
            "source_dataset": "meajor",
            "source_record_id": str(idx),
            "text": cleaned,
            "channel": "email",
            "language": "en",
            "original_label": orig_lbl,
            "project_label": proj_lbl,
            "scam_category": None,
            "url_present": url_present or (float(row.get("url_count", 0.0) or 0.0) > 0),
            "extracted_urls": urls,
            "url_features": None,
            "intent_features": None,
            "is_natural": True,
            "augmentation_type": "none",
            "source_hash": h,
            "split": None,
        })

    df = pd.DataFrame(records)
    logger.info(f"Processed MeAJOR: {len(df):,} valid records.")
    return df


def process_indian_scam(raw_path: Path) -> pd.DataFrame:
    logger.info("Processing Indian Cyber Scam Communication dataset...")
    df_raw = pd.read_csv(raw_path)
    records = []

    for idx, row in df_raw.iterrows():
        raw_text = str(row.get("text", ""))
        cleaned, urls, url_present = clean_text(raw_text)
        if not cleaned:
            continue

        h = compute_sha256(cleaned)
        orig_lbl = str(row.get("label", ""))
        proj_lbl = "benign" if orig_lbl in ("0", "0.0") else ("scam" if orig_lbl in ("1", "1.0") else "unknown")
        category = str(row.get("scam_category", "none"))
        scam_cat = category if category != "none" else None

        records.append({
            "source_dataset": "indian_scam",
            "source_record_id": str(idx),
            "text": cleaned,
            "channel": "transcript",
            "language": "hinglish",
            "original_label": orig_lbl,
            "project_label": proj_lbl,
            "scam_category": scam_cat,
            "url_present": url_present,
            "extracted_urls": urls,
            "url_features": None,
            "intent_features": None,
            "is_natural": False,  # Templated slot-filled calls
            "augmentation_type": "template_generated",
            "source_hash": h,
            "split": None,
        })

    df = pd.DataFrame(records)
    logger.info(f"Processed Indian Scam: {len(df):,} valid records.")
    return df


def process_dravidian_sms(raw_path: Path) -> pd.DataFrame:
    logger.info("Processing Dravidian & Indian SMS Corpus...")
    df_raw = pd.read_excel(raw_path)
    records = []

    lang_code_map = {
        1.0: "en",
        2.0: "hi",
        3.0: "te",  # Telugu-English code-mixed
        4.0: "te",  # Telugu-English code-mixed
        11.0: "en",
    }

    for idx, row in df_raw.iterrows():
        raw_text = str(row.get("msg", ""))
        cleaned, urls, url_present = clean_text(raw_text)
        if not cleaned:
            continue

        h = compute_sha256(cleaned)
        orig_lbl = str(row.get("label", "")).strip().lower()
        # Rule 5: Preserve distinction! ham -> benign, spam -> spam (NOT phishing)
        proj_lbl = "benign" if orig_lbl == "ham" else ("spam" if orig_lbl == "spam" else "unknown")
        code_val = float(row.get("code", 1.0))
        lang = lang_code_map.get(code_val, "en")

        records.append({
            "source_dataset": "dravidian_sms",
            "source_record_id": str(idx),
            "text": cleaned,
            "channel": "sms",
            "language": lang,
            "original_label": orig_lbl,
            "project_label": proj_lbl,
            "scam_category": None,
            "url_present": url_present,
            "extracted_urls": urls,
            "url_features": None,
            "intent_features": None,
            "is_natural": True,
            "augmentation_type": "none",
            "source_hash": h,
            "split": None,
        })

    df = pd.DataFrame(records)
    logger.info(f"Processed Dravidian SMS: {len(df):,} valid records.")
    return df


def process_llm_phishing(data_dir: Path) -> pd.DataFrame:
    logger.info("Processing LLM-generated Phishing Benchmark...")
    records = []

    files = [
        (data_dir / "human_corpus_sampled.csv", True, "none"),
        (data_dir / "llm_corpus_sampled.csv", False, "llm_generated"),
    ]

    for filepath, is_nat, aug_type in files:
        if not filepath.exists():
            continue
        df_sub = pd.read_csv(filepath)
        for idx, row in df_sub.iterrows():
            raw_text = str(row.get("text", ""))
            if not raw_text.strip():
                raw_text = f"{row.get('subject', '')}\n\n{row.get('body', '')}"

            cleaned, urls, url_present = clean_text(raw_text)
            if not cleaned:
                continue

            h = compute_sha256(cleaned)
            records.append({
                "source_dataset": "llm_phishing",
                "source_record_id": f"{aug_type}_{idx}",
                "text": cleaned,
                "channel": "email",
                "language": "en",
                "original_label": str(row.get("label", "1")),
                "project_label": "phishing",
                "scam_category": "phishing",
                "url_present": url_present,
                "extracted_urls": urls,
                "url_features": None,
                "intent_features": None,
                "is_natural": is_nat,
                "augmentation_type": aug_type,
                "source_hash": h,
                "split": None,
            })

    df = pd.DataFrame(records)
    logger.info(f"Processed LLM Phishing: {len(df):,} valid records.")
    return df


def run_standardization() -> pd.DataFrame:
    raw_dir = PROJECT_ROOT / "data" / "raw"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    reports_dir = PROJECT_ROOT / "reports"
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    dfs: list[pd.DataFrame] = []

    # 1. MeAJOR
    meajor_p = raw_dir / "meajor" / "meajor_cleaned_preprocessed.parquet.gzip"
    if meajor_p.exists():
        dfs.append(process_meajor(meajor_p))

    # 2. Indian Cyber Scam
    indian_p = raw_dir / "indian_cyber_scam" / "India_Cyber_Scam_Hinglish_Dataset.csv"
    if indian_p.exists():
        dfs.append(process_indian_scam(indian_p))

    # 3. Dravidian SMS
    sms_p = raw_dir / "revised_indian_sms" / "revisedindiandataset.xls"
    if sms_p.exists():
        dfs.append(process_dravidian_sms(sms_p))

    # 4. LLM Phishing
    llm_p = raw_dir / "llm_generated_phishing" / "cross-model-phishing" / "data"
    if llm_p.exists():
        dfs.append(process_llm_phishing(llm_p))

    if not dfs:
        raise RuntimeError("No raw datasets found in data/raw to standardize!")

    # Combine all
    unified_df = pd.concat(dfs, ignore_index=True)
    raw_total = len(unified_df)
    logger.info(f"Unified raw total before deduplication: {raw_total:,} records.")

    # Deduplication by SHA-256 source_hash (Rule 3)
    dedup_df = unified_df.drop_duplicates(subset=["source_hash"], keep="first").copy()
    dedup_total = len(dedup_df)
    dropped_count = raw_total - dedup_total
    drop_pct = round((dropped_count / raw_total) * 100, 2)
    logger.info(f"Deduplication complete: {dropped_count:,} duplicates dropped ({drop_pct}%). Retained: {dedup_total:,} records.")

    # Generate message_id for deduplicated rows
    dedup_df["message_id"] = dedup_df.apply(
        lambda r: generate_record_id(r["source_dataset"], r["source_record_id"], r["source_hash"]),
        axis=1,
    )

    # Reorder to exact canonical columns
    dedup_df = dedup_df[CANONICAL_COLUMNS]

    # Save to Parquet
    out_parquet = processed_dir / "cleaned.parquet"
    dedup_df.to_parquet(out_parquet, index=False, engine="pyarrow")
    logger.info(f"Saved canonical cleaned dataset to {out_parquet}")

    # Generate Data Quality Report
    generate_data_quality_report(
        raw_total=raw_total,
        dedup_total=dedup_total,
        dropped_count=dropped_count,
        drop_pct=drop_pct,
        df=dedup_df,
        reports_dir=reports_dir,
    )

    return dedup_df


def generate_data_quality_report(
    raw_total: int,
    dedup_total: int,
    dropped_count: int,
    drop_pct: float,
    df: pd.DataFrame,
    reports_dir: Path,
) -> None:
    source_counts = df["source_dataset"].value_counts().to_dict()
    label_counts = df["project_label"].value_counts().to_dict()
    lang_counts = df["language"].value_counts().to_dict()
    channel_counts = df["channel"].value_counts().to_dict()
    url_pct = round((df["url_present"].sum() / len(df)) * 100, 2)
    natural_counts = df["is_natural"].value_counts().to_dict()

    quality_report = {
        "raw_total_records": raw_total,
        "clean_deduplicated_records": dedup_total,
        "dropped_duplicates_count": dropped_count,
        "dropped_duplicates_pct": drop_pct,
        "source_dataset_distribution": {str(k): int(v) for k, v in source_counts.items()},
        "project_label_distribution": {str(k): int(v) for k, v in label_counts.items()},
        "language_distribution": {str(k): int(v) for k, v in lang_counts.items()},
        "channel_distribution": {str(k): int(v) for k, v in channel_counts.items()},
        "url_presence_overall_pct": url_pct,
        "natural_vs_synthetic": {str(k): int(v) for k, v in natural_counts.items()},
        "output_file": "data/processed/cleaned.parquet",
    }

    # Save JSON
    json_path = reports_dir / "data_quality_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2)

    # Save Markdown
    md_path = reports_dir / "data_quality_report.md"
    md_lines = [
        "# ScamShield — Data Quality & Standardization Report (Phase 3)",
        "",
        "## 1. Deduplication & Volume Summary",
        f"- **Raw Ingested Records:** {raw_total:,}",
        f"- **Clean Deduplicated Records:** {dedup_total:,}",
        f"- **Dropped Duplicates (Exact SHA-256 match):** {dropped_count:,} ({drop_pct}%)",
        "",
        "## 2. Canonical Distributions",
        "",
        "### Records by Source Dataset",
        "| Dataset | Clean Records |",
        "|---|---|",
    ]
    for src, cnt in source_counts.items():
        md_lines.append(f"| **{src}** | {cnt:,} |")

    md_lines.extend([
        "",
        "### Project Labels",
        "| Project Label | Count |",
        "|---|---|",
    ])
    for lbl, cnt in label_counts.items():
        md_lines.append(f"| **{lbl}** | {cnt:,} |")

    md_lines.extend([
        "",
        "### Language Breakdown",
        "| Language | Count |",
        "|---|---|",
    ])
    for lng, cnt in lang_counts.items():
        md_lines.append(f"| **{lng}** | {cnt:,} |")

    md_lines.extend([
        "",
        "### Channel Distribution",
        "| Channel | Count |",
        "|---|---|",
    ])
    for ch, cnt in channel_counts.items():
        md_lines.append(f"| **{ch}** | {cnt:,} |")

    md_lines.extend([
        "",
        f"- **Overall URL Presence Rate:** {url_pct}%",
        f"- **Natural vs Synthetic:** {natural_counts.get(True, 0):,} Natural / {natural_counts.get(False, 0):,} Synthetic",
    ])

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    logger.info(f"Saved quality reports to {json_path} and {md_path}")


if __name__ == "__main__":
    df_clean = run_standardization()
    print("\n" + "=" * 65)
    print("      SCAMSHIELD — STANDARDIZATION & CLEANING COMPLETE")
    print("=" * 65)
    print(f"Clean Records Written: {len(df_clean):,}")
    print(f"Output File:            data/processed/cleaned.parquet")
    print(f"Quality Summary:        reports/data_quality_report.md")
    print("=" * 65)
