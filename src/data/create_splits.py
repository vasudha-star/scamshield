"""Leakage-Safe Stratified Train/Validation/Test Splitter for ScamShield (Phase 5).

Implements:
1. Leakage-free split protocol (70% train, 15% validation, 15% test).
2. Multi-key stratification across (source_dataset, project_label) to guarantee
   balanced representation of minority modalities (Telugu SMS, Hindi voice scam,
   LLM phishing) across all partitions.
3. Strict zero-leakage assertions: verifies 0 content hash overlap across splits.
4. Generates immutable split_hash to be recorded in experiment_log.csv.
5. Saves locked split assignments into data/processed/cleaned.parquet.
6. Produces reports/split_report.json and reports/split_report.md.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from src.utils.logger import get_logger

logger = get_logger("create_splits")


def load_config() -> dict[str, Any]:
    cfg_path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_splits() -> pd.DataFrame:
    config = load_config()
    seed = config.get("project", {}).get("random_seed", 42)
    splits_cfg = config.get("splits", {})
    train_ratio = splits_cfg.get("train_ratio", 0.70)
    val_ratio = splits_cfg.get("validation_ratio", 0.15)
    test_ratio = splits_cfg.get("test_ratio", 0.15)

    processed_file = PROJECT_ROOT / "data" / "processed" / "cleaned.parquet"
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not processed_file.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {processed_file}. Run Phase 3 first.")

    logger.info(f"Loading cleaned dataset from {processed_file}...")
    df = pd.read_parquet(processed_file)
    logger.info(f"Loaded {len(df):,} total records.")

    # Filter out unlabelled records (project_label == 'unknown')
    initial_count = len(df)
    valid_df = df[df["project_label"] != "unknown"].copy().reset_index(drop=True)
    if len(valid_df) < initial_count:
        logger.warning(f"Filtered out {initial_count - len(valid_df)} record(s) with 'unknown' project_label.")

    # Multi-attribute stratification key: dataset + label
    valid_df["stratum"] = valid_df["source_dataset"].astype(str) + "::" + valid_df["project_label"].astype(str)

    # For tiny strata (<3 samples), fall back to project_label to allow stratified splitting
    stratum_counts = valid_df["stratum"].value_counts()
    rare_strata = stratum_counts[stratum_counts < 3].index.tolist()
    if rare_strata:
        logger.info(f"Re-grouping {len(rare_strata)} rare strata into project_label to ensure stratification stability.")
        valid_df.loc[valid_df["stratum"].isin(rare_strata), "stratum"] = valid_df["project_label"]

    logger.info(f"Executing 70/15/15 Stratified Split with random_seed={seed}...")

    # Step 1: 70% Train, 30% Temp (Val + Test)
    temp_ratio = val_ratio + test_ratio  # 0.30
    train_df, temp_df = train_test_split(
        valid_df,
        test_size=temp_ratio,
        random_state=seed,
        stratify=valid_df["stratum"],
    )

    # Step 2: Split 30% Temp into 15% Val and 15% Test (50% of Temp each)
    val_temp_ratio = val_ratio / temp_ratio  # 0.50
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_temp_ratio),
        random_state=seed,
        stratify=temp_df["stratum"],
    )

    # Assign split column
    valid_df.loc[train_df.index, "split"] = "train"
    valid_df.loc[val_df.index, "split"] = "validation"
    valid_df.loc[test_df.index, "split"] = "test"

    # Drop intermediate stratum column
    valid_df.drop(columns=["stratum"], inplace=True)

    # -------------------------------------------------------------
    # Rigorous Zero-Leakage Verification Assertions (Rule 3)
    # -------------------------------------------------------------
    logger.info("Executing zero-leakage assertions...")
    train_hashes = set(valid_df[valid_df["split"] == "train"]["source_hash"])
    val_hashes = set(valid_df[valid_df["split"] == "validation"]["source_hash"])
    test_hashes = set(valid_df[valid_df["split"] == "test"]["source_hash"])

    train_val_overlap = len(train_hashes & val_hashes)
    train_test_overlap = len(train_hashes & test_hashes)
    val_test_overlap = len(val_hashes & test_hashes)

    assert train_val_overlap == 0, f"DATA LEAKAGE: {train_val_overlap} overlapping hashes between train and val!"
    assert train_test_overlap == 0, f"DATA LEAKAGE: {train_test_overlap} overlapping hashes between train and test!"
    assert val_test_overlap == 0, f"DATA LEAKAGE: {val_test_overlap} overlapping hashes between val and test!"
    logger.info("Zero-leakage assertions PASSED! Exactly 0 hash overlap across train, validation, and test splits.")

    # Compute deterministic split fingerprint
    sorted_test_ids = sorted(valid_df[valid_df["split"] == "test"]["message_id"].tolist())
    split_hash = hashlib.sha256("".join(sorted_test_ids).encode("utf-8")).hexdigest()[:16]
    logger.info(f"Deterministic Split Fingerprint (split_hash): {split_hash}")

    # Save locked dataset with split assignments
    valid_df.to_parquet(processed_file, index=False, engine="pyarrow")
    logger.info(f"Saved updated dataset with locked splits to {processed_file}")

    # Generate Split Reports
    generate_split_report(valid_df, split_hash, reports_dir)

    return valid_df


def generate_split_report(df: pd.DataFrame, split_hash: str, reports_dir: Path) -> None:
    total = len(df)
    train_cnt = int((df["split"] == "train").sum())
    val_cnt = int((df["split"] == "validation").sum())
    test_cnt = int((df["split"] == "test").sum())

    label_by_split = df.groupby(["project_label", "split"]).size().unstack(fill_value=0).to_dict()
    source_by_split = df.groupby(["source_dataset", "split"]).size().unstack(fill_value=0).to_dict()
    lang_by_split = df.groupby(["language", "split"]).size().unstack(fill_value=0).to_dict()
    channel_by_split = df.groupby(["channel", "split"]).size().unstack(fill_value=0).to_dict()

    report_data = {
        "split_summary": {
            "total_records": total,
            "train_records": train_cnt,
            "train_percentage": round(train_cnt / total * 100, 2),
            "validation_records": val_cnt,
            "validation_percentage": round(val_cnt / total * 100, 2),
            "test_records": test_cnt,
            "test_percentage": round(test_cnt / total * 100, 2),
            "split_fingerprint_sha256": split_hash,
            "random_seed": 42,
        },
        "labels_by_split": label_by_split,
        "source_datasets_by_split": source_by_split,
        "languages_by_split": lang_by_split,
        "channels_by_split": channel_by_split,
    }

    # Save JSON
    json_path = reports_dir / "split_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    # Save Markdown
    md_path = reports_dir / "split_report.md"
    lines = [
        "# ScamShield — Train / Validation / Test Split Report (Phase 5)",
        "",
        "## 1. Split Allocation & Fingerprint",
        f"- **Total Leakage-Free Records:** {total:,}",
        f"- **Train Partition:** {train_cnt:,} records ({round(train_cnt / total * 100, 2)}%)",
        f"- **Validation Partition:** {val_cnt:,} records ({round(val_cnt / total * 100, 2)}%)",
        f"- **Test Partition:** {test_cnt:,} records ({round(test_cnt / total * 100, 2)}%)",
        f"- **Locked Split Fingerprint (split_hash):** `{split_hash}`",
        f"- **Random Seed:** 42",
        "",
        "## 2. Class Label Stratification",
        "| Project Label | Train (70%) | Validation (15%) | Test (15%) | Total |",
        "|---|---|---|---|---|",
    ]

    for lbl in ["benign", "phishing", "spam", "scam"]:
        tr = label_by_split.get("train", {}).get(lbl, 0)
        va = label_by_split.get("validation", {}).get(lbl, 0)
        te = label_by_split.get("test", {}).get(lbl, 0)
        tot = tr + va + te
        lines.append(f"| **{lbl}** | {tr:,} | {va:,} | {te:,} | {tot:,} |")

    lines.extend([
        "",
        "## 3. Source Dataset Representation",
        "| Dataset Source | Train (70%) | Validation (15%) | Test (15%) | Total |",
        "|---|---|---|---|---|",
    ])
    for src in ["meajor", "llm_phishing", "dravidian_sms", "indian_scam"]:
        tr = source_by_split.get("train", {}).get(src, 0)
        va = source_by_split.get("validation", {}).get(src, 0)
        te = source_by_split.get("test", {}).get(src, 0)
        tot = tr + va + te
        lines.append(f"| **{src}** | {tr:,} | {va:,} | {te:,} | {tot:,} |")

    lines.extend([
        "",
        "## 4. Multilingual Representation",
        "| Language | Train (70%) | Validation (15%) | Test (15%) | Total |",
        "|---|---|---|---|---|",
    ])
    for lng in ["en", "hinglish", "te", "hi"]:
        tr = lang_by_split.get("train", {}).get(lng, 0)
        va = lang_by_split.get("validation", {}).get(lng, 0)
        te = lang_by_split.get("test", {}).get(lng, 0)
        tot = tr + va + te
        lines.append(f"| **{lng}** | {tr:,} | {va:,} | {te:,} | {tot:,} |")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    logger.info(f"Saved split reports to {json_path} and {md_path}")


if __name__ == "__main__":
    df_splits = create_splits()
    print("\n" + "=" * 65)
    print("      SCAMSHIELD — DATA SPLITTING COMPLETE & VERIFIED")
    print("=" * 65)
    counts = df_splits["split"].value_counts()
    for s_name, s_count in counts.items():
        print(f"  Split: {s_name:<12} -> {s_count:>7,} records ({s_count/len(df_splits)*100:.2f}%)")
    print("=" * 65)
