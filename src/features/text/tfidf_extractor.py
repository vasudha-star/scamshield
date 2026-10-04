"""TF-IDF Text Feature Engineering Pipeline for ScamShield (Phase 6).

Implements:
1. Sublinear TF-IDF vectorization with word unigrams and bigrams (1, 2).
2. Preserves scam vocabulary cues (urgency, verification, monetary prompts).
3. Strict leakage prevention: fits vectorizer ONLY on the train split;
   transforms validation and test splits using fitted parameters.
4. Serializes fitted vectorizer to models/tfidf_vectorizer.joblib and
   experiments/M1_text/tfidf_vectorizer.joblib.
5. Saves feature matrices and vocabulary analytics to reports/text_feature_report.json
   and reports/text_feature_report.md.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils.logger import get_logger

logger = get_logger("text_features")


def get_vectorizer(
    max_features: int = 15000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 3,
    sublinear_tf: bool = True,
) -> TfidfVectorizer:
    """Configures the ScamShield text vectorizer.

    Args:
        max_features: Maximum vocabulary size to prevent overfitting and high dimensionality.
        ngram_range: (1, 2) captures single tokens and compound phrases ('verify account', 'click here').
        min_df: Prunes ultra-rare typo tokens appearing in fewer than 3 documents.
        sublinear_tf: Uses 1 + log(tf) scaling to dampen the dominance of excessively long emails.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
        strip_accents="unicode",
        lowercase=True,
        stop_words="english",
        token_pattern=r"(?u)\b\w[\w-]+\b",
    )


def extract_top_features_per_class(
    X_train: sparse.csr_matrix,
    y_train: pd.Series,
    feature_names: np.ndarray,
    top_n: int = 15,
) -> dict[str, list[dict[str, Any]]]:
    """Computes mean TF-IDF weights per class to identify top distinctive vocabulary."""
    classes = y_train.unique()
    top_per_class = {}

    for c in classes:
        idx = (y_train == c).to_numpy().nonzero()[0]
        if len(idx) == 0:
            continue
        mean_weights = np.asarray(X_train[idx].mean(axis=0)).flatten()
        top_indices = np.argsort(mean_weights)[::-1][:top_n]
        top_per_class[str(c)] = [
            {"term": str(feature_names[i]), "mean_tfidf": round(float(mean_weights[i]), 5)}
            for i in top_indices
        ]
    return top_per_class


def run_text_feature_pipeline() -> dict[str, Any]:
    start_time = time.time()
    data_path = PROJECT_ROOT / "data" / "processed" / "cleaned.parquet"
    models_dir = PROJECT_ROOT / "models"
    exp_dir = PROJECT_ROOT / "experiments" / "M1_text"
    reports_dir = PROJECT_ROOT / "reports"

    models_dir.mkdir(parents=True, exist_ok=True)
    exp_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading cleaned dataset from {data_path}...")
    df = pd.read_parquet(data_path)

    # Segregate splits
    train_mask = df["split"] == "train"
    val_mask = df["split"] == "validation"
    test_mask = df["split"] == "test"

    train_df = df[train_mask].reset_index(drop=True)
    val_df = df[val_mask].reset_index(drop=True)
    test_df = df[test_mask].reset_index(drop=True)

    logger.info(f"Splits loaded: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")

    # Build and fit vectorizer STRICTLY on train texts (Rule 3: No Leakage)
    vectorizer = get_vectorizer(max_features=15000, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    logger.info("Fitting TfidfVectorizer STRICTLY on Train partition...")
    fit_start = time.time()
    X_train = vectorizer.fit_transform(train_df["text"].fillna(""))
    fit_duration = round(time.time() - fit_start, 2)
    logger.info(f"Vectorizer fit completed in {fit_duration}s. Vocabulary size: {len(vectorizer.vocabulary_):,}")

    # Transform validation and test sets (never fit on them!)
    logger.info("Transforming validation and test partitions with fitted vectorizer...")
    X_val = vectorizer.transform(val_df["text"].fillna(""))
    X_test = vectorizer.transform(test_df["text"].fillna(""))

    feature_names = np.array(vectorizer.get_feature_names_out())
    sparsity_train = 100.0 * (1.0 - (X_train.nnz / (X_train.shape[0] * X_train.shape[1])))

    # Compute top characteristic terms per class on train set
    top_terms = extract_top_features_per_class(
        X_train=X_train,
        y_train=train_df["project_label"],
        feature_names=feature_names,
        top_n=15,
    )

    # Save fitted vectorizer
    vec_path_models = models_dir / "tfidf_vectorizer.joblib"
    vec_path_exp = exp_dir / "tfidf_vectorizer.joblib"
    joblib.dump(vectorizer, vec_path_models)
    joblib.dump(vectorizer, vec_path_exp)
    logger.info(f"Saved fitted TfidfVectorizer to {vec_path_models} and {vec_path_exp}")

    # Save matrices in experiments/M1_text/ for M1 training in Phase 7
    sparse.save_npz(exp_dir / "X_train_text.npz", X_train)
    sparse.save_npz(exp_dir / "X_val_text.npz", X_val)
    sparse.save_npz(exp_dir / "X_test_text.npz", X_test)
    logger.info(f"Saved sparse text feature matrices to {exp_dir}")

    total_time = round(time.time() - start_time, 2)

    metrics = {
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
        "ngram_range": [1, 2],
        "max_features": 15000,
        "sublinear_tf": True,
        "fit_time_seconds": fit_duration,
        "total_pipeline_time_seconds": total_time,
        "matrices": {
            "train_shape": [int(X_train.shape[0]), int(X_train.shape[1])],
            "train_non_zeros": int(X_train.nnz),
            "train_sparsity_pct": round(float(sparsity_train), 4),
            "val_shape": [int(X_val.shape[0]), int(X_val.shape[1])],
            "val_non_zeros": int(X_val.nnz),
            "test_shape": [int(X_test.shape[0]), int(X_test.shape[1])],
            "test_non_zeros": int(X_test.nnz),
        },
        "top_features_by_class": top_terms,
    }

    # Save JSON Report
    json_path = reports_dir / "text_feature_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save Markdown Report
    md_path = reports_dir / "text_feature_report.md"
    generate_markdown_report(metrics, md_path)

    return metrics


def generate_markdown_report(metrics: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# ScamShield — Text Feature Engineering Report (Phase 6)",
        "",
        "## 1. Feature Extraction & Vectorizer Specifications",
        f"- **Vocabulary Size:** {metrics['vocabulary_size']:,} n-grams",
        f"- **N-gram Range:** Unigrams + Bigrams (1, 2)",
        f"- **Sublinear TF Scaling:** Enabled ($1 + \\log(\\text{{tf}})$)",
        f"- **Train Matrix Dimensions:** {metrics['matrices']['train_shape'][0]:,} samples × {metrics['matrices']['train_shape'][1]:,} features",
        f"- **Matrix Sparsity:** {metrics['matrices']['train_sparsity_pct']}%",
        f"- **Vectorization Time:** {metrics['fit_time_seconds']}s (Fit on train)",
        "",
        "## 2. Top Discriminative N-grams by Project Label (Train Partition)",
        "",
    ]

    for label_name, terms in metrics["top_features_by_class"].items():
        lines.extend([
            f"### Class: `{label_name}`",
            "| Rank | N-gram Token | Mean TF-IDF Weight |",
            "|---|---|---|",
        ])
        for idx, item in enumerate(terms, start=1):
            lines.append(f"| {idx} | `{item['term']}` | {item['mean_tfidf']} |")
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    report = run_text_feature_pipeline()
    print("\n" + "=" * 65)
    print("      SCAMSHIELD — TEXT FEATURE EXTRACTION COMPLETE")
    print("=" * 65)
    print(f"Vocabulary Size:     {report['vocabulary_size']:,} n-grams")
    print(f"Train Matrix Shape:  {report['matrices']['train_shape']}")
    print(f"Val Matrix Shape:    {report['matrices']['val_shape']}")
    print(f"Test Matrix Shape:   {report['matrices']['test_shape']}")
    print(f"Train Sparsity:      {report['matrices']['train_sparsity_pct']}%")
    print(f"Vectorizer Saved:    models/tfidf_vectorizer.joblib")
    print(f"Report:              reports/text_feature_report.md")
    print("=" * 65)
