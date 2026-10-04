"""M1: Text-Only Baseline Classifier for ScamShield (Phase 7).

Implements:
1. Candidate benchmarking on validation split:
   - Multinomial Naive Bayes
   - Logistic Regression (L2 regularized)
   - Calibrated Linear Support Vector Machine (LinearSVC)
2. Quantitative selection of best baseline based on validation F1 & PR-AUC.
3. Final evaluation on the locked test partition (17,901 samples).
4. Full metrics computation: Precision, Recall, F1, ROC-AUC, PR-AUC, FPR, FNR.
5. Serializes model weights, prediction probabilities, and confusion matrix.
6. Automatically logs results to experiment_log.csv.
"""

from __future__ import annotations

import datetime
import json
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from src.utils.logger import get_logger

logger = get_logger("m1_baseline")


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Calculates all essential detection metrics including FPR and FNR."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, y_prob))

    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = float(auc(rec_curve, prec_curve))

    return {
        "precision": round(precision, 5),
        "recall": round(recall, 5),
        "f1": round(f1, 5),
        "roc_auc": round(roc_auc, 5),
        "pr_auc": round(pr_auc, 5),
        "fpr": round(fpr, 5),
        "fnr": round(fnr, 5),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def plot_confusion_matrix(cm: np.ndarray, out_path: Path, title: str = "M1 Text Baseline Confusion Matrix") -> None:
    """Renders and saves a high-resolution confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    classes = ["Benign (0)", "Threat (1)"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]:,}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=11, fontweight="bold",
            )
    ax.set_ylabel("Ground Truth Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def append_to_experiment_log(
    exp_id: str,
    research_q: str,
    model_name: str,
    feature_set: str,
    split_hash: str,
    train_n: int,
    val_n: int,
    test_n: int,
    metrics: dict[str, float],
    artifacts_path: str,
    notes: str,
) -> None:
    """Appends an empirical record to the central experiment tracking log."""
    log_file = PROJECT_ROOT / "experiment_log.csv"
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    record = (
        f"{exp_id},{timestamp},\"{research_q}\",{model_name},{feature_set},"
        f"{split_hash},{train_n},{val_n},{test_n},42,"
        f"{metrics['precision']},{metrics['recall']},{metrics['f1']},"
        f"{metrics['roc_auc']},{metrics['pr_auc']},{metrics['fpr']},{metrics['fnr']},"
        f"{artifacts_path},\"{notes}\"\n"
    )
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(record)
    logger.info(f"Recorded experiment {exp_id} in {log_file}")


def run_m1_experiment() -> dict[str, Any]:
    exp_dir = PROJECT_ROOT / "experiments" / "M1_text"
    reports_dir = PROJECT_ROOT / "reports"
    models_dir = PROJECT_ROOT / "models"

    exp_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Parquet Data for Labels & IDs
    df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "cleaned.parquet")
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "validation"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    # Binary Threat Mapping: 0 = Benign, 1 = Threat (Phishing / Scam / Spam)
    y_train = (train_df["project_label"] != "benign").astype(int).to_numpy()
    y_val = (val_df["project_label"] != "benign").astype(int).to_numpy()
    y_test = (test_df["project_label"] != "benign").astype(int).to_numpy()

    # 2. Load Sparse Feature Matrices
    X_train = sparse.load_npz(exp_dir / "X_train_text.npz")
    X_val = sparse.load_npz(exp_dir / "X_val_text.npz")
    X_test = sparse.load_npz(exp_dir / "X_test_text.npz")

    logger.info("Benchmarking M1 Candidate Models on Validation Set...")
    candidates = {
        "MultinomialNB": MultinomialNB(alpha=0.1),
        "LogisticRegression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        "CalibratedLinearSVC": CalibratedClassifierCV(LinearSVC(C=0.5, random_state=42, max_iter=2000), cv=3),
    }

    val_benchmarks = {}
    fitted_models = {}

    for name, clf in candidates.items():
        t0 = time.time()
        logger.info(f"Training candidate: {name} on {X_train.shape[0]:,} samples...")
        clf.fit(X_train, y_train)
        fit_time = round(time.time() - t0, 2)

        # Validation prediction
        y_val_pred = clf.predict(X_val)
        y_val_prob = clf.predict_proba(X_val)[:, 1]
        m = calculate_metrics(y_val, y_val_pred, y_val_prob)
        m["fit_time_seconds"] = fit_time
        val_benchmarks[name] = m
        fitted_models[name] = clf
        logger.info(f"Candidate {name} Val F1: {m['f1']} | PR-AUC: {m['pr_auc']} | FPR: {m['fpr']} (trained in {fit_time}s)")

    # 3. Model Selection based strictly on measured Validation F1 & PR-AUC (Rule 7)
    best_model_name = max(val_benchmarks, key=lambda k: val_benchmarks[k]["f1"])
    best_clf = fitted_models[best_model_name]
    logger.info(f"Selected Champion Baseline Model: {best_model_name} (Val F1: {val_benchmarks[best_model_name]['f1']})")

    # 4. Final Evaluation on Locked Test Split
    logger.info("Evaluating selected champion model on LOCKED Test Split...")
    y_test_pred = best_clf.predict(X_test)
    y_test_prob = best_clf.predict_proba(X_test)[:, 1]
    test_metrics = calculate_metrics(y_test, y_test_pred, y_test_prob)
    logger.info(f"M1 Test Metrics: F1={test_metrics['f1']}, PR-AUC={test_metrics['pr_auc']}, Precision={test_metrics['precision']}, Recall={test_metrics['recall']}")

    # 5. Save Model and Predictions
    model_save_path = exp_dir / "m1_model.joblib"
    joblib.dump(best_clf, model_save_path)
    joblib.dump(best_clf, models_dir / "m1_text_baseline.joblib")

    test_preds_df = pd.DataFrame({
        "message_id": test_df["message_id"],
        "project_label": test_df["project_label"],
        "true_threat": y_test,
        "predicted_threat": y_test_pred,
        "predicted_probability": np.round(y_test_prob, 5),
    })
    preds_path = exp_dir / "m1_test_predictions.parquet"
    test_preds_df.to_parquet(preds_path, index=False)

    # 6. Save Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_test_pred)
    cm_plot_path = reports_dir / "M1_confusion_matrix.png"
    plot_confusion_matrix(cm, cm_plot_path, title=f"M1 Text-Only ({best_model_name}) Confusion Matrix")
    logger.info(f"Saved confusion matrix plot to {cm_plot_path}")

    # 7. Save M1 Metrics CSV & JSON
    metrics_summary = {
        "model_id": "M1_text",
        "model_architecture": best_model_name,
        "features": "TF-IDF (1,2 n-grams, 15k features, sublinear tf)",
        "random_seed": 42,
        "candidate_validation_benchmarks": val_benchmarks,
        "final_test_metrics": test_metrics,
    }

    with open(reports_dir / "M1_results.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    # Build CSV summary
    df_metrics = pd.DataFrame([
        {"Split": "Validation", "Model": k, **val_benchmarks[k]} for k in val_benchmarks
    ] + [
        {"Split": "Test", "Model": best_model_name, **test_metrics}
    ])
    df_metrics.to_csv(reports_dir / "M1_metrics.csv", index=False)
    logger.info(f"Saved M1 metrics to {reports_dir / 'M1_metrics.csv'}")

    # 8. Append to experiment_log.csv
    # Read split_hash from split_report.json
    split_report_file = reports_dir / "split_report.json"
    with open(split_report_file, "r", encoding="utf-8") as f:
        s_rep = json.load(f)
    split_hash = s_rep.get("split_summary", {}).get("split_fingerprint_sha256", "85851abd839f4971")

    append_to_experiment_log(
        exp_id="M1_TEXT_BASELINE",
        research_q="Does text-only TF-IDF baseline provide sufficient discrimination for digital scam detection?",
        model_name=best_model_name,
        feature_set="text_tfidf_15k",
        split_hash=split_hash,
        train_n=len(train_df),
        val_n=len(val_df),
        test_n=len(test_df),
        metrics=test_metrics,
        artifacts_path="experiments/M1_text",
        notes="M1 Text-only champion evaluated on locked test split.",
    )

    return metrics_summary


if __name__ == "__main__":
    res = run_m1_experiment()
    print("\n" + "=" * 65)
    print("      SCAMSHIELD — M1 TEXT BASELINE EVALUATION COMPLETE")
    print("=" * 65)
    print(f"Champion Model:       {res['model_architecture']}")
    tm = res["final_test_metrics"]
    print(f"Test Precision:       {tm['precision']}")
    print(f"Test Recall:          {tm['recall']}")
    print(f"Test F1-Score:        {tm['f1']}")
    print(f"Test PR-AUC:          {tm['pr_auc']}")
    print(f"Test ROC-AUC:         {tm['roc_auc']}")
    print(f"Test FPR:             {tm['fpr']}")
    print(f"Test FNR:             {tm['fnr']}")
    print("=" * 65)
