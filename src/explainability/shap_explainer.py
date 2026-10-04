"""SHAP and Linear Attribution Explainer for ScamShield (Phase 19).

Computes exact global and local Shapley values and feature attributions for M4
(Full ScamShield Multimodal Classifier: 15,000 Text + 18 URL + 10 Intent = 15,028 features).

Mathematical Foundation:
For linear models f(x) = sum(w_j * x_j) + b, the exact Shapley value for feature j
relative to a baseline reference is:
    phi_j(x) = w_j * (x_j - E[x_j])

Provides:
- Exact O(D) Shapley computation with zero Monte Carlo approximation error
- Global multimodal feature importance ranking (Text, URL, Intent)
- Local sample attributions (threat vs. benign pushing forces)
- Publication-quality global summary and local waterfall plots
"""

from __future__ import annotations

import sys
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

from src.features.intent.intent_extractor import extract_intent_features
from src.features.url.url_extractor import extract_message_url_features
from src.preprocessing.text_cleaner import clean_text
from src.utils.logger import get_logger

logger = get_logger("shap_explainer")

URL_FEATURE_NAMES = [
    "url_length",
    "domain_length",
    "path_length",
    "num_dots",
    "num_hyphens",
    "num_underscores",
    "num_slashes",
    "num_question_marks",
    "num_equal_signs",
    "num_at_symbols",
    "num_digits",
    "digit_ratio",
    "has_ip",
    "num_subdomains",
    "has_suspicious_tld",
    "has_shortener",
    "shannon_entropy",
    "has_redirect_tokens",
]

INTENT_FEATURE_NAMES = [
    "credential_request",
    "otp_request",
    "payment_request",
    "account_threat",
    "urgency",
    "authority_impersonation",
    "reward_prize",
    "fear_threat",
    "call_to_action",
    "social_engineering_score",
]


class ScamShieldSHAPExplainer:
    """Computes exact Shapley feature attributions for the M4 Multimodal Ensemble."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        tfidf_path: Path | str | None = None,
        url_scaler_path: Path | str | None = None,
        intent_scaler_path: Path | str | None = None,
    ) -> None:
        models_dir = PROJECT_ROOT / "models"
        self.model_path = Path(model_path or (models_dir / "m4_full_scamshield.joblib"))
        self.tfidf_path = Path(tfidf_path or (models_dir / "tfidf_vectorizer.joblib"))
        self.url_scaler_path = Path(url_scaler_path or (models_dir / "url_scaler.joblib"))
        self.intent_scaler_path = Path(intent_scaler_path or (models_dir / "intent_scaler.joblib"))

        self._load_artifacts()
        self._build_feature_names()
        self._compute_ensemble_coefficients()

    def _load_artifacts(self) -> None:
        """Loads model and preprocessing scalers/vectorizers."""
        logger.info("Loading M4 model and feature transformers...")
        self.model = joblib.load(self.model_path)
        self.tfidf = joblib.load(self.tfidf_path)
        self.url_scaler = joblib.load(self.url_scaler_path)
        self.intent_scaler = joblib.load(self.intent_scaler_path)

    def _build_feature_names(self) -> None:
        """Constructs unified 15,028 feature names."""
        self.text_feature_names = self.tfidf.get_feature_names_out().tolist()
        self.url_feature_prefixed = [f"URL::{f}" for f in URL_FEATURE_NAMES]
        self.intent_feature_prefixed = [f"INTENT::{f}" for f in INTENT_FEATURE_NAMES]

        self.all_feature_names = (
            self.text_feature_names
            + self.url_feature_prefixed
            + self.intent_feature_prefixed
        )
        self.n_features = len(self.all_feature_names)
        assert self.n_features == 15028, f"Expected 15,028 features, got {self.n_features}"

    def _compute_ensemble_coefficients(self) -> None:
        """Computes average weights across the 3 CalibratedClassifierCV folds."""
        if hasattr(self.model, "calibrated_classifiers_"):
            coef_list = [clf.estimator.coef_[0] for clf in self.model.calibrated_classifiers_]
            intercept_list = [clf.estimator.intercept_[0] for clf in self.model.calibrated_classifiers_]
            self.weights = np.mean(coef_list, axis=0)
            self.intercept = float(np.mean(intercept_list))
        elif hasattr(self.model, "coef_"):
            self.weights = self.model.coef_[0]
            self.intercept = float(self.model.intercept_[0])
        else:
            raise AttributeError("Unsupported model object for linear attribution.")

    def transform_raw_input(self, text: str, url: str | None = None) -> sparse.csr_matrix:
        """Extracts and concatenates Text, URL, and Intent features from raw text and optional URL."""
        cleaned, extracted_urls, _ = clean_text(text)
        urls = list(extracted_urls)
        if url and url not in urls:
            urls.append(url)

        x_text = self.tfidf.transform([cleaned])

        raw_url_feats = extract_message_url_features(urls)
        x_url = self.url_scaler.transform(raw_url_feats.reshape(1, -1))

        raw_intent_feats = extract_intent_features(text)
        x_intent = self.intent_scaler.transform(raw_intent_feats.reshape(1, -1))

        x_fused = sparse.hstack([x_text, x_url, x_intent], format="csr")
        return x_fused

    def get_global_feature_importance(self, top_n: int = 30) -> dict[str, Any]:
        """Returns top positive (threat) and top negative (benign) features globally."""
        sorted_indices = np.argsort(self.weights)
        top_benign_idx = sorted_indices[:top_n]
        top_threat_idx = sorted_indices[::-1][:top_n]

        def _format_items(indices: np.ndarray) -> list[dict[str, Any]]:
            items = []
            for idx in indices:
                fname = self.all_feature_names[idx]
                w = float(self.weights[idx])
                modality = "TEXT"
                if fname.startswith("URL::"):
                    modality = "URL"
                elif fname.startswith("INTENT::"):
                    modality = "INTENT"

                items.append({
                    "feature_index": int(idx),
                    "feature_name": fname,
                    "weight": round(w, 5),
                    "modality": modality,
                })
            return items

        return {
            "top_threat_features": _format_items(top_threat_idx),
            "top_benign_features": _format_items(top_benign_idx),
            "intercept": round(self.intercept, 5),
            "total_features": self.n_features,
        }

    def explain_sample(
        self,
        X_sample: sparse.csr_matrix | np.ndarray,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Computes exact local Shapley attributions for a single sample vector."""
        if sparse.issparse(X_sample):
            x_dense = X_sample.toarray().flatten()
        else:
            x_dense = np.asarray(X_sample).flatten()

        # Exact linear Shapley values (relative to zero baseline)
        attributions = self.weights * x_dense
        decision_val = float(np.sum(attributions) + self.intercept)

        # Get non-zero contributors
        non_zero_mask = x_dense != 0.0
        active_indices = np.where(non_zero_mask)[0]

        threat_contributors = []
        benign_contributors = []

        for idx in active_indices:
            attr = float(attributions[idx])
            val = float(x_dense[idx])
            fname = self.all_feature_names[idx]
            modality = "TEXT"
            if fname.startswith("URL::"):
                modality = "URL"
            elif fname.startswith("INTENT::"):
                modality = "INTENT"

            entry = {
                "feature_name": fname,
                "feature_value": round(val, 4),
                "shap_value": round(attr, 5),
                "modality": modality,
            }
            if attr > 0:
                threat_contributors.append(entry)
            elif attr < 0:
                benign_contributors.append(entry)

        # Sort threat contributors descending (most positive impact)
        threat_contributors.sort(key=lambda x: x["shap_value"], reverse=True)
        # Sort benign contributors ascending (most negative impact)
        benign_contributors.sort(key=lambda x: x["shap_value"])

        # Predict probability from model if possible
        if sparse.issparse(X_sample):
            prob = float(self.model.predict_proba(X_sample)[0, 1])
        else:
            prob = float(self.model.predict_proba(sparse.csr_matrix(X_sample))[0, 1])

        return {
            "predicted_probability": round(prob, 5),
            "decision_value": round(decision_val, 5),
            "base_value": round(self.intercept, 5),
            "threat_attributions": threat_contributors[:top_k],
            "benign_attributions": benign_contributors[:top_k],
            "total_active_features": len(active_indices),
        }

    def explain_text(self, text: str, top_k: int = 10) -> dict[str, Any]:
        """Convenience method: transforms text and computes local SHAP attributions."""
        X_fused = self.transform_raw_input(text)
        exp = self.explain_sample(X_fused, top_k=top_k)
        exp["raw_text"] = text
        return exp

    def plot_global_feature_importance(
        self,
        out_path: Path | str,
        top_n: int = 20,
    ) -> None:
        """Plots top positive and negative features with modality color coding."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        global_info = self.get_global_feature_importance(top_n=top_n)
        threat_feats = global_info["top_threat_features"][:top_n][::-1]
        benign_feats = global_info["top_benign_features"][:top_n]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

        # Modality colors
        mod_colors = {
            "TEXT": "#2b5c8f",     # Deep blue
            "URL": "#d95f02",      # Crimson orange
            "INTENT": "#7570b3",   # Violet
        }

        # Plot Threat Indicators
        names1 = [f["feature_name"] for f in threat_feats]
        weights1 = [f["weight"] for f in threat_feats]
        colors1 = [mod_colors[f["modality"]] for f in threat_feats]

        ax1.barh(names1, weights1, color=colors1, edgecolor="black", alpha=0.9)
        ax1.set_title(f"Top {top_n} Threat Indicators (Scam Drivers)", fontsize=13, fontweight="bold")
        ax1.set_xlabel("Linear Model Weight (+ Threat)", fontsize=11, fontweight="bold")
        for i, v in enumerate(weights1):
            ax1.text(v + 0.05, i, f"{v:+.2f}", va="center", fontsize=9, fontweight="bold")

        # Plot Benign Indicators
        names2 = [f["feature_name"] for f in benign_feats]
        weights2 = [f["weight"] for f in benign_feats]
        colors2 = [mod_colors[f["modality"]] for f in benign_feats]

        ax2.barh(names2, weights2, color=colors2, edgecolor="black", alpha=0.9)
        ax2.set_title(f"Top {top_n} Benign Indicators (Legitimate Drivers)", fontsize=13, fontweight="bold")
        ax2.set_xlabel("Linear Model Weight (- Benign)", fontsize=11, fontweight="bold")
        for i, v in enumerate(weights2):
            ax2.text(v - 0.25, i, f"{v:+.2f}", va="center", fontsize=9, fontweight="bold")

        # Modality Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=mod_colors["TEXT"], edgecolor="black", label="Text (TF-IDF)"),
            Patch(facecolor=mod_colors["URL"], edgecolor="black", label="URL Structural"),
            Patch(facecolor=mod_colors["INTENT"], edgecolor="black", label="Intent / Psychological"),
        ]
        fig.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=3, fontsize=11)

        plt.suptitle("ScamShield (M4) Global Multimodal Feature Attributions", fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved global feature importance plot to {out_path}")

    def plot_local_waterfall(
        self,
        explanation: dict[str, Any],
        title: str,
        out_path: Path | str,
        top_k: int = 8,
    ) -> None:
        """Plots a local contribution bar chart for an individual prediction."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        threats = explanation["threat_attributions"][:top_k]
        benigns = explanation["benign_attributions"][:top_k]

        combined = threats + benigns
        if not combined:
            logger.warning("No active features to plot in local waterfall.")
            return

        # Sort by absolute SHAP value
        combined.sort(key=lambda x: abs(x["shap_value"]))

        feature_labels = [f"{x['feature_name']} = {x['feature_value']}" for x in combined]
        shap_vals = [x["shap_value"] for x in combined]
        colors = ["#d95f02" if v > 0 else "#2b5c8f" for v in shap_vals]

        fig, ax = plt.subplots(figsize=(10, max(5, len(combined) * 0.45)))
        ax.barh(feature_labels, shap_vals, color=colors, edgecolor="black", alpha=0.85)

        ax.axvline(0, color="gray", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Local SHAP Attribution ($\phi_j$ Impact on Threat Score)", fontsize=11, fontweight="bold")
        ax.set_title(
            f"{title}\nPredicted Threat Probability: {explanation['predicted_probability']*100:.1f}% "
            f"(Decision Value: {explanation['decision_value']:+.2f})",
            fontsize=13,
            fontweight="bold",
            pad=12,
        )

        for i, v in enumerate(shap_vals):
            offset = 0.02 if v > 0 else -0.02
            ha = "left" if v > 0 else "right"
            ax.text(v + offset, i, f"{v:+.3f}", va="center", ha=ha, fontsize=9, fontweight="bold")

        plt.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved local attribution plot to {out_path}")
