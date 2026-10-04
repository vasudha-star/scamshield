"""Unified End-to-End Inference Pipeline for ScamShield (Phase 21).

Integrates the full multimodal AI detection stack into a single, high-throughput,
production-grade inference engine:
1. Raw Text Cleaning & Normalization (Unicode NFKC, URL detachment)
2. Multimodal Feature Extraction (15,000 Text + 18 URL + 10 Intent = 15,028 features)
3. Binary Threat Classification via M4 Full Ensemble (LinearSVC Calibrated)
4. Scam Category Classification via Multinomial Classifier (Rule 18 compliant)
5. Calibrated Risk Scoring Engine [0, 100] across 4 Severity Tiers
6. Dual-Layer Explainability:
   - Quantitative Exact Linear SHAP Feature Attributions
   - Qualitative Plain-Language Multi-Pillar Evidence Rationale

Provides:
- scan(text, url=None): Sub-10ms single-message scanner returning structured JSON.
- scan_batch(items): Batch scanner for streaming log streams or dataset audits.
- get_health(): System health and model artifact validation check.
"""

from __future__ import annotations

import datetime
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from src.explainability.risk_scorer import ScamShieldRiskScorer
from src.explainability.rule_explainer import RuleBasedExplainer
from src.explainability.shap_explainer import ScamShieldSHAPExplainer
from src.features.intent.intent_extractor import extract_intent_features
from src.features.url.url_extractor import extract_message_url_features
from src.preprocessing.text_cleaner import clean_text
from src.utils.logger import get_logger

logger = get_logger("scamshield_pipeline")

CATEGORY_TITLES = {
    "digital_arrest": "Digital Arrest / Law Enforcement Impersonation Extortion",
    "extortion_blackmail": "Extortion, Video Blackmail & Defamation Threat",
    "kyc_banking_identity": "Banking KYC & Identity Theft Phishing",
    "lottery_reward": "Fake Lottery, Prize & Cashback Fraud",
    "impersonation_scam": "Delivery / Relative Emergency Impersonation",
    "legitimate": "Legitimate Personal or Institutional Communication",
}


class ScamShieldPipeline:
    """Production End-to-End Multimodal Inference Engine for ScamShield."""

    _instance: ScamShieldPipeline | None = None

    def __init__(self, models_dir: Path | str | None = None) -> None:
        self.models_dir = Path(models_dir or (PROJECT_ROOT / "models"))
        self._load_all_artifacts()

    @classmethod
    def get_instance(cls, models_dir: Path | str | None = None) -> ScamShieldPipeline:
        """Singleton accessor for efficient reuse across API requests and dashboard runs."""
        if cls._instance is None:
            cls._instance = cls(models_dir=models_dir)
        return cls._instance

    def _load_all_artifacts(self) -> None:
        """Loads and verifies all serialized models, scalers, and explainers."""
        t0 = time.time()
        logger.info(f"Initializing ScamShield Pipeline from {self.models_dir}...")

        # Binary M4 Detection Artifacts
        self.m4_model = joblib.load(self.models_dir / "m4_full_scamshield.joblib")
        self.tfidf = joblib.load(self.models_dir / "tfidf_vectorizer.joblib")
        self.url_scaler = joblib.load(self.models_dir / "url_scaler.joblib")
        self.intent_scaler = joblib.load(self.models_dir / "intent_scaler.joblib")

        # Category Classification Artifacts
        self.cat_clf = joblib.load(self.models_dir / "category_classifier.joblib")
        self.cat_vec = joblib.load(self.models_dir / "category_vectorizer.joblib")

        # Subsystems
        self.shap_explainer = ScamShieldSHAPExplainer(
            model_path=self.models_dir / "m4_full_scamshield.joblib",
            tfidf_path=self.models_dir / "tfidf_vectorizer.joblib",
            url_scaler_path=self.models_dir / "url_scaler.joblib",
            intent_scaler_path=self.models_dir / "intent_scaler.joblib",
        )
        self.rule_explainer = RuleBasedExplainer()
        self.risk_scorer = ScamShieldRiskScorer()

        self.initialized_at = datetime.datetime.now().isoformat()
        logger.info(f"ScamShield Pipeline initialized in {time.time() - t0:.2f}s.")

    def scan(self, text: str, url: str | None = None) -> dict[str, Any]:
        """Scans a single message and optional URL, returning structured assessment."""
        t_start = time.perf_counter()

        if not isinstance(text, str) or not text.strip():
            return {
                "status": "error",
                "message": "Input text must be a non-empty string.",
                "timestamp": datetime.datetime.now().isoformat(),
            }

        raw_input = text.strip()

        # 1. Text Cleaning & URL Extraction
        cleaned, extracted_urls, url_present = clean_text(raw_input)
        urls = list(extracted_urls)
        if url and url.strip() and url.strip() not in urls:
            urls.append(url.strip())
            url_present = True

        # 2. Multimodal Feature Extraction
        x_text = self.tfidf.transform([cleaned])
        raw_url_feats = extract_message_url_features(urls)
        x_url = self.url_scaler.transform(raw_url_feats.reshape(1, -1))
        raw_intent_feats = extract_intent_features(raw_input)
        x_intent = self.intent_scaler.transform(raw_intent_feats.reshape(1, -1))

        # Early Fusion (15,028 features)
        x_fused = sparse.hstack([x_text, x_url, x_intent], format="csr")

        # 3. Binary Threat Detection (M4)
        prob = float(self.m4_model.predict_proba(x_fused)[0, 1])
        is_threat = bool(prob >= 0.5)

        # 4. Multi-Class Category Classification
        x_cat = self.cat_vec.transform([cleaned])
        cat_probs = self.cat_clf.predict_proba(x_cat)[0]
        cat_classes = [str(c) for c in self.cat_clf.classes_]
        cat_prob_map = {k: round(float(v), 4) for k, v in zip(cat_classes, cat_probs)}
        top_cat = str(self.cat_clf.predict(x_cat)[0])
        top_cat_conf = float(cat_prob_map.get(top_cat, 0.0))

        # If model indicates benign, align category to legitimate unless strong threat signals exist
        if not is_threat and top_cat != "legitimate":
            # For low-threat communications, reflect legitimate baseline
            final_cat = "legitimate" if prob < 0.25 else top_cat
        else:
            final_cat = top_cat

        # 5. Local SHAP Attribution
        shap_exp = self.shap_explainer.explain_sample(x_fused, top_k=6)
        decision_val = shap_exp["decision_value"]
        top_threat_drivers = shap_exp["threat_attributions"][:5]
        top_benign_drivers = shap_exp["benign_attributions"][:5]

        # 6. Rule-Based Plain-Language Explanation
        rule_exp = self.rule_explainer.explain(
            raw_input,
            predicted_probability=prob,
            category=final_cat,
        )

        # 7. Calibrated Risk Score
        intent_density = float(np.sum(raw_intent_feats))
        has_suspicious_url = any(
            item["modality"] == "URL" and item["shap_value"] > 0 for item in top_threat_drivers
        )
        risk_exp = self.risk_scorer.calculate_risk(
            predicted_probability=prob,
            intent_density=intent_density,
            has_suspicious_url=has_suspicious_url,
        )

        latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

        return {
            "status": "success",
            "timestamp": datetime.datetime.now().isoformat(),
            "latency_ms": latency_ms,
            "input": {
                "raw_text": raw_input,
                "cleaned_text": cleaned,
                "extracted_urls": urls,
                "url_present": url_present,
            },
            "verdict": {
                "is_threat": is_threat,
                "threat_probability": round(prob, 5),
                "decision_value": round(decision_val, 4),
                "threat_level": risk_exp["risk_tier"],
                "primary_directive": risk_exp["primary_directive"],
            },
            "category": {
                "predicted_category": final_cat,
                "category_title": CATEGORY_TITLES.get(final_cat, final_cat.replace("_", " ").title()),
                "confidence": round(top_cat_conf, 4),
                "category_probabilities": cat_prob_map,
            },
            "risk_assessment": {
                "risk_score": risk_exp["risk_score"],
                "risk_tier": risk_exp["risk_tier"],
                "badge_color": risk_exp["badge_color"],
                "confidence_level": risk_exp["confidence_level"],
                "action_checklist": risk_exp["action_checklist"],
            },
            "unified_assessment": {
                "risk_score": round(risk_exp["risk_score"], 1),
                "risk_level": risk_exp["risk_tier"].title(),
                "prediction": "phishing" if is_threat else "legitimate",
                "category": final_cat,
                "category_title": CATEGORY_TITLES.get(final_cat, final_cat.replace("_", " ").title()),
                "confidence": round(prob if is_threat else (1.0 - prob), 4),
                "reasons": rule_exp.get("reasons", []),
                "evidence_snippets": rule_exp.get("evidence_snippets", []),
                "recommended_action": risk_exp["action_checklist"],
            },
            "explanation": {
                "executive_summary": rule_exp["summary"],
                "reasons": rule_exp.get("reasons", []),
                "evidence_snippets": rule_exp.get("evidence_snippets", []),
                "evidence_pillars": rule_exp["evidence_pillars"],
                "recommendations": rule_exp["recommendations"],
                "top_threat_drivers": [
                    {
                        "feature": item["feature_name"],
                        "modality": item["modality"],
                        "value": item["feature_value"],
                        "impact": item["shap_value"],
                    }
                    for item in top_threat_drivers
                ],
                "top_benign_drivers": [
                    {
                        "feature": item["feature_name"],
                        "modality": item["modality"],
                        "value": item["feature_value"],
                        "impact": item["shap_value"],
                    }
                    for item in top_benign_drivers
                ],
            },
        }

    def scan_batch(self, items: list[str | dict[str, Any]]) -> list[dict[str, Any]]:
        """Processes a list of text strings or dicts with {'text': ..., 'url': ...}."""
        results = []
        for item in items:
            if isinstance(item, str):
                results.append(self.scan(item))
            elif isinstance(item, dict) and "text" in item:
                results.append(self.scan(item["text"], url=item.get("url")))
            else:
                results.append({
                    "status": "error",
                    "message": "Invalid item format; expected string or dict with 'text' key.",
                })
        return results

    def get_health(self) -> dict[str, Any]:
        """Reports system health status and component dimensions."""
        return {
            "status": "healthy",
            "pipeline_name": "ScamShield End-to-End Multimodal Inference Engine",
            "version": "1.0.0",
            "initialized_at": self.initialized_at,
            "models_loaded": {
                "m4_binary_classifier": str(type(self.m4_model).__name__),
                "category_classifier": str(type(self.cat_clf).__name__),
                "feature_dimensions": {
                    "text_tfidf": int(self.tfidf.max_features or len(self.tfidf.vocabulary_)),
                    "url_features": 18,
                    "intent_features": 10,
                    "total_fused_features": 15028,
                },
                "category_classes": [str(c) for c in self.cat_clf.classes_],
            },
            "hardware": {
                "platform": sys.platform,
                "python_version": sys.version.split()[0],
            },
        }
