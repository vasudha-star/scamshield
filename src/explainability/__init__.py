"""Explainability and Risk Scoring Subsystem for ScamShield (Phases 19 & 20).

Exports:
- ScamShieldSHAPExplainer: Quantitative global and local Shapley feature attributions.
- RuleBasedExplainer: Qualitative plain-language multi-pillar evidence narratives.
- ScamShieldRiskScorer: Standardized [0, 100] calibrated threat scoring and tier grading.
"""

from src.explainability.risk_scorer import ScamShieldRiskScorer
from src.explainability.rule_explainer import RuleBasedExplainer
from src.explainability.shap_explainer import ScamShieldSHAPExplainer

__all__ = [
    "ScamShieldSHAPExplainer",
    "RuleBasedExplainer",
    "ScamShieldRiskScorer",
]
