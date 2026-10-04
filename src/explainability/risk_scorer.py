"""Calibrated Risk Scoring Engine for ScamShield (Phase 20).

Maps model threat probability, intent coercion density, and URL risk indicators
into an intuitive, standardized Risk Score [0, 100] across 4 Severity Tiers:
- LOW      [0 - 29]   : Routine legitimate communication
- MEDIUM   [30 - 59]  : Caution / Advisory, verify independently
- HIGH     [60 - 84]  : Probable scam / phishing attack
- CRITICAL [85 - 100] : Severe active attack requiring immediate defense

Computes:
- Calibrated 0-100 score
- Severity tier and badge styling
- Model confidence level (LOW, MEDIUM, HIGH)
- Primary directive and mitigation action checklist
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger

logger = get_logger("risk_scorer")


class ScamShieldRiskScorer:
    """Computes calibrated [0, 100] risk scores and actionable mitigation directives."""

    def __init__(self) -> None:
        pass

    def calculate_risk(
        self,
        predicted_probability: float,
        intent_density: float = 0.0,
        has_suspicious_url: bool = False,
    ) -> dict[str, Any]:
        """Calculates calibrated risk score, tier, confidence, and action items."""
        prob = max(0.0, min(1.0, float(predicted_probability)))

        # Base score from calibrated probability
        score = prob * 100.0

        # Contextual threat modifier
        # If threat probability is above threshold and intense coercive signals exist, apply scaling
        if prob >= 0.5:
            bonus = 0.0
            if intent_density > 0.3:
                bonus += 5.0
            if has_suspicious_url:
                bonus += 5.0
            score = min(100.0, score + bonus)
        elif prob < 0.2:
            # Dampen low probabilities if clean
            score = max(0.0, score)

        score = round(score, 1)

        # Assign Severity Tier (Empirically Calibrated Quartile Tiers)
        if score <= 25.0:
            tier = "LOW"
            badge_color = "#00e676"  # Emerald Green
            primary_directive = "Safe Communication: No immediate risk detected."
            action_checklist = [
                "Communication appears standard and uncompromised.",
                "Exercise routine awareness with unknown contacts.",
            ]
        elif score <= 50.0:
            tier = "MEDIUM"
            badge_color = "#ffb703"  # Amber / Yellow
            primary_directive = "Caution Advised: Unverified links or marketing pressure present."
            action_checklist = [
                "Verify the sender's identity through official independent channels.",
                "Do not click links if the sender's email domain or phone number looks unfamiliar.",
                "Never share sensitive personal or account details.",
            ]
        elif score <= 75.0:
            tier = "HIGH"
            badge_color = "#ff7043"  # Coral / Orange-Red
            primary_directive = "High Risk Scam: Strong social engineering or deceptive link detected."
            action_checklist = [
                "DO NOT CLICK any links, attachments, or embedded buttons.",
                "DO NOT share passwords, OTPs, UPI PINs, or NetBanking credentials.",
                "Block and report the sender on your messaging or email client.",
            ]
        else:
            tier = "CRITICAL"
            badge_color = "#ff2a5f"  # Crimson
            primary_directive = "Immediate Attack Alert: Severe psychological coercion or financial threat."
            action_checklist = [
                "IMMEDIATELY TERMINATE all communication with the sender.",
                "DO NOT TRANSFER ANY MONEY under any circumstances.",
                "Remember: Official agencies (Police, CBI, Courts, RBI) NEVER conduct arrests via video call.",
                "Report immediately to the National Cybercrime Portal (cybercrime.gov.in / Helpline 1930).",
            ]

        # Calculate Confidence Level
        # Confidence is high when probability is far from decision boundary (0.5)
        dist_from_boundary = abs(prob - 0.5)
        if dist_from_boundary >= 0.35:
            confidence = "HIGH"
        elif dist_from_boundary >= 0.15:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        return {
            "risk_score": score,
            "risk_tier": tier,
            "badge_color": badge_color,
            "confidence_level": confidence,
            "distance_from_boundary": round(dist_from_boundary, 4),
            "primary_directive": primary_directive,
            "action_checklist": action_checklist,
        }
