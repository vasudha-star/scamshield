"""Rule-Based Plain-Language Explainer for ScamShield (Phase 19).

Translates multimodal feature activations and lexical signals into natural-language
explanations tailored for end-users, bank compliance officers, and cyber investigators.

Organizes evidence across 3 core pillars:
1. Textual Evidence (Specific threat keywords and lexical signals)
2. URL / Link Structural Evidence (IP host, shortener, entropy, suspicious TLD)
3. Psychological Intent Evidence (Urgency, Authority Impersonation, Coercion, Credential Request)

Produces:
- Plain-language executive summary
- Evidence breakdown by pillar
- Actionable defensive recommendations
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.intent.intent_extractor import extract_intent_features
from src.features.url.url_extractor import extract_single_url_features
from src.preprocessing.text_cleaner import extract_urls
from src.utils.logger import get_logger

logger = get_logger("rule_explainer")

# Curated High-Risk Lexical Keyword Mappings
THREAT_KEYWORDS = {
    "digital_arrest": [
        "digital arrest", "video call", "cbi", "cyber crime", "police", "customs",
        "narcotics", "court", "warrant", "fir", "money laundering", "dcp", "sp",
    ],
    "banking_kyc": [
        "kyc", "pan card", "aadhaar", "bank account", "deactivated", "blocked",
        "suspended", "sbi", "hdfc", "icici", "rbi", "netbanking", "update now",
    ],
    "extortion_blackmail": [
        "compromised", "video", "recording", "webcam", "blackmail", "pay bitcoin",
        "leak", "reputation", "ransom", "intimate", "family", "contacts",
    ],
    "lottery_reward": [
        "lottery", "winner", "won", "crore", "lakhs", "prize", "lucky draw",
        "claim prize", "cash reward", "kbc", "congratulations",
    ],
    "credential_harvest": [
        "otp", "pin", "password", "cvv", "login details", "security code",
        "enter credentials", "verify identity",
    ],
    "urgency_triggers": [
        "immediately", "within 24 hours", "urgent", "2 hours", "today only",
        "expiring", "act now", "last chance", "hurry",
    ],
}


class RuleBasedExplainer:
    """Generates human-readable explanations based on multimodal evidence rules."""

    def __init__(self) -> None:
        pass

    def explain(
        self,
        text: str,
        predicted_probability: float | None = None,
        category: str | None = None,
    ) -> dict[str, Any]:
        """Analyzes text and URL evidence to generate a structured plain-language explanation."""
        text_lower = text.lower()

        # 1. Textual Evidence
        textual_evidence = []
        for group, keywords in THREAT_KEYWORDS.items():
            matched = [kw for kw in keywords if re.search(r"\b" + re.escape(kw) + r"\b", text_lower)]
            if matched:
                group_title = group.replace("_", " ").title()
                textual_evidence.append({
                    "category": group_title,
                    "matched_keywords": matched,
                    "severity": "HIGH" if group in ["digital_arrest", "extortion_blackmail", "credential_harvest"] else "MEDIUM",
                })

        # 2. URL / Link Structural Evidence
        url_evidence = []
        urls = extract_urls(text)
        for u in urls:
            u_feats = extract_single_url_features(u)
            reasons = []
            # Index mapping: 0:url_len, 1:domain_len, 2:path_len, 3:query_len, 4:subdomain_count,
            # 5:has_ip, 6:digits, 7:digit_ratio, 8:specials, 9:has_at, 10:has_double_slash,
            # 11:has_hex, 12:is_https, 13:is_shortener, 14:entropy, 15:keyword_count
            has_ip = len(u_feats) > 5 and u_feats[5] == 1.0
            is_shortener = len(u_feats) > 13 and u_feats[13] == 1.0
            subdomain_count = u_feats[4] if len(u_feats) > 4 else 0.0
            entropy = u_feats[14] if len(u_feats) > 14 else 0.0

            if has_ip:
                reasons.append("URL uses a raw IP address instead of a legitimate domain name.")
            if is_shortener:
                reasons.append("URL uses a link shortener (obscuring the final destination).")
            if subdomain_count >= 2:
                reasons.append(f"URL contains excessive subdomains ({int(subdomain_count)} levels).")
            if entropy > 4.2:
                reasons.append(f"URL exhibits abnormally high character entropy ({entropy:.2f}).")
            if any(tld in u.lower() for tld in [".xyz", ".top", ".work", ".buzz", ".tk", ".ml"]):
                reasons.append("URL uses a high-risk or unusual top-level domain.")

            if reasons:
                url_evidence.append({
                    "url": u,
                    "structural_red_flags": reasons,
                    "severity": "HIGH" if (has_ip or is_shortener) else "MEDIUM",
                })

        # 3. Psychological Intent Evidence
        intent_evidence = []
        raw_intents = extract_intent_features(text)
        intent_names = [
            "credential_request", "otp_request", "payment_request", "account_threat",
            "urgency", "authority_impersonation", "reward_prize", "fear_threat",
            "call_to_action", "social_engineering_score",
        ]
        intent_dict = dict(zip(intent_names, raw_intents))

        if intent_dict.get("urgency", 0.0) > 0.1:
            intent_evidence.append({
                "intent": "Urgency & Artificial Time Pressure",
                "score": round(float(intent_dict["urgency"]), 3),
                "description": "The message creates artificial panic demanding immediate action before verification.",
            })
        if intent_dict.get("authority_impersonation", 0.0) > 0.1:
            intent_evidence.append({
                "intent": "Authority Impersonation",
                "score": round(float(intent_dict["authority_impersonation"]), 3),
                "description": "The message falsely cites law enforcement, regulators, or government departments.",
            })
        if intent_dict.get("credential_request", 0.0) > 0.1:
            intent_evidence.append({
                "intent": "Credential Solicitation",
                "score": round(float(intent_dict["credential_request"]), 3),
                "description": "The message directly or indirectly solicits passwords, PINs, or credentials.",
            })
        if intent_dict.get("fear_threat", 0.0) > 0.1:
            intent_evidence.append({
                "intent": "Fear & Legal Coercion",
                "score": round(float(intent_dict["fear_threat"]), 3),
                "description": "The message uses severe threats of arrest, prosecution, or financial penalties.",
            })
        if intent_dict.get("reward_prize", 0.0) > 0.1:
            intent_evidence.append({
                "intent": "Unrealistic Financial Lure",
                "score": round(float(intent_dict["reward_prize"]), 3),
                "description": "The message promises unverified cash prizes, lotteries, or rewards.",
            })
        if intent_dict.get("payment_request", 0.0) > 0.1:
            intent_evidence.append({
                "intent": "Payment Transfer Demand",
                "score": round(float(intent_dict["payment_request"]), 3),
                "description": "The message demands wire transfers, UPI deposits, or processing clearance fees.",
            })

        # 4. Formulate Plain-Language Executive Summary
        summary = self._formulate_summary(
            predicted_probability,
            category,
            textual_evidence,
            url_evidence,
            intent_evidence,
        )

        # 5. Formulate Actionable Safety Recommendations
        recommendations = self._formulate_recommendations(
            textual_evidence,
            url_evidence,
            intent_evidence,
        )

        # 6. Extract dynamic evidence snippets directly from input text (Evidence Grounding)
        evidence_snippets: list[str] = []
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
        for text_ev in textual_evidence:
            for kw in text_ev["matched_keywords"]:
                for s in sentences:
                    if re.search(r"\b" + re.escape(kw) + r"\b", s.lower()):
                        if s not in evidence_snippets:
                            evidence_snippets.append(s)
                            if len(evidence_snippets) >= 4:
                                break

        for u_ev in url_evidence:
            if u_ev["url"] not in evidence_snippets:
                evidence_snippets.append(u_ev["url"])

        # 7. Synthesize concise main reasons
        reasons: list[str] = []
        for i_ev in intent_evidence:
            reasons.append(f"{i_ev['intent']} detected: {i_ev['description']}")
        for u_ev in url_evidence:
            for r in u_ev["structural_red_flags"]:
                if r not in reasons:
                    reasons.append(r)
        if not reasons:
            reasons.append("Message conforms to standard legitimate communication patterns without coercive pressure.")

        return {
            "summary": summary,
            "reasons": reasons,
            "evidence_snippets": evidence_snippets,
            "evidence_pillars": {
                "textual_evidence": textual_evidence,
                "url_structural_evidence": url_evidence,
                "psychological_intent_evidence": intent_evidence,
            },
            "recommendations": recommendations,
        }

    def _formulate_summary(
        self,
        prob: float | None,
        category: str | None,
        text_ev: list[dict[str, Any]],
        url_ev: list[dict[str, Any]],
        intent_ev: list[dict[str, Any]],
    ) -> str:
        """Constructs a concise 1-2 sentence executive assessment."""
        is_threat = (prob >= 0.5) if prob is not None else (len(text_ev) > 0 and len(intent_ev) > 0)

        if not is_threat:
            if text_ev or intent_ev:
                return (
                    "Verified Legitimate Communication (Hard Negative): While security terms "
                    "(e.g., OTP, login, or verification) were detected, they appear in legitimate "
                    "banking or advisory context. No coercive extortion or malicious redirect links were found."
                )
            return (
                "Verified Communication: No coercive psychological patterns, "
                "deceptive authority impersonation, or suspicious URL structures were detected."
            )

        # Highlight key drivers
        drivers = []
        if any(i["intent"] == "Authority Impersonation" for i in intent_ev):
            drivers.append("alleged law enforcement/official authority")
        if any(i["intent"] == "Urgency & Artificial Time Pressure" for i in intent_ev):
            drivers.append("severe time-pressure urgency")
        if any(i["intent"] == "Credential Solicitation" for i in intent_ev):
            drivers.append("credential harvesting")
        if any(i["intent"] == "Payment Transfer Demand" for i in intent_ev):
            drivers.append("demands for financial transfer")
        if url_ev:
            drivers.append("suspicious link structures")

        if not drivers and text_ev:
            drivers.append(f"high-risk keywords ({', '.join(text_ev[0]['matched_keywords'][:2])})")

        if category and category != "legitimate":
            cat_prefix = f"Potential [{category.replace('_', ' ').title()}] Attack: "
        else:
            cat_prefix = "High-Risk Threat Detected: "

        if drivers:
            summary = (
                f"{cat_prefix}This message exhibits {', '.join(drivers[:3])}. "
                "The psychological coercion pattern is characteristic of social engineering attacks."
            )
        else:
            summary = f"{cat_prefix}Multiple deceptive threat indicators detected across message content."

        return summary

    def _formulate_recommendations(
        self,
        text_ev: list[dict[str, Any]],
        url_ev: list[dict[str, Any]],
        intent_ev: list[dict[str, Any]],
    ) -> list[str]:
        """Provides tailored, practical safety instructions."""
        recs = []

        if any(i["intent"] == "Authority Impersonation" for i in intent_ev):
            recs.append("Remember: Government and law enforcement agencies never conduct arrests or interrogations over Skype/WhatsApp video calls.")
        if any(i["intent"] == "Credential Solicitation" for i in intent_ev) or any(t["category"] == "Credential Harvest" for t in text_ev):
            recs.append("Never enter your passwords, NetBanking PINs, or card CVVs into non-official web forms.")
        if any(i["intent"] == "Payment Transfer Demand" for i in intent_ev):
            recs.append("Do not transfer funds to 'clear charges' or 'secure accounts'. Legitimate banks never ask customers to transfer money to personal accounts.")
        if url_ev:
            recs.append("Do NOT click or open any links in this message. Visit the official organization portal by typing the URL manually in your browser.")

        recs.append("If you suspect fraud, report immediately to the National Cybercrime Portal (cybercrime.gov.in or helpline 1930).")
        return recs
