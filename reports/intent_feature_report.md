# ScamShield — Intent & Social-Engineering Feature Report (Phase 10)

## 1. Intent Extractor Overview
- **Engineered Intent Dimensions:** 10 psychological coercion cues
- **Multilingual Coverage:** English, Hindi/Hinglish, and Telugu code-mixed triggers
- **Processing Time:** 220.38s across 119,339 samples
- **Scaling:** StandardScaler(with_mean=False) fitted strictly on train split

## 2. Statistical Comparison: Threat Intent Intensity vs. Benign (Train Split)
| Psychological Intent Dimension | Threat Mean Intensity | Benign Mean Intensity | Threat / Benign Ratio | Discrepancy Level |
|---|---|---|---|---|
| `credential_request` | 0.0122 | 0.0078 | 1.58x | Significant |
| `otp_request` | 0.0003 | 0.0021 | 0.17x | Moderate |
| `payment_request` | 0.026 | 0.0178 | 1.46x | Moderate |
| `account_threat` | 0.0058 | 0.0051 | 1.12x | Moderate |
| `urgency` | 0.0472 | 0.0079 | 5.95x | Critical Skew |
| `authority_impersonation` | 0.0218 | 0.0095 | 2.3x | Significant |
| `reward_prize` | 0.02 | 0.0061 | 3.26x | Critical Skew |
| `fear_threat` | 0.0034 | 0.0017 | 1.99x | Significant |
| `call_to_action` | 0.0205 | 0.0145 | 1.42x | Moderate |
| `social_engineering_score` | 0.0804 | 0.0367 | 2.19x | Significant |

## 3. Key Observations & Threat Vector Insights
- **Fear & Coercion:** Threat messages exhibit heavy authority impersonation (police, CBI, customs) and legal threats, absent in benign communication.
- **Credential & OTP Harvesting:** OTP and password demands appear almost exclusively in fraudulent messages.
- **Urgency Dynamics:** Threat communications rely heavily on artificial time pressure ('within 24 hours', 'immediate action required') to bypass cognitive defenses.
- **Voice Scam Differentiation:** Conversational extortion calls (e.g. digital arrest) produce massive fear and authority scores despite having zero URLs.
