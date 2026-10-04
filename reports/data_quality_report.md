# ScamShield — Data Quality & Standardization Report (Phase 3)

## 1. Deduplication & Volume Summary
- **Raw Ingested Records:** 133,238
- **Clean Deduplicated Records:** 119,340
- **Dropped Duplicates (Exact SHA-256 match):** 13,898 (10.43%)

## 2. Canonical Distributions

### Records by Source Dataset
| Dataset | Clean Records |
|---|---|
| **meajor** | 104,789 |
| **llm_phishing** | 9,986 |
| **dravidian_sms** | 3,822 |
| **indian_scam** | 743 |

### Project Labels
| Project Label | Count |
|---|---|
| **benign** | 60,960 |
| **phishing** | 56,782 |
| **spam** | 964 |
| **scam** | 633 |
| **unknown** | 1 |

### Language Breakdown
| Language | Count |
|---|---|
| **en** | 117,913 |
| **hinglish** | 743 |
| **te** | 586 |
| **hi** | 98 |

### Channel Distribution
| Channel | Count |
|---|---|
| **email** | 114,775 |
| **sms** | 3,822 |
| **transcript** | 743 |

- **Overall URL Presence Rate:** 57.62%
- **Natural vs Synthetic:** 113,611 Natural / 5,729 Synthetic
