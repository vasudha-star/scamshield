# ScamShield — Train / Validation / Test Split Report (Phase 5)

## 1. Split Allocation & Fingerprint
- **Total Leakage-Free Records:** 119,339
- **Train Partition:** 83,537 records (70.0%)
- **Validation Partition:** 17,901 records (15.0%)
- **Test Partition:** 17,901 records (15.0%)
- **Locked Split Fingerprint (split_hash):** `85851abd839f4971`
- **Random Seed:** 42

## 2. Class Label Stratification
| Project Label | Train (70%) | Validation (15%) | Test (15%) | Total |
|---|---|---|---|---|
| **benign** | 42,672 | 9,143 | 9,145 | 60,960 |
| **phishing** | 39,747 | 8,518 | 8,517 | 56,782 |
| **spam** | 675 | 145 | 144 | 964 |
| **scam** | 443 | 95 | 95 | 633 |

## 3. Source Dataset Representation
| Dataset Source | Train (70%) | Validation (15%) | Test (15%) | Total |
|---|---|---|---|---|
| **meajor** | 73,351 | 15,719 | 15,718 | 104,788 |
| **llm_phishing** | 6,990 | 1,498 | 1,498 | 9,986 |
| **dravidian_sms** | 2,676 | 573 | 573 | 3,822 |
| **indian_scam** | 520 | 111 | 112 | 743 |

## 4. Multilingual Representation
| Language | Train (70%) | Validation (15%) | Test (15%) | Total |
|---|---|---|---|---|
| **en** | 82,531 | 17,697 | 17,684 | 117,912 |
| **hinglish** | 520 | 111 | 112 | 743 |
| **te** | 419 | 80 | 87 | 586 |
| **hi** | 67 | 13 | 18 | 98 |
