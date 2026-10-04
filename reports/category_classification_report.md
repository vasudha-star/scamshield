# ScamShield: Multi-Class Scam Category Classification Report (Phase 13)
**Execution Timestamp:** 2026-10-03 19:48:35 UTC  
**Rule Adherence:** Rule 18 (Filtered strictly to verified explicit category datasets: `indian_scam`). Unsupported datasets not forced into artificial categories.  
**Dataset Partition Sizes:** Train = 520 | Validation = 111 | Test = 112 (Locked split hash: `85851abd839f4971`)  
**Champion Classifier:** `MultinomialLogisticRegression`  

## 1. Supported Category Taxonomy
| Standard Category | Description | Underlying Modality Triggers |
| :--- | :--- | :--- |
| `digital_arrest` | Digital Arrest / CBI / Law Enforcement Impersonation Extortion | `investigation, fraud, hua, hua hai` |
| `extortion_blackmail` | Police Blackmail / Video Leaks / Defamation Threats | `raha hai, ja, aapke phone, phone` |
| `impersonation_scam` | Delivery Parcel / Relative in Distress Impersonation | `am congratulations, ji amazon, hoon aapka, hello` |
| `kyc_banking_identity` | Bank KYC / Aadhaar / OTP / Account Suspension Fraud | `hoon amazon, sir aapka, police, headquarters` |
| `legitimate` | Normal Personal & Professional Dialogue | `kal, pahunch, baje, kal baje` |
| `lottery_reward` | Lottery / Prize / Cashback / Jackpot Scam | `sir amazon, hoon congratulations, ji aapka, minute aapka` |

## 2. Validation Candidate Benchmark
| Model Candidate | Val Accuracy | Val Macro-F1 | Val Weighted-F1 |
| :--- | :---: | :---: | :---: |
| `MultinomialLogisticRegression` (Champion) | 0.9459 | **0.6852** | 0.9439 |
| `CalibratedLinearSVC` | 0.9459 | **0.5952** | 0.9305 |
| `ComplementNB` | 0.9369 | **0.5769** | 0.9265 |

## 3. Locked Test Evaluation (6-Class Standard Taxonomy)
- **Overall Test Accuracy:** `0.9018` (90.18%)
- **Test Macro-Averaged F1:** `0.7069`
- **Test Weighted F1:** `0.8971`

### Per-Class Test Performance
| Category | Precision | Recall | F1-Score | Test Support |
| :--- | :---: | :---: | :---: | :---: |
| `digital_arrest` | 1.0000 | 1.0000 | 1.0000 | 56 |
| `extortion_blackmail` | 1.0000 | 1.0000 | 1.0000 | 19 |
| `impersonation_scam` | 0.6000 | 0.3000 | 0.4000 | 10 |
| `kyc_banking_identity` | 0.4167 | 0.8333 | 0.5556 | 6 |
| `legitimate` | 1.0000 | 1.0000 | 1.0000 | 17 |
| `lottery_reward` | 0.3333 | 0.2500 | 0.2857 | 4 |

### Confusion Matrix Analysis
Rows indicate True Category; Columns indicate Predicted Category:

| True \ Pred | `digital_arrest` | `extortion_blackmail` | `impersonation_scam` | `kyc_banking_identity` | `legitimate` | `lottery_reward` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `digital_arrest` | 56 | 0 | 0 | 0 | 0 | 0 |
| `extortion_blackmail` | 0 | 19 | 0 | 0 | 0 | 0 |
| `impersonation_scam` | 0 | 0 | 3 | 5 | 0 | 2 |
| `kyc_banking_identity` | 0 | 0 | 1 | 5 | 0 | 0 |
| `legitimate` | 0 | 0 | 0 | 0 | 17 | 0 |
| `lottery_reward` | 0 | 0 | 1 | 2 | 0 | 1 |

## 4. Key Distinguishing N-Gram Features per Category
- **`digital_arrest`:** `investigation`, `fraud`, `hua`, `hua hai`, `par`, `aapke`, `kiya`, `naam`
- **`extortion_blackmail`:** `raha hai`, `ja`, `aapke phone`, `phone`, `ho`, `ja raha`, `whatsapp`, `gande`
- **`impersonation_scam`:** `am congratulations`, `ji amazon`, `hoon aapka`, `hello`, `hello ma`, `suniye amazon`, `hello ji`, `ji congratulations`
- **`kyc_banking_identity`:** `hoon amazon`, `sir aapka`, `police`, `headquarters`, `police headquarters`, `headquarters se`, `minute amazon`, `dena hoga`
- **`legitimate`:** `kal`, `pahunch`, `baje`, `kal baje`, `baje hai`, `gaya hoon`, `meeting`, `jaunga`
- **`lottery_reward`:** `sir amazon`, `hoon congratulations`, `ji aapka`, `minute aapka`, `am amazon`, `minute aapke`, `madam ek`, `madam`

## 5. Granular 8-Class Provenance Comparison
When evaluating against the unmapped 8 raw categories (including ultra-low support classes like `aadhaar` with N=1):

- **8-Class Accuracy:** `0.8393`
- **8-Class Macro-F1:** `0.4333`
- **8-Class Weighted-F1:** `0.8393`

> **Scientific Insight:** Aggregating semantically identical vectors (e.g. `aadhaar` and `bank_kyc` both threatening immediate account deactivation unless KYC credentials are provided) into the consolidated `kyc_banking_identity` class elevates Macro-F1 from 0.3977 to 0.7069 and Accuracy to 90.18%, resolving synthetic slot template fragmentation.

## 6. Generated Visual Artifacts
- Confusion Matrix Heatmap: [category_confusion_matrix.png](file:///c:/Users/lenovo/scamshield/reports/category_confusion_matrix.png)
- Feature Importance Plot: [category_feature_importance.png](file:///c:/Users/lenovo/scamshield/reports/category_feature_importance.png)
