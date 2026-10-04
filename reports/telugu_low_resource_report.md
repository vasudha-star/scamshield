# ScamShield: Low-Resource Telugu Adaptation Report (Phase 15)
**Execution Timestamp:** 2026-10-03 20:11:16 UTC  
**Evaluation Split Hash:** `85851abd839f4971`  
**Target Test Partition:** Locked Telugu SMS Test Partition (`N = 87`, 10 Threats, 77 Benign)  

## 1. Monolingual Telugu Data-Efficiency Learning Curve
| Target N | Actual Train N | Threats | Benign | Vocabulary | Precision | Recall | F1-Score | Accuracy |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `25` | 25 | 3 | 22 | 340 | `1.0000` | `0.8000` | **`0.8889`** | `0.9770` |
| `50` | 50 | 5 | 45 | 561 | `1.0000` | `0.8000` | **`0.8889`** | `0.9770` |
| `100` | 100 | 11 | 89 | 1,021 | `1.0000` | `0.7000` | **`0.8235`** | `0.9655` |
| `200` | 200 | 21 | 179 | 1,719 | `1.0000` | `0.9000` | **`0.9474`** | `0.9885` |
| `300` | 300 | 32 | 268 | 2,141 | `1.0000` | `1.0000` | **`1.0000`** | `1.0000` |
| `419` | 419 | 45 | 374 | 2,586 | `1.0000` | `1.0000` | **`1.0000`** | `1.0000` |

## 2. Global Cross-Lingual Threat Scaling on Telugu Evaluation
| Global Pool N | Telugu in Train | Global Vocab | Precision | Recall | F1-Score | Accuracy |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `100` | 0 | 15,000 | `0.1739` | `0.8000` | **`0.2857`** | `0.5402` |
| `250` | 0 | 15,000 | `0.2381` | `1.0000` | **`0.3846`** | `0.6322` |
| `500` | 1 | 15,000 | `0.2250` | `0.9000` | **`0.3600`** | `0.6322` |
| `1,000` | 5 | 15,000 | `0.2222` | `0.8000` | **`0.3478`** | `0.6552` |
| `2,000` | 10 | 15,000 | `0.2222` | `0.8000` | **`0.3478`** | `0.6552` |
| `83,537` | 419 | 15,000 | `0.9000` | `0.9000` | **`0.9000`** | `0.9770` |

## 3. Key Scientific Insights
1. **High Sample Efficiency of Monolingual Telugu:** Even with as few as **25 Telugu training samples** (containing only 3 positive spam examples), the model achieves **`F1 = 0.8889`** (80% recall, 100% precision). Telugu scam messages possess distinct transactional lexical patterns (e.g. `lakhs`, `prizes`, `urgent`, `OTP`) that stand out sharply against personal conversational dialect.
2. **Convergence Threshold at N = 300:** At $N=300$ samples (32 threats), performance saturates to **`1.0000` F1** (100% precision, 100% recall), demonstrating that massive billion-token datasets are not mandatory for specialized domain classification when linguistic purity is preserved.
3. **The Dilution Phenomenon in Cross-Lingual Random Subsampling:** In Regimen 2, when randomly subsampling from the 83k global pool without language stratification at $N=100-2000$, Telugu samples form <0.5% of the data. The English-dominated feature space leads to high false alarm rates on Telugu benign text (precision = `0.22`, F1 = `0.34`). Only when full multilingual training incorporates the complete stratified Telugu corpus ($N=419$) does precision rebound to **`0.9000`** and accuracy to **`0.9770`**.

## 4. Visual Artifacts
- Telugu Learning Curve: [telugu_learning_curve.png](file:///c:/Users/lenovo/scamshield/reports/telugu_learning_curve.png)
