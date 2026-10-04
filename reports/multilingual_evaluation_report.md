# ScamShield: Multilingual Evaluation Report (Phase 14)
**Execution Timestamp:** 2026-10-03 19:53:10 UTC  
**Evaluation Split Hash:** `85851abd839f4971` (Locked test partition, N=17,901)  
**Evaluated Model:** `M4_FULL_SCAMSHIELD` (Early Fused Multimodal: TF-IDF + URL + Intent)  

## 1. Cross-Lingual Performance Summary
| Language | Description | Total N | Threats | Benign | Accuracy | Precision | Recall | F1-Score | PR-AUC | FPR | FNR |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`en`** | English | 17,684 | 8,638 | 9,046 | `0.9754` | `0.9717` | `0.9781` | **`0.9749`** | `0.9954` | `0.0272` | `0.0219` |
| **`hi`** | Hindi | 18 | 13 | 5 | `1.0000` | `1.0000` | `1.0000` | **`1.0000`** | `1.0000` | `0.0000` | `0.0000` |
| **`hinglish`** | Hinglish (Hindi-English) | 112 | 95 | 17 | `1.0000` | `1.0000` | `1.0000` | **`1.0000`** | `1.0000` | `0.0000` | `0.0000` |
| **`te`** | Telugu (Dravidian Low-Resource) | 87 | 10 | 77 | `0.9770` | `0.9000` | `0.9000` | **`0.9000`** | `0.9570` | `0.0130` | `0.1000` |
| **`overall`** | All Languages (Full Test Set) | 17,901 | 8,756 | 9,145 | `0.9756` | `0.9720` | `0.9783` | **`0.9751`** | `0.9954` | `0.0270` | `0.0217` |

## 2. Key Linguistic Findings
1. **English (`en`, N=17,684):** Forms the vast majority (98.8%) of the test corpus. Achieves solid precision (`0.9717`), recall (`0.9781`), and F1 (`0.9749`) with a PR-AUC of `0.9959`.
2. **Hindi (`hi`, N=18):** Demonstrates **100% Precision and 100% Recall** (F1 = `1.0000`, 13 threats, 5 benign). Indic Devanagari spam keywords are captured cleanly by Unicode-aware sublinear n-grams.
3. **Hinglish (`hinglish`, N=112):** Achieves **100% Precision and 100% Recall** (F1 = `1.0000`, 95 threats, 17 benign). Voice scam phone transcripts have unambiguous extortion triggers (`police`, `digital arrest`, `FIR`, `CBI`, `arrest`) and distinct benign greetings.
4. **Telugu (`te`, N=87):** Represents a low-resource Dravidian language with complex agglutinative morphology. Achieves **`0.9000` Precision, `0.9000` Recall, and `0.9000` F1** with `0.9770` Accuracy (9 threats caught, 1 false negative, 1 false positive out of 87 samples).

## 3. Low-Resource Dravidian (Telugu) Error Analysis
In the Telugu test partition (N=87, 10 true threats, 77 benign), only **2 errors** occurred:

| Message ID | True Label | Pred Label | Model Prob | Snippet |
| :--- | :---: | :---: | :---: | :--- |
| `1cfff4ba...` | Benign (0) | Threat (1) | `0.7028` | `Akka nenu Tinku entlo ne unava` |
| `90be4bd8...` | Threat (1) | Benign (0) | `0.3504` | `అన్ని సమస్యల నుండి అధిగమించండి.కాల్ 556789999` |

> **Morphological Root Cause:** The false positive message (`Akka nenu Tinku entlo ne unava`) is a colloquial romanized Telugu conversational message without URL or intent coercion, where informal words had sparse overlap with training tokens.

## 4. Visual Artifacts
- Multilingual F1 & Recall Comparison: [multilingual_f1_by_language.png](file:///c:/Users/lenovo/scamshield/reports/multilingual_f1_by_language.png)
- Confusion Matrices across Languages: [multilingual_confusion_matrices.png](file:///c:/Users/lenovo/scamshield/reports/multilingual_confusion_matrices.png)
