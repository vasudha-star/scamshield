# ScamShield — Dataset Audit Summary (Phase 2)

This audit documents the raw datasets ingested into `data/raw/` prior to any standardization or preprocessing.

| Dataset | Records | Labels Present | URL Presence (%) | Duplicates (%) | Avg Chars | Status |
|---|---|---|---|---|---|---|
| **MeAJOR Phishing Benchmark** | 108,685 | 0.0: 60650, 1.0: 48034, nan: 1 | 58.93% | 4.75% | 1168.11 | Ingested & Verified |
| **Indian Cyber Scam Communication** | 10,000 | 0: 5000, 1: 5000 | 0.0% | 92.57% | 80.86 | Ingested & Verified |
| **Dravidian & Indian SMS Corpus** | 4,567 | ham: 3356, spam: 1211 | 31.66% | 16.25% | 140.64 | Ingested & Verified |
| **LLM-Generated Phishing Benchmark** | 9,986 | human: 5000, llm_generated: 4986 | 50.01% | 0.0% | 1222.83 | Ingested & Verified |

## Key Audit Findings
- **MeAJOR:** Large scale English email benchmark, ideal for Big Data baseline and TF-IDF calibration.
- **Indian Cyber Scam:** Distinct Indian scam scenarios; high templated text rate requires rigorous deduplication in Phase 3.
- **Dravidian SMS:** Contains real SMS with regional Indian language code distributions (Telugu codes 3 & 4), suitable for Telugu low-resource analysis.
- **LLM Benchmark:** Phishing corpus balanced between human and LLM generation, dedicated to Phase 18 robustness experiments.
