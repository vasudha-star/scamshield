# SCAMSHIELD

Explainable Multimodal AI for Phishing and Digital Scam Detection
(Text + URL/Structural + Intent Analysis), built on a Big Data (Spark) pipeline.

## Recommended environment

- **PySpark local mode** (`local[*]`) — no cluster needed for 100K–500K record scale. Runs on
  a laptop or in Google Colab identically via the same code.
- **Google Colab** (free/Pro tier) for shared GPU access when training XLM‑R — everyone on the
  team can run the same notebooks without individual GPU setups.
- **GitHub** as the single source of truth for code + `data/manifest.csv` (raw data itself is
  git-ignored — too large; keep it in Google Drive or a shared folder, referenced by manifest).
- **Local machine** for day-to-day dev of Spark jobs and Streamlit dashboard (fast iteration),
  pushing to Colab only for GPU model training (XLM‑R/MuRIL).

This avoids cloud costs entirely until/unless you outgrow it (Section 27 of the blueprint —
B2B/API stage is the point to reconsider AWS/GCP).

## Team split (from blueprint Section 25) → repo ownership

| Person | Workstream | Owns |
|---|---|---|
| A | Data Engineering | `src/ingestion/`, `data/manifest.csv`, Spark jobs |
| B | Text Analytics | `src/features/text_features.py`, EDA notebooks |
| C | URL/Structural + Modeling | `src/features/url_features.py`, `src/models/` |
| D | Explainability & Evaluation | `src/models/explain.py`, `src/dashboard/` |

With 2-4 people, double up: e.g. 2 people → {Data+Text} and {URL+Modeling+Explain}.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Java 11/17 is required for PySpark — verify with `java -version`.

## Repo layout

```
config/           dataset registry, canonical schema, seeds
data/
  raw/            untouched downloads (git-ignored)
  interim/        Parquet after Spark standardization
  processed/      gold feature-engineered dataset (versioned)
  manifest.csv    every dataset: source, URL/DOI, license, retrieval date, record count
src/
  ingestion/      download + Spark ingestion/cleaning/dedup
  features/       text (TF-IDF/n-gram), URL/structural, intent/persuasion features
  models/         M1-M4 ablation models, XLM-R, explainability (SHAP)
  dashboard/      Streamlit analytics dashboard
notebooks/        EDA, experiments (not the source of truth — promote logic into src/)
docs/             dataset manifest notes, experiment logs
```

## Week-by-week (from blueprint Section 24) — current phase

- [x] Weeks 1-2: literature review / research questions (this blueprint)
- [ ] **Week 3 (start here)**: download datasets, build manifest, raw data lake, canonical schema
- [ ] Week 4: Spark ingestion, cleaning, dedup, language handling
- [ ] Week 5: EDA + dashboard v1
- [ ] Week 6: TF-IDF + NB/LR/SVM/RF baselines (M1)
- [ ] Week 7: URL features + M2
- [ ] Week 8: Intent features + M4 (full model)
- [ ] Week 9: XLM-R multilingual + per-language eval
- [ ] Week 10: Telugu learning curve + translation augmentation
- [ ] Week 11: robustness (paraphrase/LLM-generated) + explainability
- [ ] Week 12: final dashboard, error analysis, paper, demo, viva prep

## Run ingestion (Week 3-4 scaffold)

```bash
python src/ingestion/download_datasets.py --dataset all
python src/ingestion/spark_pipeline.py --input data/raw --output data/interim
```
