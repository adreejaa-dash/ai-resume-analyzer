# AI-Powered Resume Matcher & Skill Gap Analyzer

A three-stage NLP/ML pipeline that evaluates resume–job description fit using
lexical similarity, semantic embeddings, structured skill matching, and
supervised classification — deployed as an interactive Streamlit application.

---

## Problem Statement

Traditional resume screening is keyword-dependent and misses semantic
alignment: a candidate with strong relevant experience may be filtered out
simply because their resume uses different terminology than the job description.
This project builds a hybrid NLP + ML system that evaluates resume–job fit
across multiple complementary signals and explains where the gaps are.

---

## Pipeline Overview

```
Resume PDF → PyMuPDF text extraction → Resume parsing (sections, contact)
                                      ↓
                           Skill Extraction (ESCO + O*NET + aliases)
                                      ↓
Job Description → JD parsing → Required / Preferred skill classification
                                      ↓
              ┌───────────────────────┼──────────────────────┐
              ↓                       ↓                      ↓
      TF-IDF Similarity        SBERT Similarity       Skill Coverage
      (Stage 1 baseline)       (Stage 2 baseline)     (Required + Preferred)
              └───────────────────────┼──────────────────────┘
                                      ↓
                         Structured Feature Engineering (14 features)
                                      ↓
                    Supervised ML Classification (LR + XGBoost)
                          Labels: Good Fit / Potential Fit / No Fit
                                      ↓
                        Hybrid Explainable Match Score (0–100)
                                      ↓
                    Skill Gap Analysis + ATS-Oriented Checks
                                      ↓
                          Streamlit Dashboard
```

---

## Architecture

### Stage 1 — TF-IDF Lexical Baseline

```python
TfidfVectorizer(ngram_range=(1,2), max_features=10000, sublinear_tf=True)
```

Resume and JD texts are vectorized independently. The vectorizer is fitted
**only on the training corpus** (no leakage). Cosine similarity between the
two resulting vectors gives a lexical overlap signal.

**Output**: `TF-IDF Similarity score`

### Stage 2 — Sentence-BERT Semantic Similarity

Uses the pretrained `sentence-transformers/all-MiniLM-L6-v2` model (~80 MB).

> **Note**: SBERT is used as a pretrained embedding model. It is **NOT fine-tuned**
> in this project.

Encodes resume and JD into 384-dimensional embeddings, then computes cosine
similarity. Captures semantic alignment even when exact keyword overlap is low
(e.g. "distributional semantics" ↔ "NLP engineer").

**Output**: `SBERT Similarity score`

### Stage 3 — Supervised Classification on Engineered Features

14 structured features per resume–JD pair:

| Feature | Description |
|---------|-------------|
| `tfidf_similarity` | TF-IDF cosine similarity |
| `sbert_similarity` | Sentence-BERT cosine similarity |
| `skill_coverage` | Overall skill overlap ratio |
| `required_skill_coverage` | Required skill match ratio |
| `preferred_skill_coverage` | Preferred skill match ratio |
| `resume_skill_count` | Skills detected in resume |
| `jd_skill_count` | Skills detected in JD |
| `matched_skill_count` | Count of matched required skills |
| `missing_skill_count` | Count of missing required skills |
| `resume_word_count` | Resume length |
| `jd_word_count` | JD length |
| `length_ratio` | Resume/JD length ratio |
| `education_match` | Education match indicator [0, 1] |
| `experience_match` | Experience match indicator [0, 1] |

Models trained: **Logistic Regression** (production) and **XGBoost** (comparison).

---

## Skill Extraction

Three-layer hybrid extraction:

**Layer 1**: Vocabulary matching against the skill taxonomy (ESCO + O\*NET +
curated categories)

**Layer 2**: Multi-word phrase patterns (sorted by length for greedy matching)

**Layer 3**: Alias normalisation (`sklearn → scikit-learn`,
`postgres → postgresql`, etc.)

### Skill Taxonomy Sources

| Source | Description | Count |
|--------|-------------|-------|
| **ESCO (via jjzha/skillspan)** | ESCO-annotated English job posting dataset (Decorte et al. 2022). BIO-tagged skill spans extracted across all splits. | 3,589 skills |
| **O\*NET Technology Skills** | Curated technology examples from O\*NET occupational data | 64 skills |
| **Curated categories** | Programming languages, data tools, ML/DL/NLP frameworks, cloud/DevOps, tools | ~170 skills |
| **Aliases** | Normalisation mappings (`js → javascript`, etc.) | 70 aliases |
| **Total** | — | **3,823 unique searchable skill entities** |

> **Note on ESCO**: The ESCO-annotated skill data was obtained from the publicly
> accessible `jjzha/skillspan` HuggingFace dataset, which contains BIO-tagged
> skill annotations derived from the ESCO taxonomy (Decorte et al., "JobBERT:
> Understanding Job Titles through Skills", 2022).
>
> **Note on O\*NET**: Curated technology skills from O\*NET are included in the
> local vocabulary (`data/skills/onet_tech_skills.csv`).

---

## Dataset

**Primary**: [`med2425/resume-job-fit-merged-v1`](https://huggingface.co/datasets/med2425/resume-job-fit-merged-v1)

- 93,733 resume–job description pairs
- Labels: `Good Fit`, `Potential Fit`, `No Fit`
- Provided train/test split preserved: **80,017 train / 13,716 test**
- **Severely imbalanced**: No Fit ≈ 75%, Potential Fit ≈ 17%, Good Fit ≈ 8%

---

## Class Imbalance Handling

- Stratified train/validation/test splitting
- `class_weight="balanced"` on Logistic Regression
- **Primary evaluation metric: Macro F1** (not accuracy — accuracy is
  misleading when one class is 75% of the data)

---

## Data Leakage Prevention

- `TfidfVectorizer` fitted **only on training corpus**
- `StandardScaler` fitted **only on training features**
- Validation split derived from training data only
- Test set used **only for final evaluation**
- No test labels influenced threshold or model selection decisions

---

## Model Results

> All metrics below are **actual test-set results**. No metrics are fabricated.

**Primary metric: Macro F1** (handles severe class imbalance).

| Model | Accuracy | Macro Prec | Macro Rec | Macro F1 | Weighted F1 |
|-------|----------|-----------|-----------|----------|-------------|
| **Logistic Regression** ✓ | 0.5546 | 0.5448 | 0.5546 | 0.5446 | 0.5446 |
| XGBoost | 0.5458 | 0.5460 | 0.5458 | 0.5407 | 0.5407 |

**Production model**: Logistic Regression (selected by Macro F1).

See `models/evaluation_report.md` and `models/model_info.json` for full
per-class metrics and confusion matrices.

---

## Error Analysis & Limitations

### Hardest Class: "Potential Fit" (F1 ≈ 0.37)

The "Potential Fit" class is the most difficult boundary. It sits in the
subjective grey area between a clear rejection ("No Fit") and a confident
interview invite ("Good Fit"). Text-similarity and skill-overlap features alone
cannot reliably distinguish this boundary without richer metadata such as
precise years of experience or salary alignment.

**Misclassification pattern (observed)**:
- Potential Fit → predicted No Fit: most common error
- Potential Fit → predicted Good Fit: second most common error
- Good Fit and No Fit boundaries are more reliably predicted

### Known Limitations

1. **SBERT is not fine-tuned** — uses a general-purpose model, not one trained
   on resume/JD data
2. **Skill extraction ignores context** — `"Python"` inside `"not proficient in Python"`
   is still extracted as a skill
3. **Experience/education extraction is conservative** — returns
   `"Unable to determine"` when uncertain (by design)
4. **No OCR** — scanned PDF resumes are not supported
5. **ESCO via annotation dataset** — skill vocabulary is ESCO-derived but not
   the full official ESCO v1.2.0 dataset
6. **Scoring weights are heuristic** — not empirically validated against actual
   hiring outcomes

---

## Hybrid Match Score

The final score (0–100) is a weighted combination of signals:

```yaml
weights:
  required_skill_coverage: 0.30
  semantic_similarity:      0.25
  experience_relevance:     0.15
  tfidf_similarity:         0.10
  ml_model_confidence:      0.10
  education_relevance:      0.05
  preferred_skill_coverage: 0.05
```

> **Important**: These weights are **application-defined heuristics**. They are
> NOT industry-standard, scientifically validated, or derived from hiring
> research. They are configurable in `config/scoring.yaml`.
>
> The `ml_model_confidence` component is the Logistic Regression classifier's
> output probability for the predicted fit class. It is **NOT** a hiring
> probability and does **NOT** predict whether a candidate will be hired.

**Match categories** (configurable):
- Strong Match: ≥ 80
- Potential Match: 60–79
- Weak Match: < 60

### Explainability

Every component is reported separately in the Streamlit output:

```
Overall Match Score: 84/100   → Strong Match

  Required Skill Coverage:    90.0%   (weight: 30%)
  Semantic Similarity:        82.0%   (weight: 25%)
  Experience Relevance:       85.0%   (weight: 15%)
  TF-IDF Similarity:          74.0%   (weight: 10%)
  ML Fit-Class Confidence:    87.0%   (weight: 10%)
  Education Relevance:       100.0%   (weight:  5%)
  Preferred Skill Coverage:   60.0%   (weight:  5%)

Predicted Fit Class: Good Fit
Missing Skills: kubernetes, terraform, spark
```

---

## Installation

```bash
git clone https://github.com/adreejaa-dash/ai-resume-analyzer
cd ai-resume-analyzer
git checkout resume-analyser

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Generate Model Artifacts

```bash
# Download datasets
python3 scripts/download_data.py

# Prepare skill taxonomy
python3 scripts/prepare_skills.py

# Train models (takes ~20–40 min on CPU)
python3 scripts/train_models.py
```

### Run the Application

```bash
streamlit run app/streamlit_app.py
```

### Run Tests

```bash
pytest tests/
```

### Docker

```bash
docker-compose up --build
# Application available at http://localhost:8501
```

---

## Project Structure

```
ai-resume-analyzer/
├── app/
│   └── streamlit_app.py          # Streamlit dashboard
├── config/
│   └── scoring.yaml              # Configurable scoring weights
├── data/
│   ├── raw/                      # Downloaded datasets (gitignored)
│   ├── samples/                  # Sample resumes and JDs for testing
│   └── skills/
│       ├── esco_skills.csv       # ESCO-derived skill vocabulary
│       ├── onet_tech_skills.csv  # O*NET technology skills
│       ├── skill_aliases.json    # Normalisation aliases
│       └── skill_categories.json # Category definitions
├── models/                       # Trained model artifacts (gitignored)
│   ├── best_model.joblib         # Production model (Logistic Regression)
│   ├── logistic_regression.joblib
│   ├── xgboost_model.joblib
│   ├── tfidf_vectorizer.joblib
│   ├── scaler.joblib
│   ├── label_encoder.joblib
│   ├── model_info.json           # Evaluation results
│   └── evaluation_report.md
├── scripts/
│   ├── download_data.py          # Dataset download
│   ├── prepare_skills.py         # Taxonomy preparation
│   └── train_models.py           # End-to-end training pipeline
├── src/
│   ├── analysis/                 # Experience, education, ATS checks
│   ├── evaluation/               # Model evaluation utilities
│   ├── matching/                 # TF-IDF and SBERT matchers
│   ├── models/                   # Model trainer
│   ├── parsing/                  # PDF, resume, JD parsers
│   ├── preprocessing/            # Text cleaning
│   ├── scoring/                  # Hybrid scoring engine
│   └── skills/                   # Taxonomy, extraction, matching
├── tests/                        # Unit tests (26 tests, 0 failures)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Future Improvements

1. Fine-tune SBERT on resume/JD pairs for domain-specific embeddings
2. OCR support for scanned PDF resumes
3. SHAP values for XGBoost feature importance explanation
4. Sensitivity analysis on scoring weights
5. Download the full ESCO v1.2.0 dataset directly from the official portal
6. Named entity recognition for skill extraction using spaCy NER

---

## Tech Stack

Python · scikit-learn · XGBoost · Sentence-Transformers · NLTK · PyMuPDF ·
Pandas · Streamlit · Docker · HuggingFace Datasets
