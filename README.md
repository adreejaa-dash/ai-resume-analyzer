# AI-Powered Resume–Job Matching & Skill Gap Analyzer

> **Evolved from**: [Complaint Classifier](https://github.com/adreejaa-dash/complaint-classifier) — NLP/ML infrastructure (TF-IDF pipeline, model training loop, evaluation framework, joblib serialization) reused and redesigned around a new problem domain.

---

## Problem Statement

Traditional resume screening is keyword-dependent and misses semantic alignment — a candidate with strong relevant experience may be rejected because their resume uses different terminology than the job description. This project builds a hybrid NLP + ML system that evaluates resume–job fit using multiple complementary signals.

## Solution

A three-stage NLP/ML pipeline that combines lexical similarity, semantic similarity, skill gap analysis, and supervised classification into an explainable hybrid match score.

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
                         Structured Feature Engineering
                                      ↓
                    Supervised ML Classification (LR + XGBoost)
                          Labels: Good Fit / Potential Fit / No Fit
                                      ↓
                        Hybrid Explainable Scoring (0–100)
                                      ↓
                    Skill Gap Analysis + ATS-Oriented Checks
                                      ↓
                          Streamlit Dashboard
```

---

## Architecture

### Components Reused from Complaint Classifier

| Component | Reuse |
|-----------|-------|
| TF-IDF vectorizer setup | `TfidfVectorizer(ngram_range=(1,2), max_features=10000, sublinear_tf=True)` |
| Model training loop | LR → NB → XGBoost comparison with macro metrics |
| Evaluation framework | Accuracy, Macro P/R/F1, Weighted F1, Confusion Matrix |
| `clean_text()` | Adapted — preserves technical terms (`C++`, `scikit-learn`) |
| Joblib serialization | Model save/load pattern |
| LR explainability | `coef_` feature importance |
| Test structure | Unit test organization |

### New Components

- PyMuPDF PDF extraction
- Resume section detection
- JD required/preferred skill separation
- ESCO + O\*NET skill taxonomy (3,823 skills)
- Three-layer skill extraction (taxonomy + pattern + alias)
- Sentence-BERT semantic similarity (pretrained `all-MiniLM-L6-v2`)
- Structured feature engineering (14 features)
- Hybrid explainable scoring engine
- ATS-oriented resume quality checks
- Streamlit dashboard

---

## NLP Pipeline

### Stage 1 — Classical TF-IDF Baseline (from Complaint Classifier)

```python
TfidfVectorizer(ngram_range=(1,2), max_features=10000, sublinear_tf=True)
```

Resume and JD texts are vectorized independently (fitted only on training data).
Cosine similarity is computed between the two vectors.

**Reports**: `TF-IDF Similarity: X%`

### Stage 2 — Sentence-BERT Semantic Matching

Uses pretrained `sentence-transformers/all-MiniLM-L6-v2` (~80MB).

> **Important**: SBERT is used as a pretrained embedding model. It is NOT fine-tuned in this project.

Encodes resume and JD into 384-dimensional embeddings, then computes cosine similarity.

**Why this matters**: SBERT captures semantic alignment even when exact keyword overlap is low (e.g., a computational linguist discussing "distributional semantics" is semantically closer to an NLP engineer JD than TF-IDF would indicate).

### Stage 3 — Hybrid Supervised Model

14 engineered features per resume/JD pair:

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
| `education_match` | Education match indicator [0,1] |
| `experience_match` | Experience match indicator [0,1] |

---

## Skill Extraction

Three-layer hybrid extraction:

**Layer 1**: Vocabulary matching against the skill taxonomy (ESCO + O*NET + curated categories)

**Layer 2**: Multi-word phrase patterns (sorted by length for greedy matching)

**Layer 3**: Alias normalization (`sklearn → scikit-learn`, `postgres → postgresql`, etc.)

### Skill Taxonomy Sources

| Source | Description | Count |
|--------|-------------|-------|
| **ESCO (via jjzha/skillspan)** | ESCO-annotated English job posting dataset (Decorte et al. 2022). BIO-tagged skill spans extracted across all splits. | 3,589 skills |
| **O*NET Technology Skills** | Curated technology examples from O*NET occupational data | 64 skills |
| **Curated categories** | Programming languages, data tools, ML/DL/NLP frameworks, cloud/DevOps, tools | ~170 skills |
| **Aliases** | Normalization mappings (`js → javascript`, etc.) | 70 aliases |
| **Total** | — | **3,823 unique skills** |

> **Note on ESCO**: Direct download from the official ESCO portal (`ec.europa.eu`) requires SSL certificate handling that Python's default context rejects. The ESCO-annotated skill data was instead obtained from the publicly accessible `jjzha/skillspan` dataset on HuggingFace, which contains BIO-tagged skill annotations derived from the ESCO taxonomy (Decorte et al., "JobBERT: Understanding Job Titles through Skills", 2022).
>
> **Note on O*NET**: Direct download from `onetcenter.org` requires registration/authentication. Curated technology skills from O*NET are included in the local vocabulary.

---

## Supervised ML Models

### Dataset

**Primary**: [`med2425/resume-job-fit-merged-v1`](https://huggingface.co/datasets/med2425/resume-job-fit-merged-v1)

- 93,733 resume–job description pairs
- Labels: `Good Fit`, `Potential Fit`, `No Fit`
- Provided train/test split preserved (80,017 train / 13,716 test)
- **Severely imbalanced**: No Fit ~64%, Potential Fit ~19%, Good Fit ~7%

**Secondary**: [`netsol/resume-score-details`](https://huggingface.co/datasets/netsol/resume-score-details)

- Used for external validation/reference only
- **NOT merged into training data**

### Class Imbalance Handling

- Stratified train/validation/test splitting
- Balanced class weights during training
- **Primary metric: Macro F1** (not accuracy)

### Models

| Model | Description |
|-------|-------------|
| **Logistic Regression** | Baseline linear model; provides feature coefficient explainability |
| **Gaussian Logistic Regression** | Classical baseline (retained from complaint classifier) |
| **XGBoost** | Gradient boosted trees; typically strongest for tabular engineered features |

### Data Leakage Prevention

- TF-IDF vectorizer fitted **only on training corpus**
- StandardScaler fitted **only on training features**
- Validation split derived from training data only
- Test set used **only for final evaluation**
- No test labels influenced threshold decisions

---

## Hybrid Scoring

The final score (0–100) is a weighted combination:

```yaml
weights:
  required_skill_coverage: 0.30
  semantic_similarity: 0.25
  tfidf_similarity: 0.10
  experience_relevance: 0.15
  education_relevance: 0.05
  preferred_skill_coverage: 0.05
  ml_model_confidence: 0.10
```

> **Important**: These weights are **application-defined heuristics**. They are NOT industry-standard, scientifically validated, or derived from hiring research. They are configurable in `config/scoring.yaml`.

**Match categories** (also configurable):
- Strong Match: ≥ 80
- Potential Match: 60–79
- Weak Match: < 60

### Explainability

The score is fully decomposable. Every component is reported separately:

```
Overall Match Score: 84/100

Required Skill Coverage:    90.0%   (weight: 30%)
Semantic Similarity:         82.0%   (weight: 25%)
TF-IDF Similarity:           74.0%   (weight: 10%)
Experience Relevance:        85.0%   (weight: 15%)
Education Relevance:        100.0%   (weight: 5%)
Preferred Skill Coverage:    60.0%   (weight: 5%)
ML Model Probability:        87.0%   (weight: 10%)
```

---

## Evaluation

> All metrics below are **actual test-set results** from training. They will be updated once training completes. No metrics are fabricated.

*Results will appear in `models/evaluation_report.md` and `models/model_info.json` after running `python3 scripts/train_models.py`.*

**Primary metric: Macro F1** (used because of severe class imbalance — accuracy alone is misleading when No Fit is 64% of the data).

---

## Error Analysis

See `notebooks/04_error_analysis.ipynb` for misclassification analysis.

Common failure modes include:
- High keyword overlap without true semantic relevance (TF-IDF false positives)
- Semantic similarity between different domains (SBERT false positives)
- Ambiguous job descriptions with insufficient context
- Resumes using terminology different from standard skill vocabulary

---

## Limitations

1. **SBERT is not fine-tuned** — semantic matching uses a general-purpose model, not one trained on resume/JD data
2. **Skill extraction misses context** — "Python" in "not proficient in Python" is still extracted as a skill
3. **Experience/education extraction is conservative** — returns "Unable to determine" when uncertain (by design)
4. **No OCR** — scanned PDF resumes are not supported
5. **ESCO via annotation dataset** — skill vocabulary is ESCO-derived but not the full official ESCO v1.2.0 dataset
6. **Class imbalance** — No Fit is dominant; model may underperform on Good Fit predictions
7. **Scoring weights are heuristic** — not empirically validated against actual hiring outcomes

---

## Installation

```bash
git clone https://github.com/adreejaa-dash/complaint-classifier
cd complaint-classifier
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
python3 tests/test_preprocessing.py
python3 tests/test_skill_extraction.py
python3 tests/test_scoring_parsing.py
```

---

## Future Improvements

1. Fine-tune SBERT on resume/JD pairs for domain-specific embeddings
2. OCR support for scanned PDF resumes
3. SHAP values for XGBoost feature importance explanation
4. Sensitivity analysis on scoring weights
5. Download the full ESCO v1.2.0 dataset when SSL issues are resolved
6. Named entity recognition for skill extraction using spaCy NER
7. Cross-lingual resume support

---

## Interview Story

> "I initially developed an NLP-based complaint classification system using TF-IDF, Logistic Regression, Logistic Regression, and XGBoost with ensemble learning. I then reused the NLP preprocessing, TF-IDF vectorization, and model-training infrastructure to redesign the problem around resume-job matching.
>
> I introduced structured skill extraction using an ESCO-annotated taxonomy (3,589 skills from Decorte et al. 2022) enriched with O\*NET technology data, semantic similarity using the pretrained Sentence-BERT model (all-MiniLM-L6-v2), and engineered features combining lexical similarity, semantic similarity, and skill coverage metrics. I then trained Logistic Regression and XGBoost classifiers on a 93K-pair dataset to predict Good Fit / Potential Fit / No Fit, using Macro F1 as the primary metric because the dataset is severely imbalanced. Finally, I built an explainable hybrid scoring system with configurable heuristic weights and deployed it as a Streamlit application."

---

## Project Structure

```
complaint-classifier/
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
│       ├── skill_aliases.json    # Normalization aliases
│       └── skill_categories.json # Category definitions
├── models/                       # Trained model artifacts (gitignored)
├── scripts/
│   ├── download_data.py          # Dataset download
│   ├── prepare_skills.py         # Taxonomy preparation
│   └── train_models.py           # End-to-end training
├── src/
│   ├── analysis/                 # Experience, education, ATS checks
│   ├── evaluation/               # Model evaluation utilities
│   ├── matching/                 # TF-IDF and SBERT matchers
│   ├── models/                   # Model trainer
│   ├── parsing/                  # PDF, resume, JD parsers
│   ├── preprocessing/            # Text cleaning
│   ├── scoring/                  # Hybrid scoring engine
│   └── skills/                   # Taxonomy, extraction, matching
└── tests/                        # Unit tests
```
