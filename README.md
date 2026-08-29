# AI Resume Analyzer

An intelligent, fully local NLP and Machine Learning application designed to help job seekers improve their resumes through deep analysis and actionable feedback.

## Features

- **Deep Resume Audit**: Scans resumes for formatting errors, writing clichés, repetitive action verbs, missing metrics, unnecessary personal info (age, marital status), tense inconsistencies, date formatting inconsistencies, and overly long/short bullets. Provides specific, actionable solutions for each detected issue.
- **Job Description Matching (Optional)**: If a Job Description is provided, the system checks for missing required skills, missing preferred skills, and calculates lexical (TF-IDF) and semantic (Sentence-BERT) alignment.
- **Context-Aware Scoring**: Adjusts expectations based on the user's Career Level (Fresher vs. Experienced) and Target Role.
- **Non-Technical UI**: The frontend is explicitly designed to be clean and simple. You receive actionable insights and clear scores without any confusing machine learning jargon (no "Logistic Regression", no "TF-IDF", no "F1").

## Architecture & Technology Stack

The application relies completely on a local, deterministic pipeline without relying on external generative LLM APIs for analysis:
- **PDF Parsing**: PyMuPDF (`fitz`) for robust text extraction.
- **Skill Extraction**: ESCO/O*NET taxonomy mapping.
- **Lexical Matching**: TF-IDF vectorization.
- **Semantic Matching**: Sentence-BERT (`all-MiniLM-L6-v2`) via HuggingFace.
- **Scoring Pipeline**: Models trained on a custom dataset, fused with a deep rule-based heuristic engine.
- **Frontend**: Streamlit.

## Quick Start (Docker)
```bash
docker-compose up --build
```
Navigate to `http://localhost:8501`.

## Local Setup
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Limitations
- Formatting analysis relies on extracted text, so it cannot fully reproduce the exact visual layout of a PDF.
- The resume scoring is an application-defined quality assessment, not a guaranteed ATS approval rating or hiring probability.
