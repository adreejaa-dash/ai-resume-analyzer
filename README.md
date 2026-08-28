# AI Resume Analyzer

An intelligent, fully local NLP and Machine Learning application designed to help job seekers improve their resumes through deep analysis and actionable feedback.

## Features

- **Deep Resume Audit**: Scans resumes for formatting errors, writing clichés, repetitive action verbs, missing metrics, and missing contact/personal information, providing specific, actionable solutions for each detected issue.
- **Job Description Matching (Optional)**: If a Job Description is provided, the system checks for missing required skills, missing preferred skills, and calculates lexical (TF-IDF) and semantic (Sentence-BERT) alignment.
- **Context-Aware Scoring**: Adjusts expectations based on the user's Career Level (Fresher vs. Experienced) and Target Role.
- **Non-Technical UI**: The frontend is explicitly designed to be clean and simple, removing all confusing ML jargon and exposing only the actionable insights the user needs to succeed.

## Architecture & Technology Stack

The application relies completely on a local, deterministic, and ML-based pipeline without relying on external generative LLM APIs for analysis:
- **PDF Parsing**: PyMuPDF (`fitz`) for robust text extraction.
- **Skill Extraction**: ESCO/O*NET taxonomy mapping.
- **Lexical Matching**: TF-IDF vectorization.
- **Semantic Matching**: Sentence-BERT (`all-MiniLM-L6-v2`) via HuggingFace.
- **Scoring Pipeline**: Logistic Regression & XGBoost models trained on a custom dataset, fused with a rule-based heuristic engine.
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
python3 app/streamlit_app.py
```
