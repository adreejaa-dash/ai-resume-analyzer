"""
AI-Powered Resume–Job Matching & Skill Gap Analyzer
Streamlit Application
"""
import os
import sys
import json

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import streamlit as st
import numpy as np
import joblib

from src.parsing.pdf_parser import extract_text_from_pdf_bytes
from src.parsing.resume_parser import parse_resume
from src.parsing.jd_parser import parse_job_description
from src.preprocessing.text_cleaner import clean_for_tfidf, clean_for_embedding
from src.skills.taxonomy import SkillTaxonomy
from src.skills.extractor import SkillExtractor, extract_jd_skills
from src.skills.matcher import match_skills
from src.matching.tfidf_matcher import TfidfMatcher
from src.matching.semantic_matcher import SemanticMatcher
from src.scoring.scorer import calculate_match_score, load_scoring_config
from src.analysis.experience_education import match_experience, match_education
from src.analysis.resume_quality import analyze_resume_quality
from src.models.trainer import build_feature_vector

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0f1117; color: #e2e8f0; }

.hero-title {
    font-size: 2.6rem; font-weight: 700; text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem; line-height: 1.2;
}
.hero-sub {
    text-align: center; color: #94a3b8; font-size: 1rem;
    margin-bottom: 2rem;
}

.score-ring {
    display: flex; flex-direction: column; align-items: center;
    padding: 2rem; background: linear-gradient(135deg, #1e1b4b, #312e81);
    border-radius: 1.5rem; border: 1px solid #4c1d95;
    box-shadow: 0 4px 24px rgba(99, 102, 241, 0.25);
}
.score-number { font-size: 4.5rem; font-weight: 800; line-height: 1; }
.score-label { font-size: 0.85rem; color: #a5b4fc; margin-top: 0.5rem; text-transform: uppercase; letter-spacing: 0.1em; }

.issue-card {
    background: #1e293b; border-radius: 0.75rem;
    padding: 1rem 1.25rem; border-left: 4px solid;
    margin-bottom: 1rem; border-color: #f87171;
}
.issue-card.High { border-color: #ef4444; }
.issue-card.Medium { border-color: #f59e0b; }
.issue-card.Minor { border-color: #3b82f6; }
.issue-card.Critical { border-color: #dc2626; }

.issue-title { font-weight: 700; font-size: 1.1rem; color: #f8fafc; margin-bottom: 0.25rem; }
.issue-meta { font-size: 0.8rem; color: #94a3b8; margin-bottom: 0.5rem; }
.issue-body { font-size: 0.9rem; color: #cbd5e1; margin-bottom: 0.5rem; }
.issue-solution { background: rgba(34,197,94,0.1); padding: 0.75rem; border-radius: 0.5rem; border: 1px solid rgba(34,197,94,0.2); color: #86efac; font-size: 0.9rem; margin-top: 0.5rem; }

div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white; border: none; border-radius: 0.5rem;
    padding: 0.6rem 2rem; font-weight: 600; font-size: 1rem;
    width: 100%; transition: opacity 0.2s;
}
div[data-testid="stButton"] > button:hover { opacity: 0.9; }
</style>
""", unsafe_allow_html=True)


# ── Resource loading (cached) ──────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading models…")
def load_resources():
    tax = SkillTaxonomy()
    return {
        "extractor": SkillExtractor(tax),
        "tfidf": load_tfidf_model(),
        "sbert": SemanticMatcher(),
        "classifier": load_classifier_model()
    }

def load_tfidf_model():
    matcher = TfidfMatcher()
    vec_path = os.path.join(ROOT, "models", "tfidf_vectorizer.joblib")
    if os.path.exists(vec_path):
        matcher.load(vec_path)
    return matcher

def load_classifier_model():
    models_dir = os.path.join(ROOT, "models")
    model_path = os.path.join(models_dir, "best_model.joblib")
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    le_path = os.path.join(models_dir, "label_encoder.joblib")
    if not all(os.path.exists(p) for p in [model_path, scaler_path, le_path]):
        return None, None, None
    return joblib.load(model_path), joblib.load(scaler_path), joblib.load(le_path)

@st.cache_data
def load_scoring_cfg():
    return load_scoring_config()


# ── Analysis engine ────────────────────────────────────────────────────────────

def run_analysis(resume_text: str, jd_text: str, career_level: str, target_role: str, yoe: str) -> dict:
    res = load_resources()
    extractor = res["extractor"]
    tfidf = res["tfidf"]
    sbert = res["sbert"]
    model, scaler, le = res["classifier"]
    config = load_scoring_cfg()

    resume_parsed = parse_resume(resume_text)
    
    # JD Parsing (only if provided)
    has_jd = bool(jd_text.strip())
    
    score_result = None
    skill_match = None
    jd_skills_data = None
    resume_skills = extractor.extract_flat_skills(resume_text)

    if has_jd:
        jd_parsed = parse_job_description(jd_text)
        jd_skills_data = extract_jd_skills(jd_text, extractor)
        skill_match = match_skills(
            resume_skills,
            jd_skills_data["required"],
            jd_skills_data["preferred"],
        )
        resume_clean = clean_for_tfidf(resume_text)
        jd_clean = clean_for_tfidf(jd_text)
        tfidf_sim = tfidf.similarity(resume_clean, jd_clean)
        
        resume_emb = clean_for_embedding(resume_text)
        jd_emb = clean_for_embedding(jd_text)
        sbert_sim = sbert.similarity(resume_emb, jd_emb)
        
        exp_match = match_experience(jd_text, resume_text)
        edu_match = match_education(jd_text, resume_text)
        
        ml_proba_value = 0.5
        if model is not None:
            feat_vec = build_feature_vector(
                tfidf_sim=tfidf_sim,
                sbert_sim=sbert_sim,
                skill_coverage=skill_match["overall_coverage"],
                required_skill_coverage=skill_match["required_coverage"],
                preferred_skill_coverage=skill_match["preferred_coverage"],
                resume_skill_count=len(resume_skills),
                jd_skill_count=len(jd_skills_data["all"]),
                matched_skill_count=skill_match["matched_required_count"],
                missing_skill_count=skill_match["missing_required_count"],
                resume_word_count=len(resume_text.split()),
                jd_word_count=len(jd_text.split()),
                education_match=edu_match["match_indicator"],
                experience_match=exp_match["match_indicator"],
            ).reshape(1, -1)
            feat_scaled = scaler.transform(feat_vec)
            proba = model.predict_proba(feat_scaled)[0]
            ml_proba_value = float(max(proba))
            
        score_result = calculate_match_score(
            required_skill_coverage=skill_match["required_coverage"],
            semantic_similarity=sbert_sim,
            tfidf_similarity=tfidf_sim,
            experience_relevance=exp_match["match_indicator"],
            education_relevance=edu_match["match_indicator"],
            preferred_skill_coverage=skill_match["preferred_coverage"],
            ml_model_probability=ml_proba_value,
            config=config,
        )

    # Resume quality (Audit)
    audit_results = analyze_resume_quality(
        text=resume_text,
        sections=resume_parsed["sections"],
        contact=resume_parsed["contact"],
        word_count=len(resume_text.split()),
        career_level=career_level,
        target_role=target_role,
        yoe=yoe
    )

    return {
        "has_jd": has_jd,
        "score": score_result,
        "skill_match": skill_match,
        "jd_required": jd_skills_data["required"] if has_jd else [],
        "jd_preferred": jd_skills_data["preferred"] if has_jd else [],
        "resume_parsed": resume_parsed,
        "audit": audit_results,
    }


# ── Main UI ────────────────────────────────────────────────────────────────────

def main():
    st.markdown('<div class="hero-title">🎯 AI Resume Analyzer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Upload your resume to get a detailed audit and actionable improvements.</div>',
        unsafe_allow_html=True,
    )

    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown("### 1. Upload Resume")
        uploaded = st.file_uploader(
            "Upload PDF resume", type=["pdf"],
            help="Upload a text-extractable PDF resume",
        )
        
        st.markdown("### 2. Context")
        career_level = st.selectbox("Career Level", ["Fresher / Student", "Experienced Professional"])
        yoe = None
        if career_level == "Experienced Professional":
            yoe = st.selectbox("Years of Experience", ["1–2 years", "3–5 years", "6–8 years", "9–12 years", "12+ years"])
            
        roles = [
            "Software Engineer", "Software Developer", "Frontend Developer", "Backend Developer",
            "Full Stack Developer", "Mobile App Developer", "Data Analyst", "Data Scientist",
            "Machine Learning Engineer", "AI Engineer", "Business Analyst", "Product Manager",
            "DevOps Engineer", "Cloud Engineer", "Cybersecurity Analyst", "QA Engineer",
            "UI/UX Designer", "Database Administrator", "Financial Analyst", "Marketing Analyst",
            "Human Resources", "Management Consultant", "Sales / Business Development", "Other"
        ]
        target_role = st.selectbox("Target Role", roles)
        if target_role == "Other":
            target_role = st.text_input("Enter Target Role")

    with col_r:
        st.markdown("### 3. JD (Optional)")
        jd_text = st.text_area(
            "Paste Job Description",
            height=300,
            placeholder="Paste the full job description here for job-specific matching...",
        )

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("🚀 Analyze Resume", use_container_width=True)

    if analyze_btn:
        if uploaded is None:
            st.error("Please upload a PDF resume.")
            return

        pdf_result = extract_text_from_pdf_bytes(uploaded.read())
        if pdf_result["error"]:
            st.error(f"PDF Error: {pdf_result['error']}")
            return
        
        resume_text = pdf_result["text"]

        with st.spinner("Analyzing resume..."):
            results = run_analysis(resume_text, jd_text, career_level, target_role, yoe)
        
        st.session_state["results"] = results
        st.session_state["has_jd"] = results["has_jd"]

    # ── Results Dashboard ──
    if "results" in st.session_state:
        r = st.session_state["results"]
        has_jd = st.session_state["has_jd"]

        st.divider()
        st.markdown("## 📊 Analysis Results")
        
        audit = r["audit"]
        score_val = audit["quality_score"]
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(f"""
            <div class="score-ring">
                <div class="score-number" style="color:#a78bfa">{score_val}</div>
                <div class="score-label">Resume Quality Score</div>
            </div>
            """, unsafe_allow_html=True)
            
            if has_jd and r["score"]:
                jd_score = r["score"]["overall_score"]
                st.markdown(f"""
                <div class="score-ring" style="margin-top: 1rem;">
                    <div class="score-number" style="color:#34d399">{jd_score}</div>
                    <div class="score-label">Resume-Job Match Score</div>
                </div>
                """, unsafe_allow_html=True)
                
        with c2:
            st.markdown("### ✨ Strengths")
            if not audit["strengths"]:
                st.write("- Resume parsed successfully.")
            for s in audit["strengths"]:
                st.markdown(f"- {s}")
                
        st.divider()
        st.markdown("### 🔍 Resume Audit")
        
        issues = audit["issues"]
        if not issues:
            st.success("No major issues found in your resume!")
        else:
            for issue in issues:
                st.markdown(f"""
                <div class="issue-card {issue['severity']}">
                    <div class="issue-title">{issue['title']}</div>
                    <div class="issue-meta"><strong>Severity:</strong> {issue['severity']} | <strong>Location:</strong> {issue['location']}</div>
                    <div class="issue-body"><strong>Why it matters:</strong> {issue['why']}</div>
                    <div class="issue-solution"><strong>Solution:</strong> {issue['solution']}</div>
                </div>
                """, unsafe_allow_html=True)
                
        if has_jd:
            st.divider()
            st.markdown("### 🎯 Job Match Gaps")
            match = r["skill_match"]
            
            mc1, mc2 = st.columns(2)
            with mc1:
                st.markdown("#### Matched Skills")
                if match["matched_required"]:
                    st.write(", ".join(match["matched_required"]))
                else:
                    st.write("None")
            with mc2:
                st.markdown("#### Missing Skills")
                if match["missing_required"]:
                    st.write(", ".join(match["missing_required"]))
                    st.caption("Only add these if you genuinely possess the skill or relevant experience.")
                else:
                    st.write("None")

if __name__ == "__main__":
    main()
