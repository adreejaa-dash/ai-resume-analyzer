"""
AI Resume Analyzer — Streamlit Application

Architecture:
  Resume PDF → PyMuPDF → Resume Parsing → Skill Extraction
  (Optional) JD Text → JD Parsing → Skill Extraction
  → TF-IDF + Sentence-BERT similarity (JD mode only)
  → Skill Gap Analysis → Resume Quality Audit → Results Dashboard
"""
import os
import sys
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import streamlit as st
import joblib

from src.parsing.pdf_parser import extract_text_from_pdf_bytes
from src.parsing.image_parser import extract_text_from_image_bytes
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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0f1117; color: #e2e8f0; }

.hero-title {
    font-size: 2.4rem; font-weight: 700; text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.25rem; line-height: 1.2;
}
.hero-sub {
    text-align: center; color: #94a3b8; font-size: 0.95rem;
    margin-bottom: 2rem;
}

.score-ring {
    display: flex; flex-direction: column; align-items: center;
    padding: 1.75rem 1.5rem;
    background: linear-gradient(135deg, #1e1b4b, #312e81);
    border-radius: 1.25rem; border: 1px solid #4c1d95;
    box-shadow: 0 4px 20px rgba(99,102,241,0.2);
    margin-bottom: 0.75rem;
}
.score-number { font-size: 4rem; font-weight: 800; line-height: 1; }
.score-label  { font-size: 0.8rem; color: #a5b4fc; margin-top: 0.4rem;
                text-transform: uppercase; letter-spacing: 0.1em; text-align: center; }
.score-note   { font-size: 0.7rem; color: #64748b; margin-top: 0.3rem; text-align: center; }

.issue-card {
    background: #1e293b; border-radius: 0.75rem;
    padding: 1rem 1.2rem; border-left: 4px solid #64748b;
    margin-bottom: 0.9rem;
}
.issue-card.Critical { border-color: #dc2626; }
.issue-card.High     { border-color: #ef4444; }
.issue-card.Medium   { border-color: #f59e0b; }
.issue-card.Minor    { border-color: #3b82f6; }

.issue-title  { font-weight: 700; font-size: 1rem; color: #f8fafc; margin-bottom: 0.2rem; }
.issue-meta   { font-size: 0.78rem; color: #94a3b8; margin-bottom: 0.4rem; }
.issue-why    { font-size: 0.88rem; color: #cbd5e1; margin-bottom: 0.4rem; }
.issue-sol    { background: rgba(34,197,94,0.08); padding: 0.65rem 0.9rem;
                border-radius: 0.4rem; border: 1px solid rgba(34,197,94,0.2);
                color: #86efac; font-size: 0.88rem; }

.skill-tag {
    display: inline-block; padding: 0.22rem 0.55rem;
    border-radius: 0.35rem; font-size: 0.78rem; font-weight: 500;
    margin: 0.18rem; border: 1px solid transparent;
}
.matched  { background: rgba(34,197,94,0.13); color:#86efac; border-color:rgba(34,197,94,0.3); }
.missing  { background: rgba(239,68,68,0.12);  color:#fca5a5; border-color:rgba(239,68,68,0.28); }
.preferred{ background: rgba(234,179,8,0.1);   color:#fde047; border-color:rgba(234,179,8,0.25); }

.align-card {
    background: #1e293b; border-radius: 0.75rem; padding: 1rem 1.2rem;
    border: 1px solid #334155; margin-bottom: 0.75rem;
}
.align-label { font-size: 0.78rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.07em; }
.align-value { font-size: 0.95rem; color:#e2e8f0; margin-top:0.2rem; }
.align-note  { font-size:0.8rem; color:#64748b; margin-top:0.2rem; }

div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white; border: none; border-radius: 0.5rem;
    padding: 0.65rem 2rem; font-weight: 600; font-size: 0.95rem;
    width: 100%; transition: opacity 0.2s;
}
div[data-testid="stButton"] > button:hover { opacity: 0.88; }
.stSelectbox input { caret-color: transparent !important; cursor: pointer !important; }
</style>
""", unsafe_allow_html=True)


# ── Cached resource loading ───────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading models…")
def load_resources():
    tax = SkillTaxonomy()
    return {
        "extractor": SkillExtractor(tax),
        "tfidf":     _load_tfidf(),
        "sbert":     SemanticMatcher(),
        "clf":       _load_classifier(),
    }

def _load_tfidf():
    m = TfidfMatcher()
    p = os.path.join(ROOT, "models", "tfidf_vectorizer.joblib")
    if os.path.exists(p):
        m.load(p)
    return m

def _load_classifier():
    d = os.path.join(ROOT, "models")
    paths = [os.path.join(d, f) for f in ("best_model.joblib", "scaler.joblib", "label_encoder.joblib")]
    if not all(os.path.exists(p) for p in paths):
        return None, None, None
    return tuple(joblib.load(p) for p in paths)

@st.cache_data
def _load_scoring_cfg():
    return load_scoring_config()


# ── Analysis pipeline ─────────────────────────────────────────────────────────

def run_analysis(resume_text: str, jd_text: str, career_level: str, target_role: str, yoe) -> dict:
    res = load_resources()
    extractor = res["extractor"]
    tfidf     = res["tfidf"]
    sbert     = res["sbert"]
    model, scaler, le = res["clf"]
    config    = _load_scoring_cfg()

    resume_parsed  = parse_resume(resume_text)
    resume_skills  = extractor.extract_flat_skills(resume_text)
    has_jd         = bool(jd_text.strip())

    score_result   = None
    skill_match    = None
    jd_skills_data = None
    exp_match      = None
    edu_match      = None
    tfidf_sim      = None
    sbert_sim      = None

    if has_jd:
        jd_skills_data = extract_jd_skills(jd_text, extractor)
        skill_match    = match_skills(resume_skills, jd_skills_data["required"], jd_skills_data["preferred"])

        resume_clean   = clean_for_tfidf(resume_text)
        jd_clean       = clean_for_tfidf(jd_text)
        tfidf_sim      = tfidf.similarity(resume_clean, jd_clean)

        resume_emb     = clean_for_embedding(resume_text)
        jd_emb         = clean_for_embedding(jd_text)
        sbert_sim      = sbert.similarity(resume_emb, jd_emb)

        exp_match      = match_experience(jd_text, resume_text)
        edu_match      = match_education(jd_text, resume_text)

        ml_proba = 0.5
        if model is not None:
            fv = build_feature_vector(
                tfidf_sim=tfidf_sim, sbert_sim=sbert_sim,
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
            ml_proba = float(max(model.predict_proba(scaler.transform(fv))[0]))

        score_result = calculate_match_score(
            required_skill_coverage=skill_match["required_coverage"],
            semantic_similarity=sbert_sim,
            tfidf_similarity=tfidf_sim,
            experience_relevance=exp_match["match_indicator"],
            education_relevance=edu_match["match_indicator"],
            preferred_skill_coverage=skill_match["preferred_coverage"],
            ml_model_probability=ml_proba,
            config=config,
        )

    audit = analyze_resume_quality(
        text=resume_text,
        sections=resume_parsed["sections"],
        contact=resume_parsed["contact"],
        word_count=len(resume_text.split()),
        career_level=career_level,
        target_role=target_role,
        yoe=yoe,
    )

    return {
        "has_jd":       has_jd,
        "score":        score_result,
        "skill_match":  skill_match,
        "jd_required":  jd_skills_data["required"]  if has_jd else [],
        "jd_preferred": jd_skills_data["preferred"] if has_jd else [],
        "exp_match":    exp_match,
        "edu_match":    edu_match,
        "tfidf_sim":    tfidf_sim,
        "sbert_sim":    sbert_sim,
        "resume_parsed": resume_parsed,
        "audit":        audit,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _skill_tags(skills, css_class):
    if not skills:
        st.markdown("<span style='color:#64748b;font-size:0.85rem'>None detected</span>", unsafe_allow_html=True)
        return
    html = " ".join(f'<span class="skill-tag {css_class}">{s}</span>' for s in skills)
    st.markdown(html, unsafe_allow_html=True)

def _align_card(label, value, note=""):
    note_html = f'<div class="align-note">{note}</div>' if note else ""
    st.markdown(
        f'<div class="align-card">'
        f'<div class="align-label">{label}</div>'
        f'<div class="align-value">{value}</div>'
        f'{note_html}</div>',
        unsafe_allow_html=True,
    )

def _pct(v):
    return f"{v*100:.0f}%"


# ── Main UI ───────────────────────────────────────────────────────────────────

TARGET_ROLES = [
    "Software Engineer", "Software Developer", "Frontend Developer",
    "Backend Developer", "Full Stack Developer", "Mobile App Developer",
    "Data Analyst", "Data Scientist", "Machine Learning Engineer", "AI Engineer",
    "Business Analyst", "Business Intelligence Analyst",
    "Product Manager", "Product Analyst",
    "Project Manager", "Program Manager",
    "DevOps Engineer", "Cloud Engineer", "Cloud Architect",
    "Cybersecurity Analyst", "Security Engineer",
    "QA Engineer", "Test Engineer", "Automation Test Engineer",
    "UI/UX Designer", "Product Designer",
    "Database Administrator", "Systems Engineer", "Network Engineer",
    "Solutions Architect", "Financial Analyst", "Marketing Analyst",
    "Digital Marketing Specialist", "Human Resources",
    "Talent Acquisition / Recruiter", "Operations Analyst",
    "Supply Chain Analyst", "Management Consultant", "Consultant",
    "Technical Writer", "Customer Success",
    "Sales / Business Development", "Other",
]


def main():
    st.markdown('<div class="hero-title">🎯 AI Resume Analyzer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Get a detailed audit of your resume with specific issues and actionable solutions.</div>',
        unsafe_allow_html=True,
    )

    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown("### 1. Upload Resume")
        uploaded = st.file_uploader("Upload PDF", type=["pdf"],
                                    help="Upload a text-extractable PDF resume.")

        st.markdown("### 2. Career Context")
        career_level = st.selectbox("Career Level", ["Fresher / Student", "Experienced Professional"])
        yoe = None
        if career_level == "Experienced Professional":
            yoe = st.selectbox("Years of Experience",
                               ["1–2 years", "3–5 years", "6–8 years", "9–12 years", "12+ years"])

        target_role = st.selectbox("Target Role", TARGET_ROLES)
        if target_role == "Other":
            target_role = st.text_input("Enter your target role")

    with col_r:
        st.markdown("### 3. Job Description *(Optional)*")
        
        jd_upload = st.file_uploader(
            "Upload JD (PDF, PNG, JPG/JPEG)", 
            type=["pdf", "png", "jpg", "jpeg"],
            help="Upload a Job Description file."
        )
        
        st.markdown("**OR**")
        
        jd_text = st.text_area(
            "Paste Job Description",
            height=200,
            placeholder="Paste the full job description here...\nLeave blank to analyze your resume independently.",
        )
        
        if jd_upload is not None:
            st.info(f"Using uploaded JD: {jd_upload.name}")
        elif jd_text.strip():
            st.info("Using pasted JD text.")

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("🚀 Analyze Resume", use_container_width=True)

    if analyze_btn:
        if uploaded is None:
            st.error("Please upload a PDF resume.")
            return
        pdf = extract_text_from_pdf_bytes(uploaded.read())
        if pdf["error"]:
            st.error(f"Could not read PDF: {pdf['error']}")
            return
        if pdf["word_count"] < 30:
            st.warning("Very little text was extracted from the PDF. "
                       "The resume may use images, scanned pages, or complex formatting. "
                       "Results may be incomplete.")
            if pdf["word_count"] < 10:
                st.error("The PDF could not be parsed. Please try a different file.")
                return

        final_jd_text = ""
        if jd_upload is not None:
            file_type = jd_upload.name.split('.')[-1].lower()
            if file_type == 'pdf':
                jd_pdf = extract_text_from_pdf_bytes(jd_upload.read())
                if jd_pdf["error"]:
                    st.error(f"Could not read JD PDF: {jd_pdf['error']}")
                    return
                final_jd_text = jd_pdf["text"]
            elif file_type in ['png', 'jpg', 'jpeg']:
                jd_img = extract_text_from_image_bytes(jd_upload.read())
                if jd_img["error"]:
                    st.error(f"Could not read JD Image: {jd_img['error']}")
                    return
                final_jd_text = jd_img["text"]
        else:
            final_jd_text = jd_text

        with st.spinner("Analyzing resume…"):
            results = run_analysis(pdf["text"], final_jd_text, career_level, target_role, yoe)

        st.session_state["results"]  = results
        st.session_state["has_jd"]   = results["has_jd"]
        st.session_state["pdf_meta"] = pdf

    # ── Results ───────────────────────────────────────────────────────────
    if "results" not in st.session_state:
        return

    r      = st.session_state["results"]
    has_jd = st.session_state["has_jd"]
    audit  = r["audit"]

    st.divider()
    st.markdown("## 📊 Analysis Results")

    # ── Scores row ────────────────────────────────────────────────────────
    score_cols = st.columns(2) if has_jd else st.columns([1, 2])

    with score_cols[0]:
        q_score = audit["quality_score"]
        q_color = "#34d399" if q_score >= 75 else "#fbbf24" if q_score >= 55 else "#f87171"
        st.markdown(f"""
        <div class="score-ring">
            <div class="score-number" style="color:{q_color}">{q_score}</div>
            <div class="score-label">Resume Quality Score / 100</div>
            <div class="score-note">Application-defined assessment · Not an official ATS score</div>
        </div>""", unsafe_allow_html=True)

    if has_jd and r["score"]:
        with score_cols[1]:
            jd_s     = r["score"]["overall_score"]
            jd_cat   = r["score"]["category"]
            jd_color = "#34d399" if jd_s >= 75 else "#fbbf24" if jd_s >= 55 else "#f87171"
            st.markdown(f"""
            <div class="score-ring">
                <div class="score-number" style="color:{jd_color}">{jd_s}</div>
                <div class="score-label">Resume–Job Match Score / 100</div>
                <div class="score-note">{jd_cat} · Application-defined assessment</div>
            </div>""", unsafe_allow_html=True)

    # ── Strengths ─────────────────────────────────────────────────────────
    if not has_jd:
        with score_cols[1]:
            st.markdown("### ✨ Strengths")
            if audit["strengths"]:
                for s in audit["strengths"]:
                    st.markdown(f"- {s}")
            else:
                st.markdown("- Resume was parsed and analyzed.")
    else:
        st.markdown("### ✨ Strengths")
        if audit["strengths"]:
            for s in audit["strengths"]:
                st.markdown(f"- {s}")

    # ── Resume Audit ──────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🔍 Resume Audit")
    issues = audit["issues"]

    if not issues:
        st.success("No significant issues found in your resume.")
    else:
        order = {"Critical": 0, "High": 1, "Medium": 2, "Minor": 3}
        for issue in sorted(issues, key=lambda x: order.get(x["severity"], 4)):
            st.markdown(f"""
            <div class="issue-card {issue['severity']}">
                <div class="issue-title">{issue['title']}</div>
                <div class="issue-meta">
                    <strong>Severity:</strong> {issue['severity']} &nbsp;|&nbsp;
                    <strong>Location:</strong> {issue['location']}
                </div>
                <div class="issue-why"><strong>Why it matters:</strong> {issue['why']}</div>
                <div class="issue-sol"><strong>Solution:</strong> {issue['solution']}</div>
            </div>""", unsafe_allow_html=True)

    # ── JD Analysis ───────────────────────────────────────────────────────
    if has_jd:
        st.divider()
        st.markdown("### 🎯 Job Match Analysis")

        match = r["skill_match"]

        # Skills
        sk1, sk2, sk3 = st.columns(3)
        with sk1:
            st.markdown("#### ✅ Matched Required Skills")
            _skill_tags(match["matched_required"], "matched")
        with sk2:
            st.markdown("#### ❌ Missing Required Skills")
            _skill_tags(match["missing_required"], "missing")
            if match["missing_required"]:
                st.caption("⚠️ Only add these if you genuinely possess the skill.")
        with sk3:
            st.markdown("#### ⭐ Missing Preferred Skills")
            _skill_tags(match["missing_preferred"], "preferred")
            if match["missing_preferred"]:
                st.caption("These are nice-to-have, not required. Only add if applicable.")

        # Keyword alignment (lexical / semantic in plain language)
        st.markdown("#### 🔑 Keyword & Relevance Alignment")
        ka1, ka2 = st.columns(2)
        with ka1:
            ts = r["tfidf_sim"] or 0
            if ts >= 0.6:
                kw_msg = "Your resume uses many of the same keywords as the job description."
                kw_indicator = "🟢 Strong"
            elif ts >= 0.35:
                kw_msg = "Your resume shares some keywords with the job description, but there is room for improvement."
                kw_indicator = "🟡 Moderate"
            else:
                kw_msg = "Your resume has low keyword overlap with the job description. Consider mirroring key terms from the JD where truthfully applicable."
                kw_indicator = "🔴 Low"
            _align_card("Keyword Alignment", kw_indicator, kw_msg)

        with ka2:
            ss = r["sbert_sim"] or 0
            if ss >= 0.6:
                sem_msg = "Your experience and the job requirements are closely aligned in meaning."
                sem_indicator = "🟢 Strong"
            elif ss >= 0.35:
                sem_msg = "Your experience and the job requirements show moderate alignment."
                sem_indicator = "🟡 Moderate"
            else:
                sem_msg = "Your background and the job requirements appear weakly aligned. Review the JD and consider how your experience maps to its requirements."
                sem_indicator = "🔴 Low"
            _align_card("Relevance Alignment", sem_indicator, sem_msg)

        # Experience + Education alignment
        st.markdown("#### 📋 Experience & Education")
        ea1, ea2 = st.columns(2)
        exp = r["exp_match"]
        edu = r["edu_match"]
        with ea1:
            _align_card(
                "Experience Requirement",
                f"JD: {exp['jd_requirement']}",
                f"Your resume: {exp['resume_experience']} — {exp['note']}",
            )
        with ea2:
            _align_card(
                "Education Requirement",
                f"JD: {edu['jd_requirement']}",
                f"Your resume: {edu['resume_degree']} — {edu['note']}",
            )


if __name__ == "__main__":
    main()
