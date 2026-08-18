"""
AI-Powered Resume–Job Matching & Skill Gap Analyzer
Streamlit Application

Architecture:
  Resume PDF → PyMuPDF → Resume Parsing → Skill Extraction
  JD Text → JD Parsing → Skill Extraction
  → TF-IDF Similarity (Stage 1 baseline)
  → Sentence-BERT Similarity (Stage 2 baseline)
  → Feature Engineering → Supervised ML Classification (Stage 3)
  → Hybrid Explainable Score → Skill Gap Analysis → Dashboard

Note: The analytical engine works without any LLM API.
      Skill extraction, matching, similarity, and classification
      are all computed locally.
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
    page_title="Resume–Job Match Analyzer",
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
.score-category { font-size: 1.1rem; font-weight: 600; margin-top: 0.5rem; }

.kpi-card {
    background: #1e293b; border-radius: 0.75rem;
    padding: 1rem 1.25rem; border-left: 3px solid;
    margin-bottom: 0.75rem;
}
.kpi-label { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em; }
.kpi-value { font-size: 1.4rem; font-weight: 700; margin-top: 0.2rem; }

.skill-tag {
    display: inline-block; padding: 0.25rem 0.6rem;
    border-radius: 0.375rem; font-size: 0.78rem; font-weight: 500;
    margin: 0.2rem; border: 1px solid transparent;
}
.skill-matched { background: rgba(34,197,94,0.15); color: #86efac; border-color: rgba(34,197,94,0.3); }
.skill-missing { background: rgba(239,68,68,0.15); color: #fca5a5; border-color: rgba(239,68,68,0.3); }
.skill-preferred { background: rgba(234,179,8,0.12); color: #fde047; border-color: rgba(234,179,8,0.25); }

.section-title {
    font-size: 1rem; font-weight: 600; color: #c4b5fd;
    margin-top: 1.5rem; margin-bottom: 0.75rem;
    text-transform: uppercase; letter-spacing: 0.1em;
}
.info-box {
    background: #1e293b; border-radius: 0.75rem; padding: 1rem 1.25rem;
    border: 1px solid #334155; margin-bottom: 0.75rem; font-size: 0.9rem;
}
.warning-box {
    background: rgba(234,179,8,0.1); border-radius: 0.5rem;
    padding: 0.75rem 1rem; border: 1px solid rgba(234,179,8,0.3);
    color: #fde047; font-size: 0.85rem; margin-bottom: 1rem;
}
.rec-item {
    padding: 0.5rem 0; border-bottom: 1px solid #1e293b;
    color: #cbd5e1; font-size: 0.88rem;
}

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

@st.cache_resource(show_spinner="Loading skill taxonomy…")
def load_taxonomy():
    return SkillTaxonomy()

@st.cache_resource(show_spinner="Loading skill extractor…")
def load_extractor():
    tax = load_taxonomy()
    return SkillExtractor(tax)

@st.cache_resource(show_spinner="Loading TF-IDF vectorizer…")
def load_tfidf():
    matcher = TfidfMatcher()
    vec_path = os.path.join(ROOT, "models", "tfidf_vectorizer.joblib")
    if os.path.exists(vec_path):
        matcher.load(vec_path)
    return matcher

@st.cache_resource(show_spinner="Loading Sentence-BERT (all-MiniLM-L6-v2)…")
def load_sbert():
    return SemanticMatcher()

@st.cache_resource(show_spinner="Loading ML classifier…")
def load_classifier():
    models_dir = os.path.join(ROOT, "models")
    model_path = os.path.join(models_dir, "best_model.joblib")
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    le_path = os.path.join(models_dir, "label_encoder.joblib")
    info_path = os.path.join(models_dir, "model_info.json")

    if not all(os.path.exists(p) for p in [model_path, scaler_path, le_path]):
        return None, None, None, None

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    le = joblib.load(le_path)
    info = json.load(open(info_path)) if os.path.exists(info_path) else {}
    return model, scaler, le, info

@st.cache_data
def load_scoring_cfg():
    return load_scoring_config()


# ── Analysis engine ────────────────────────────────────────────────────────────

def run_analysis(resume_text: str, jd_text: str) -> dict:
    """
    Full analysis pipeline — no LLM involved.
    Skill extraction, matching, similarity, and classification
    are all computed locally.
    """
    extractor = load_extractor()
    tfidf = load_tfidf()
    sbert = load_sbert()
    model, scaler, le, model_info = load_classifier()
    config = load_scoring_cfg()

    # Parse resume
    resume_parsed = parse_resume(resume_text)
    jd_parsed = parse_job_description(jd_text)

    # Skill extraction
    resume_skills = extractor.extract_flat_skills(resume_text)
    jd_skills_data = extract_jd_skills(jd_text, extractor)

    # Skill matching
    skill_match = match_skills(
        resume_skills,
        jd_skills_data["required"],
        jd_skills_data["preferred"],
    )

    # TF-IDF similarity (Stage 1 baseline)
    resume_clean = clean_for_tfidf(resume_text)
    jd_clean = clean_for_tfidf(jd_text)
    tfidf_sim = tfidf.similarity(resume_clean, jd_clean)

    # SBERT similarity (Stage 2 baseline)
    resume_emb = clean_for_embedding(resume_text)
    jd_emb = clean_for_embedding(jd_text)
    sbert_sim = sbert.similarity(resume_emb, jd_emb)

    # Experience & Education matching (secondary, conservative)
    exp_match = match_experience(jd_text, resume_text)
    edu_match = match_education(jd_text, resume_text)

    # ML classification (Stage 3)
    ml_prediction = None
    ml_probability = None
    ml_model_name = None
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
        pred_idx = model.predict(feat_scaled)[0]
        proba = model.predict_proba(feat_scaled)[0]
        ml_prediction = le.inverse_transform([pred_idx])[0]
        ml_proba_value = float(max(proba))
        ml_probability = float(proba[pred_idx])
        ml_model_name = model_info.get("best_model", "ML Model")

    # Hybrid scoring
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

    # Resume quality (secondary, ATS-oriented checks)
    quality = analyze_resume_quality(
        text=resume_text,
        sections=resume_parsed["sections"],
        contact=resume_parsed["contact"],
        word_count=len(resume_text.split()),
        page_count=1,
    )

    return {
        "score": score_result,
        "tfidf_sim": tfidf_sim,
        "sbert_sim": sbert_sim,
        "resume_skills": resume_skills,
        "jd_required": jd_skills_data["required"],
        "jd_preferred": jd_skills_data["preferred"],
        "skill_match": skill_match,
        "exp_match": exp_match,
        "edu_match": edu_match,
        "ml_prediction": ml_prediction,
        "ml_probability": ml_probability,
        "ml_model_name": ml_model_name,
        "resume_parsed": resume_parsed,
        "quality": quality,
    }


# ── Helpers ────────────────────────────────────────────────────────────────────

def score_color(score: float) -> str:
    if score >= 80:
        return "#34d399"
    elif score >= 60:
        return "#fbbf24"
    return "#f87171"


def render_skill_tags(skills: list, css_class: str):
    if not skills:
        st.markdown("<span style='color:#64748b; font-size:0.85rem'>None detected</span>",
                    unsafe_allow_html=True)
        return
    tags = " ".join(
        f'<span class="skill-tag {css_class}">{s}</span>' for s in skills
    )
    st.markdown(tags, unsafe_allow_html=True)


def pct(v: float) -> str:
    return f"{v * 100:.1f}%"


# ── Main UI ────────────────────────────────────────────────────────────────────

def main():
    st.markdown('<div class="hero-title">🎯 Resume–Job Match Analyzer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">AI-powered NLP + ML matching · TF-IDF · Sentence-BERT · Skill Gap Analysis</div>',
        unsafe_allow_html=True,
    )

    # Check model availability
    model, _, _, model_info = load_classifier()
    if model is None:
        st.markdown("""
        <div class="warning-box">
        ⚠️ <strong>ML classifier not found.</strong>
        Run <code>python3 scripts/train_models.py</code> to train the models.
        The app will still compute TF-IDF similarity, Sentence-BERT similarity,
        skill gap analysis, and a heuristic score without the classifier.
        </div>
        """, unsafe_allow_html=True)

    # ── Input section ──
    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown("### 📄 Resume")
        uploaded = st.file_uploader(
            "Upload PDF resume", type=["pdf"],
            help="Upload a text-extractable PDF resume",
        )
        resume_text_input = st.text_area(
            "Or paste resume text directly",
            height=200,
            placeholder="Paste your resume text here if you don't have a PDF…",
        )

    with col_r:
        st.markdown("### 📋 Job Description")
        jd_text = st.text_area(
            "Paste the job description",
            height=300,
            placeholder="Paste the full job description here…",
        )

    analyze_btn = st.button("🚀 Analyze Resume", use_container_width=True)

    # ── Extract resume text ──
    resume_text = ""
    if uploaded is not None:
        pdf_result = extract_text_from_pdf_bytes(uploaded.read())
        if pdf_result["error"]:
            st.error(f"PDF Error: {pdf_result['error']}")
        else:
            resume_text = pdf_result["text"]
            st.success(f"✅ PDF extracted: {pdf_result['word_count']} words, "
                       f"{pdf_result['page_count']} page(s)")
    elif resume_text_input.strip():
        resume_text = resume_text_input.strip()

    # ── Analysis ──
    if analyze_btn:
        if not resume_text:
            st.error("Please upload a PDF resume or paste resume text.")
            return
        if not jd_text.strip():
            st.error("Please paste a job description.")
            return

        if len(resume_text.split()) < 30:
            st.warning("Resume text seems very short. Results may be inaccurate.")
        if len(jd_text.split()) < 20:
            st.warning("Job description seems very short. Results may be inaccurate.")

        with st.spinner("Running full analysis pipeline…"):
            results = run_analysis(resume_text, jd_text)

        st.session_state["results"] = results
        st.session_state["resume_text"] = resume_text
        st.session_state["jd_text"] = jd_text

    # ── Results Dashboard ──
    if "results" in st.session_state:
        r = st.session_state["results"]
        score = r["score"]["overall_score"]
        category = r["score"]["category"]
        color = score_color(score)

        st.divider()
        st.markdown("## 📊 Analysis Results")

        # Row 1: Overall score + KPIs
        col_score, col_kpis = st.columns([1, 2], gap="large")

        with col_score:
            st.markdown(f"""
            <div class="score-ring">
                <div class="score-number" style="color:{color}">{score}</div>
                <div class="score-label">Overall Match Score / 100</div>
                <div class="score-category" style="color:{color}">{category}</div>
                <div style="color:#64748b; font-size:0.75rem; margin-top:0.5rem; text-align:center">
                    Application-defined heuristic score.<br>Not an industry-standard hiring metric.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_kpis:
            st.markdown("#### Score Breakdown")
            breakdown = r["score"]["breakdown"]

            metrics = [
                ("Required Skill Coverage", breakdown["required_skill_coverage"]["raw"],
                 "#a78bfa", 100),
                ("Semantic Similarity (SBERT)", breakdown["semantic_similarity"]["raw"],
                 "#60a5fa", 100),
                ("TF-IDF Lexical Similarity", breakdown["tfidf_similarity"]["raw"],
                 "#34d399", 100),
                ("Experience Relevance", breakdown["experience_relevance"]["raw"],
                 "#fb923c", 100),
                ("Education Relevance", breakdown["education_relevance"]["raw"],
                 "#e879f9", 100),
                ("Preferred Skill Coverage", breakdown["preferred_skill_coverage"]["raw"],
                 "#facc15", 100),
            ]
            for label, val, color_c, max_v in metrics:
                st.markdown(f"""
                <div class="kpi-card" style="border-color:{color_c}">
                  <div class="kpi-label">{label}</div>
                  <div class="kpi-value" style="color:{color_c}">{val:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

        # Row 2: ML model result
        if r["ml_prediction"]:
            st.divider()
            st.markdown("#### 🤖 ML Classification")
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.metric("Predicted Fit", r["ml_prediction"])
            with mc2:
                st.metric("Model Probability", f"{r['ml_probability']:.2f}")
            with mc3:
                st.metric("Model", r["ml_model_name"])
            st.caption(
                "⚠️ Model probability is the classifier's output, not a confirmed hiring probability. "
                "Calibration may vary."
            )

        # Row 3: Skill Gap Analysis
        st.divider()
        st.markdown("### 🔍 Skill Gap Analysis")

        sg1, sg2, sg3 = st.columns(3)
        with sg1:
            st.markdown('<div class="section-title">✅ Matched Required Skills</div>',
                        unsafe_allow_html=True)
            render_skill_tags(r["skill_match"]["matched_required"], "skill-matched")
        with sg2:
            st.markdown('<div class="section-title">❌ Missing Required Skills</div>',
                        unsafe_allow_html=True)
            render_skill_tags(r["skill_match"]["missing_required"], "skill-missing")
        with sg3:
            st.markdown('<div class="section-title">⭐ Missing Preferred Skills</div>',
                        unsafe_allow_html=True)
            render_skill_tags(r["skill_match"]["missing_preferred"], "skill-preferred")

        # Resume skills detected
        with st.expander("📋 All Skills Detected in Resume"):
            render_skill_tags(r["resume_skills"], "skill-matched")

        # Row 4: Experience & Education
        st.divider()
        st.markdown("### 👤 Profile Matching")
        pc1, pc2 = st.columns(2)
        with pc1:
            exp = r["exp_match"]
            st.markdown('<div class="section-title">Experience</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="info-box">
              <div><strong>JD Requirement:</strong> {exp['jd_requirement']}</div>
              <div><strong>Resume:</strong> {exp['resume_experience']}</div>
              <div style="color:#94a3b8; margin-top:0.4rem">ℹ️ {exp['note']}</div>
            </div>
            """, unsafe_allow_html=True)
        with pc2:
            edu = r["edu_match"]
            st.markdown('<div class="section-title">Education</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="info-box">
              <div><strong>JD Requirement:</strong> {edu['jd_requirement']}</div>
              <div><strong>Resume:</strong> {edu['resume_degree']}</div>
              <div style="color:#94a3b8; margin-top:0.4rem">ℹ️ {edu['note']}</div>
            </div>
            """, unsafe_allow_html=True)

        # Row 5: ATS-Oriented Resume Checks
        st.divider()
        st.markdown("### 📝 ATS-Oriented Resume Checks")
        st.caption(
            "These are heuristic quality signals. Not a universal ATS score."
        )
        q = r["quality"]
        qa1, qa2, qa3, qa4 = st.columns(4)
        checks = [
            ("Section Completeness", q["section_completeness"]["score"] >= 0.75,
             f"{q['section_completeness']['score']*100:.0f}%"),
            ("Contact Info", q["contact_info"]["score"] >= 0.5,
             f"{q['contact_info']['score']*100:.0f}%"),
            ("Quantified Results", q["quantification"]["has_quantified_achievements"],
             "Yes" if q["quantification"]["has_quantified_achievements"] else "No"),
            ("Action Verbs", q["action_verbs"]["has_action_verbs"],
             f"{len(q['action_verbs']['found'])} found"),
        ]
        for col, (label, is_ok, value) in zip([qa1, qa2, qa3, qa4], checks):
            icon = "✅" if is_ok else "⚠️"
            col.metric(f"{icon} {label}", value)

        # Row 6: Recommendations
        st.divider()
        st.markdown("### 💡 Recommendations")
        recs = _build_recommendations(r)
        for rec in recs:
            st.markdown(f'<div class="rec-item">• {rec}</div>', unsafe_allow_html=True)

        # Scoring methodology disclosure
        with st.expander("ℹ️ Scoring Methodology"):
            weights = r["score"]["weights_used"]
            thresholds = r["score"]["thresholds_used"]
            st.markdown(f"""
**Scoring Weights** (application-defined heuristics, configurable in `config/scoring.yaml`):
| Component | Weight |
|-----------|--------|
| Required Skill Coverage | {weights['required_skill_coverage']:.0%} |
| Semantic Similarity (SBERT) | {weights['semantic_similarity']:.0%} |
| TF-IDF Lexical Similarity | {weights['tfidf_similarity']:.0%} |
| Experience Relevance | {weights['experience_relevance']:.0%} |
| Education Relevance | {weights['education_relevance']:.0%} |
| Preferred Skill Coverage | {weights['preferred_skill_coverage']:.0%} |
| ML Model Confidence | {weights['ml_model_confidence']:.0%} |

**Match Thresholds** (configurable): Strong ≥ {thresholds['strong_match']}, Potential ≥ {thresholds['potential_match']}, Weak < {thresholds['potential_match']}

These weights are NOT industry-standard. They are heuristic application parameters.
            """)


def _build_recommendations(r: dict) -> list:
    """Generate actionable recommendations from analysis results."""
    recs = []

    missing_req = r["skill_match"]["missing_required"]
    if missing_req:
        top_missing = missing_req[:4]
        recs.append(
            f"<strong>Add missing required skills</strong> to your resume: "
            f"{', '.join(top_missing)}" +
            (f" (and {len(missing_req)-4} more)" if len(missing_req) > 4 else "")
        )

    if r["tfidf_sim"] < 0.3:
        recs.append(
            "<strong>Increase keyword alignment</strong>: Your resume has low lexical overlap "
            "with the job description. Mirror the JD's terminology where truthfully applicable."
        )

    if r["sbert_sim"] < 0.4:
        recs.append(
            "<strong>Improve semantic alignment</strong>: The resume and JD have low "
            "semantic similarity. Consider adding more context about your relevant experience."
        )

    q = r["quality"]
    if not q["quantification"]["has_quantified_achievements"]:
        recs.append(
            "<strong>Add quantified achievements</strong>: Use numbers, percentages, "
            "or metrics to describe your impact (e.g., 'reduced latency by 40%')."
        )

    if not q["action_verbs"]["has_action_verbs"]:
        recs.append(
            "<strong>Use action verbs</strong>: Start bullet points with strong verbs "
            "like 'built', 'optimized', 'deployed', 'reduced'."
        )

    if q["contact_info"]["score"] < 0.5:
        missing_contact = q["contact_info"]["missing"]
        recs.append(
            f"<strong>Complete contact information</strong>: Add {', '.join(missing_contact)} "
            "to your resume header."
        )

    missing_pref = r["skill_match"]["missing_preferred"]
    if missing_pref:
        recs.append(
            f"<strong>Consider adding preferred skills</strong>: "
            f"{', '.join(missing_pref[:3])} are listed as preferred qualifications."
        )

    if not recs:
        recs.append("✨ Strong application — good alignment with the job description.")

    return recs


if __name__ == "__main__":
    main()
