"""
Personalized Resume Suggestion Engine (Phase 2)

Generates actionable, resume-specific recommendations grounded in:
  - The actual resume text, skills, and structure
  - The specific job description (when provided)
  - The selected target role and career level
  - Existing quality-audit results and skill-gap analysis

Design principles:
  - Every suggestion references concrete evidence from the resume or JD
  - No duplicate suggestions within a single analysis
  - No invented skills, qualifications, or experience
  - Role-aware: recommendations adapt to the selected target role
  - Strictly separate from the quality-audit (different purpose/audience)
  - No external APIs or paid services; pure rule + NLP signals

Output: list of dicts, each with:
  {
    "category":    str,   # grouping label (e.g. "Skills", "Keywords", "Impact")
    "priority":    str,   # "High" | "Medium" | "Low"
    "title":       str,   # short headline
    "rationale":   str,   # WHY this matters for THIS resume + role
    "action":      str,   # specific, concrete step to take
    "evidence":    str,   # quote or fact from resume/JD that triggered it
  }
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Set


# ── Role-specific high-value keywords ──────────────────────────────────────────
# Used when no JD is provided. Keys are substrings of TARGET_ROLES.
ROLE_KEYWORDS: Dict[str, List[str]] = {
    "data scientist": [
        "machine learning", "statistics", "python", "sql", "model evaluation",
        "feature engineering", "experiment design", "a/b testing", "pandas", "scikit-learn",
    ],
    "machine learning": [
        "pytorch", "tensorflow", "model training", "hyperparameter tuning",
        "mlops", "pipeline", "feature engineering", "gpu", "distributed training",
    ],
    "data analyst": [
        "sql", "tableau", "power bi", "excel", "python", "statistics",
        "dashboards", "data visualization", "business intelligence", "kpis",
    ],
    "software engineer": [
        "algorithms", "data structures", "system design", "rest api",
        "unit testing", "ci/cd", "git", "code review", "scalability",
    ],
    "backend developer": [
        "rest api", "databases", "microservices", "docker", "sql", "authentication",
        "performance optimization", "message queues", "caching", "server-side",
    ],
    "frontend developer": [
        "react", "javascript", "typescript", "css", "html", "responsive design",
        "accessibility", "web performance", "state management", "testing",
    ],
    "full stack": [
        "rest api", "databases", "frontend", "backend", "docker",
        "deployment", "authentication", "git", "testing",
    ],
    "devops": [
        "kubernetes", "docker", "ci/cd", "terraform", "monitoring",
        "aws", "linux", "scripting", "infrastructure as code",
    ],
    "cloud": [
        "aws", "azure", "gcp", "kubernetes", "terraform", "iam",
        "networking", "cost optimization", "serverless", "containers",
    ],
    "ai engineer": [
        "llm", "prompt engineering", "langchain", "embeddings", "vector database",
        "fine-tuning", "rag", "mlops", "python", "hugging face",
    ],
    "product manager": [
        "roadmap", "user research", "kpis", "agile", "stakeholder management",
        "go-to-market", "prioritization", "metrics", "wireframes",
    ],
    "cybersecurity": [
        "penetration testing", "siem", "vulnerability", "incident response",
        "firewalls", "identity", "encryption", "compliance", "soc",
    ],
    "qa engineer": [
        "test automation", "selenium", "test cases", "bug tracking",
        "regression testing", "api testing", "performance testing", "ci/cd",
    ],
}

QUANTIFICATION_PATTERNS = [
    r"\d+%",
    r"\d+x\b",
    r"\$[\d,]+",
    r"\d+\+?\s*(?:users?|customers?|clients?)",
    r"\d+\s*(?:million|billion|thousand|k\b)",
    r"(?:reduced|improved|increased|decreased)\s+by\s+\d+",
]

WEAK_VERB_PATTERNS = [
    r"\bresponsible for\b",
    r"\bduties included?\b",
    r"\bassisted with\b",
    r"\binvolved in\b",
    r"\bparticipated in\b",
    r"\bworked on\b",
    r"\bhelped with\b",
    r"\bhelped to\b",
    r"\bwas tasked with\b",
]

IMPACT_ACTION_VERBS = [
    "reduced", "increased", "improved", "accelerated", "saved", "delivered",
    "launched", "scaled", "optimized", "automated", "eliminated", "boosted",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_tfidf_high_idf_keywords(text: str, n: int = 30) -> List[str]:
    """
    Return the top-N most distinctive (high character-weight) technical
    tokens from text. Lightweight proxy for IDF weighting when no vectorizer
    is available: longer, rarer technical words score higher.
    """
    words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9\+\#\.\-]{2,}\b", text.lower())
    stopwords = {
        "and", "the", "for", "with", "that", "this", "are", "will", "have",
        "from", "your", "you", "our", "its", "has", "can", "not", "all",
        "any", "was", "were", "been", "would", "should", "their", "than",
        "more", "also", "both", "each", "who", "what", "how", "but", "we",
        "is", "in", "of", "to", "a", "on", "by", "at", "an", "as", "or",
        "be", "do", "if", "it", "no", "so", "up", "he", "she", "they",
        "them", "his", "her", "him", "into", "over", "out", "about",
        "skills", "experience", "work", "team", "using", "strong", "good",
        "years", "role", "company", "position", "job", "responsibilities",
        "required", "preferred", "ability", "knowledge", "understanding",
    }
    freq: Dict[str, int] = {}
    for w in words:
        if w not in stopwords and len(w) > 3:
            freq[w] = freq.get(w, 0) + 1

    # Score: frequency in JD but NOT in resume → missing keywords
    return sorted(freq, key=lambda w: (len(w) * freq[w]), reverse=True)[:n]


def _find_missing_keywords(resume_text: str, jd_text: str, top_n: int = 8) -> List[str]:
    """
    Find distinctive JD keywords absent from the resume.
    Returns the top_n most prominent missing terms.
    """
    jd_keywords = set(_extract_tfidf_high_idf_keywords(jd_text, n=50))
    resume_lower = resume_text.lower()
    missing = [kw for kw in jd_keywords if kw not in resume_lower]
    return missing[:top_n]


def _get_role_missing_keywords(resume_text: str, target_role: str) -> List[str]:
    """Return expected keywords for a role that are missing from the resume."""
    role_lower = target_role.lower()
    matched_role = None
    for key in ROLE_KEYWORDS:
        if key in role_lower:
            matched_role = key
            break
    if matched_role is None:
        return []
    resume_lower = resume_text.lower()
    return [kw for kw in ROLE_KEYWORDS[matched_role] if kw not in resume_lower]


def _get_experience_bullets(sections: Dict[str, str]) -> List[str]:
    """Extract bullet lines from experience/projects sections."""
    bullets = []
    for sec in ["experience", "projects", "internships"]:
        if sec in sections:
            for line in sections[sec].split("\n"):
                clean = re.sub(r"^[\s•\-–*▸▶►]+", "", line).strip()
                if len(clean.split()) >= 4:
                    bullets.append(clean)
    return bullets


def _has_quantification(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in QUANTIFICATION_PATTERNS)


def _has_weak_verb(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in WEAK_VERB_PATTERNS)


def _first_verb(bullet: str) -> str:
    """Return the first word of a bullet, lowercased."""
    words = bullet.split()
    return words[0].lower().rstrip(".,;:") if words else ""


# ── Core suggestion generator ──────────────────────────────────────────────────

def generate_suggestions(
    resume_text: str,
    resume_parsed: Dict,
    resume_skills: List[str],
    target_role: str,
    career_level: str,
    audit_result: Dict,
    # Optional JD-mode inputs
    has_jd: bool = False,
    jd_text: str = "",
    jd_required_skills: Optional[List[str]] = None,
    jd_preferred_skills: Optional[List[str]] = None,
    skill_match: Optional[Dict] = None,
    tfidf_sim: Optional[float] = None,
    sbert_sim: Optional[float] = None,
    score_result: Optional[Dict] = None,
    # Context helpers
    max_suggestions: int = 10,
) -> List[Dict]:
    """
    Generate personalized, evidence-backed suggestions for this resume.

    Each suggestion is unique, grounded in the actual resume/JD content,
    and avoids repeating what the quality-audit already covers.
    """
    suggestions: List[Dict] = []
    seen_titles: Set[str] = set()

    def add(category: str, priority: str, title: str,
            rationale: str, action: str, evidence: str = ""):
        if title in seen_titles:
            return
        seen_titles.add(title)
        suggestions.append({
            "category": category,
            "priority": priority,
            "title": title,
            "rationale": rationale,
            "action": action,
            "evidence": evidence,
        })

    sections = resume_parsed.get("sections", {})
    bullets = _get_experience_bullets(sections)
    resume_lower = resume_text.lower()

    # ── 1. Missing Required Skills (JD mode) ─────────────────────────────
    if has_jd and skill_match and jd_required_skills:
        missing_req = skill_match.get("missing_required", [])
        if missing_req:
            top_missing = missing_req[:5]
            add(
                category="Skills Gap",
                priority="High",
                title=f"Add {len(missing_req)} Missing Required Skills to Your Resume",
                rationale=(
                    f"The job description explicitly requires "
                    f"{', '.join(top_missing[:3])} "
                    f"{'and more' if len(top_missing) > 3 else ''}. "
                    f"Your resume does not mention these, which will likely cause ATS rejection."
                ),
                action=(
                    f"If you genuinely have experience with these skills, add them clearly "
                    f"to your Skills section and support each with a project or achievement: "
                    f"{', '.join(top_missing)}."
                ),
                evidence=f"Missing required skills: {', '.join(missing_req[:6])}",
            )

        # Per-skill recommendation for top-3 missing required
        for skill in missing_req[:3]:
            add(
                category="Skills Gap",
                priority="High",
                title=f"Demonstrate Your {skill.title()} Experience",
                rationale=(
                    f"The JD lists '{skill}' as a required skill. "
                    f"It does not appear in your resume — even if you have used it in a project."
                ),
                action=(
                    f"If applicable: add a specific bullet such as "
                    f"\"Used {skill} to [achieve concrete outcome] in [project name].\" "
                    f"Only add this if it accurately reflects your experience."
                ),
                evidence=f"'{skill}' required by JD, absent from resume.",
            )

    # ── 2. Missing Preferred Skills (JD mode) ────────────────────────────
    if has_jd and skill_match:
        missing_pref = skill_match.get("missing_preferred", [])
        if missing_pref:
            top_pref = missing_pref[:4]
            add(
                category="Skills Gap",
                priority="Medium",
                title="Consider Highlighting Preferred Skills You Already Have",
                rationale=(
                    f"The JD lists preferred (nice-to-have) skills: "
                    f"{', '.join(top_pref)}. Candidates who can demonstrate these "
                    f"stand out over those who only meet required criteria."
                ),
                action=(
                    f"If you have any experience with {', '.join(top_pref[:2])}, "
                    f"add them to your Skills section or mention them in a project bullet."
                ),
                evidence=f"Preferred by JD but not found in resume: {', '.join(top_pref)}",
            )

    # ── 3. Missing JD Keywords (JD mode) ─────────────────────────────────
    if has_jd and jd_text and tfidf_sim is not None and tfidf_sim < 0.5:
        missing_kw = _find_missing_keywords(resume_text, jd_text, top_n=6)
        if missing_kw:
            add(
                category="Keyword Alignment",
                priority="High" if tfidf_sim < 0.3 else "Medium",
                title="Mirror Key Terms from the Job Description",
                rationale=(
                    f"Your resume has low keyword overlap with this JD "
                    f"(lexical similarity: {tfidf_sim*100:.0f}%). "
                    f"ATS systems score resumes by keyword frequency — missing the exact "
                    f"terminology reduces your chances of passing screening."
                ),
                action=(
                    f"Where truthfully applicable, incorporate these JD-specific terms "
                    f"into your resume: {', '.join(missing_kw)}. "
                    f"Use them in context (project bullets, skills list) — do not keyword-stuff."
                ),
                evidence=f"Terms prominent in JD but absent from resume: {', '.join(missing_kw)}",
            )

    # ── 4. Semantic Alignment Gap (JD mode) ──────────────────────────────
    if has_jd and sbert_sim is not None and sbert_sim < 0.45:
        add(
            category="Content Alignment",
            priority="Medium",
            title="Reframe Your Experience to Match This Role's Language",
            rationale=(
                f"Semantic similarity between your resume and the JD is low "
                f"({sbert_sim*100:.0f}%). This suggests your experience, while potentially "
                f"relevant, is described using different vocabulary and framing than the role requires."
            ),
            action=(
                f"Re-read the JD's 'Responsibilities' section. For each major responsibility, "
                f"check if your experience section has a matching bullet. If yes, rewrite it "
                f"using the JD's framing. For example: if the JD says 'building scalable pipelines', "
                f"your bullet should say 'built/designed [scalable pipeline]' not just 'worked on data'."
            ),
            evidence=f"Semantic similarity: {sbert_sim*100:.0f}%",
        )

    # ── 5. Role-Specific Missing Keywords (no JD / always) ───────────────
    role_missing = _get_role_missing_keywords(resume_text, target_role)
    if role_missing:
        top_role_missing = role_missing[:5]
        add(
            category="Role Alignment",
            priority="Medium",
            title=f"Add Key {target_role} Skills to Your Resume",
            rationale=(
                f"Strong candidates for {target_role} roles typically demonstrate "
                f"proficiency in: {', '.join(top_role_missing)}. "
                f"None of these appear clearly in your resume."
            ),
            action=(
                f"If you have experience with any of these, add them to your Skills section "
                f"and reference them in a project or work bullet: "
                f"{', '.join(top_role_missing)}."
            ),
            evidence=f"Expected for {target_role} role, not detected in resume.",
        )

    # ── 6. Weak / Duty-Based Bullet Improvement ───────────────────────────
    weak_bullets = [b for b in bullets if _has_weak_verb(b)]
    if weak_bullets:
        example = weak_bullets[0][:120]
        add(
            category="Impact",
            priority="High",
            title="Rewrite Duty-Based Bullets as Achievement Statements",
            rationale=(
                f"Your resume contains {len(weak_bullets)} bullet(s) that describe job "
                f"duties rather than personal contributions. Recruiters scan for 'what you built' "
                f"not 'what you were responsible for'. Duty-based phrasing signals a passive role."
            ),
            action=(
                f"Rewrite each duty bullet using the format: "
                f"[Strong Verb] + [What you built/did] + [For whom / scale] + [Outcome/metric]. "
                f"Example fix: \"{example}…\" → \"Built/Engineered [X] that reduced [Y] by [Z%].\""
            ),
            evidence=f"Found: \"{example}\"",
        )

    # ── 7. Quantification opportunity ────────────────────────────────────
    unquantified_impact_bullets = [
        b for b in bullets
        if not _has_quantification(b)
        and any(v in b.lower() for v in IMPACT_ACTION_VERBS)
        and len(b.split()) >= 8
    ]
    if len(unquantified_impact_bullets) >= 2:
        example_bullets = unquantified_impact_bullets[:2]
        add(
            category="Impact",
            priority="Medium",
            title="Strengthen Impact Bullets with Measurable Outcomes",
            rationale=(
                f"You have {len(unquantified_impact_bullets)} impact-oriented bullets "
                f"(using verbs like 'reduced', 'improved', 'launched') but none include "
                f"a number, percentage, or scale. Quantified achievements are 2–3x more "
                f"memorable to recruiters and more credible."
            ),
            action=(
                f"For each of these bullets, ask: How many users? What % improvement? "
                f"How much time saved? Add the metric if you can support it. "
                f"Examples to strengthen: "
                + " | ".join(f'\"{b[:80]}\"' for b in example_bullets)
            ),
            evidence=f"{len(unquantified_impact_bullets)} impact bullets lack quantification.",
        )

    # ── 8. Repeated first verbs ──────────────────────────────────────────
    from collections import Counter
    ACTION_VERBS = {
        "developed", "implemented", "built", "designed", "created", "engineered",
        "optimized", "improved", "reduced", "increased", "automated", "deployed",
        "architected", "led", "managed", "analyzed", "researched", "trained",
    }
    verb_starts = [_first_verb(b) for b in bullets if _first_verb(b) in ACTION_VERBS]
    verb_counts = Counter(verb_starts)
    repeated = [(v, c) for v, c in verb_counts.most_common() if c >= 3]
    if repeated:
        v, c = repeated[0]
        alternatives = [w for w in IMPACT_ACTION_VERBS if w != v][:3]
        add(
            category="Writing Quality",
            priority="Low",
            title=f"Diversify Action Verbs ('{v.capitalize()}' used {c}×)",
            rationale=(
                f"Starting {c} bullets with \"{v.capitalize()}\" creates a repetitive, "
                f"monotonous reading experience. Varied verbs demonstrate a broader range of "
                f"contributions and keep recruiters engaged."
            ),
            action=(
                f"Replace some occurrences of '{v}' with more specific verbs: "
                f"{', '.join(alternatives)}. Choose the verb that most accurately describes "
                f"what you actually did."
            ),
            evidence=f"'{v.capitalize()}' starts {c} bullets in your resume.",
        )

    # ── 9. Missing GitHub / Portfolio Link ───────────────────────────────
    contact = resume_parsed.get("contact", {})
    has_github = bool(contact.get("github")) or "github" in resume_lower
    has_portfolio = "portfolio" in resume_lower or "behance" in resume_lower or "dribbble" in resume_lower
    if not has_github and target_role.lower() not in ["product manager", "hr", "human resources",
                                                        "talent acquisition / recruiter",
                                                        "financial analyst", "marketing analyst"]:
        add(
            category="Visibility",
            priority="Medium",
            title="Add Your GitHub Profile (or Portfolio URL)",
            rationale=(
                f"For a {target_role} role, a GitHub profile (or equivalent portfolio) "
                f"provides immediate, verifiable evidence of your technical skills. "
                f"Recruiters increasingly check these before interviews."
            ),
            action=(
                "Add your GitHub URL to your resume header (e.g., github.com/yourhandle). "
                "If your repos are private, make at least 2–3 representative ones public "
                "with a clear README explaining the project and your contribution."
            ),
            evidence="No GitHub or portfolio URL detected in resume.",
        )

    # ── 10. Career-level specific: Summary missing ────────────────────────
    if career_level == "Experienced Professional":
        has_summary = "summary" in sections or "objective" in sections or "profile" in sections
        if not has_summary:
            add(
                category="Structure",
                priority="Medium",
                title=f"Add a Targeted Professional Summary for {target_role} Roles",
                rationale=(
                    "Experienced candidates are expected to open with a 2–3 sentence summary "
                    "that directly answers: Who are you? What's your specialization? "
                    "What value do you bring to this specific role? Without it, recruiters "
                    "must hunt through your resume to understand your positioning."
                ),
                action=(
                    f"Write a 2–3 sentence summary at the top of your resume tailored to "
                    f"the {target_role} role. For example: "
                    f"\"[X]-year [role] with expertise in [top 2 skills]. "
                    f"Delivered [key achievement]. Seeking [specific opportunity].\""
                ),
                evidence="No professional summary section detected.",
            )

    # ── 11. Project specificity (no JD, fresher/student) ─────────────────
    if career_level == "Fresher / Student" and "projects" in sections:
        proj_text = sections["projects"]
        proj_bullets = [b for b in bullets if b in proj_text]
        vague_proj = [b for b in proj_bullets if len(b.split()) < 12]
        if len(vague_proj) >= 2:
            add(
                category="Projects",
                priority="Medium",
                title="Expand Project Descriptions with Technical Specificity",
                rationale=(
                    f"Your project section has {len(vague_proj)} brief bullets "
                    f"(under 12 words). For a student/fresher profile, well-described projects "
                    f"are your primary evidence of competence — they substitute for work experience."
                ),
                action=(
                    "For each project, answer: What was the problem? What did you build? "
                    "Which technologies did you use (be specific — e.g., 'React 18 + FastAPI' "
                    "not just 'web app')? What was the outcome or scale? "
                    "Aim for 2–3 lines per project minimum."
                ),
                evidence=f"{len(vague_proj)} project bullet(s) are under 12 words.",
            )

    # ── 12. JD match score context ────────────────────────────────────────
    if has_jd and score_result:
        score = score_result.get("overall_score", 100)
        category_label = score_result.get("category", "")
        if score < 60 and category_label == "Weak Match":
            add(
                category="Overall Strategy",
                priority="High",
                title="This Resume Needs Significant Tailoring for This Role",
                rationale=(
                    f"Your overall match score is {score:.0f}/100 (Weak Match). "
                    f"This means multiple signals — skill coverage, keyword overlap, "
                    f"and semantic alignment — are all below threshold. "
                    f"Submitting this resume as-is will likely result in rejection."
                ),
                action=(
                    "Prioritise these actions: (1) Add the missing required skills you genuinely have. "
                    "(2) Rewrite 3–5 bullets to mirror the JD's core responsibilities. "
                    "(3) Add JD-specific keywords throughout your skills and experience sections. "
                    "(4) Only apply if your background is genuinely relevant to this role."
                ),
                evidence=f"Match score: {score:.0f}/100",
            )

    # Return up to max_suggestions, sorted by priority
    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    suggestions.sort(key=lambda x: priority_order.get(x["priority"], 3))
    return suggestions[:max_suggestions]
