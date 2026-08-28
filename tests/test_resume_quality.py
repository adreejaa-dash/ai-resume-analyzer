"""
Tests for the expanded resume quality audit engine.
"""
import pytest
from src.analysis.resume_quality import analyze_resume_quality


def _run(text, sections, contact, word_count=300, career_level="Fresher / Student",
         target_role="Software Engineer", yoe=None):
    return analyze_resume_quality(
        text=text, sections=sections, contact=contact,
        word_count=word_count, career_level=career_level,
        target_role=target_role, yoe=yoe,
    )


def issue_titles(result):
    return [i["title"] for i in result["issues"]]


# ── Contact Info ──────────────────────────────────────────────────────────────

def test_missing_email_and_phone():
    r = _run("John Doe", {"skills": "Python"}, {"email": None, "phone": None}, word_count=300)
    assert "Missing Essential Contact Info" in issue_titles(r)

def test_contact_ok_no_critical():
    r = _run("John Doe john@x.com 555-1234", {"skills": "Python", "education": "BSc", "experience": "..."},
             {"email": "john@x.com", "phone": "555-1234", "linkedin": "linkedin.com/in/john"}, word_count=300)
    assert "Missing Essential Contact Info" not in issue_titles(r)

def test_personal_info_dob():
    text = "John Doe\nDOB: 1998\nSkills: Python"
    r = _run(text, {"skills": "Python", "education": "BSc", "projects": "x"}, {"email": "j@x.com", "phone": "123"})
    titles = issue_titles(r)
    assert any("Unnecessary Personal Info" in t for t in titles)

def test_personal_info_marital():
    text = "Jane Doe\nMarital Status: Single\nSkills: Java"
    r = _run(text, {"skills": "Java", "education": "BSc", "projects": "x"}, {"email": "j@x.com", "phone": "123"})
    titles = issue_titles(r)
    assert any("Unnecessary Personal Info" in t for t in titles)


# ── Section Structure ─────────────────────────────────────────────────────────

def test_missing_core_sections():
    r = _run("Resume text", {}, {"email": "a@b.com", "phone": "123"}, word_count=300)
    assert "Missing Core Sections" in issue_titles(r)

def test_non_standard_headings():
    r = _run(
        "text",
        {"skills": "x", "education": "y", "experience": "z", "awesomework": "extra"},
        {"email": "a@b.com", "phone": "123"},
    )
    assert "Non-standard Section Headings" in issue_titles(r)

def test_missing_summary_experienced():
    r = _run(
        "text",
        {"skills": "x", "education": "y", "experience": "Developed apps"},
        {"email": "a@b.com", "phone": "123"},
        career_level="Experienced Professional", yoe="3-5 years",
    )
    assert "Missing Professional Summary" in issue_titles(r)


# ── Length ────────────────────────────────────────────────────────────────────

def test_too_short_critical():
    r = _run("Short resume", {"skills": "x"}, {"email": "a@b.com", "phone": "123"}, word_count=50)
    assert "Resume Extremely Short" in issue_titles(r)

def test_fresher_too_long():
    r = _run(
        "x" * 2000,
        {"skills": "x", "education": "y", "projects": "z"},
        {"email": "a@b.com", "phone": "123"},
        word_count=1500, career_level="Fresher / Student",
    )
    assert "Resume Too Long for Fresher / Student" in issue_titles(r)

def test_experienced_very_long():
    r = _run(
        "x" * 4000,
        {"skills": "x", "education": "y", "experience": "z"},
        {"email": "a@b.com", "phone": "123"},
        word_count=2200, career_level="Experienced Professional", yoe="3-5 years",
    )
    assert "Resume Excessively Long" in issue_titles(r)


# ── Bullets ───────────────────────────────────────────────────────────────────

def test_weak_phrasing_responsible_for():
    text = "resume\nExperience:\n- Responsible for maintaining the backend API\n- Helped with deployments"
    r = _run(text, {"skills": "x", "education": "y", "experience": "- Responsible for maintaining the backend API\n- Helped with deployments"},
             {"email": "a@b.com", "phone": "123"}, word_count=400)
    assert "Duty-based Phrasing in Bullets" in issue_titles(r)

# test_long_bullets_flagged and test_repeated_action_verbs are
# tested with correct word counts at the bottom of this file.


# ── Clichés ───────────────────────────────────────────────────────────────────

def test_cliche_detected():
    text = "I am a hardworking team player with excellent communication skills."
    r = _run(text, {"skills": "Python", "education": "BSc", "projects": "x"},
             {"email": "a@b.com", "phone": "123"}, word_count=300)
    assert "Generic Soft Skills / Clichés" in issue_titles(r)

def test_no_cliche_clean_resume():
    text = "Developed scalable APIs. Reduced latency by 30%. Led a team of 4."
    r = _run(
        text,
        {"skills": "Python, API", "education": "BSc", "experience": "- Developed scalable APIs."},
        {"email": "a@b.com", "phone": "123"}, word_count=400,
    )
    assert "Generic Soft Skills / Clichés" not in issue_titles(r)


# ── Repetition & Duplicates ───────────────────────────────────────────────────

def test_duplicate_skills():
    r = _run(
        "text",
        {"skills": "Python Python Java JavaScript", "education": "y", "projects": "z"},
        {"email": "a@b.com", "phone": "123"}, word_count=300,
    )
    assert "Duplicate Skills Listed" in issue_titles(r)


# ── Every issue has required fields ──────────────────────────────────────────

def test_every_issue_has_required_fields():
    text = ("Hardworking team player. DOB: 1997. Responsible for various tasks.\n"
            "Developed x\nDeveloped y\nDeveloped z\nDeveloped w")
    r = _run(
        text,
        {"skills": "Python Python", "education": "BSc", "experience": text},
        {"email": None, "phone": None}, word_count=300,
    )
    for issue in r["issues"]:
        assert issue.get("title"), "Missing title"
        assert issue.get("severity") in ("Critical", "High", "Medium", "Minor"), f"Bad severity: {issue}"
        assert issue.get("location"), "Missing location"
        assert issue.get("why"), "Missing why"
        assert issue.get("solution"), "Missing solution"


# ── Quantification — NOT flagged when bullets are strong ─────────────────────

def test_no_false_quantification_flag_on_strong_bullets():
    """Bullets with clear detail and action verbs should NOT trigger the vague/metric flag."""
    exp = (
        "- Engineered a real-time data pipeline using Apache Kafka and Spark that processes streaming events\n"
        "- Designed and deployed microservices architecture improving system reliability\n"
        "- Refactored legacy codebase to eliminate technical debt and improve maintainability\n"
        "- Automated CI/CD workflows using GitHub Actions reducing manual deployment effort\n"
    )
    r = _run(
        exp,
        {"skills": "Python", "education": "BSc", "experience": exp},
        {"email": "a@b.com", "phone": "123"}, word_count=500,
    )
    titles = issue_titles(r)
    assert "Vague Bullets Without Scale or Outcome (Consider Adding)" not in titles


# ── No-JD mode works ──────────────────────────────────────────────────────────

def test_quality_score_present():
    r = _run("text", {"skills": "x", "education": "y", "experience": "z"},
             {"email": "a@b.com", "phone": "123"})
    assert "quality_score" in r
    assert 0 <= r["quality_score"] <= 100


# ── YOE only for Experienced ─────────────────────────────────────────────────

def test_yoe_not_required_for_fresher():
    """Fresher should not trigger missing professional summary."""
    r = _run(
        "text",
        {"skills": "x", "education": "y", "projects": "z"},
        {"email": "a@b.com", "phone": "123"},
        career_level="Fresher / Student", yoe=None,
    )
    assert "Missing Professional Summary" not in issue_titles(r)


# ── Corrected boundary tests ──────────────────────────────────────────────────

def test_long_bullets_flagged_actual_threshold():
    """Bullet must be >40 words to be flagged. Use a 45-word bullet."""
    long_bullet = (
        "Developed a comprehensive machine learning pipeline that ingested enormous volumes of raw "
        "data from multiple external sources and processed them through a distributed compute layer "
        "and then stored final results in a highly available distributed database for downstream "
        "analytics consumption by business stakeholders"
    )
    assert len(long_bullet.split()) > 40, f"Test bullet must be >40 words, got {len(long_bullet.split())}"
    r = _run(
        "text",
        {"skills": "x", "education": "y", "experience": f"- {long_bullet}"},
        {"email": "a@b.com", "phone": "123"}, word_count=400,
    )
    assert "Overly Long Bullet Points" in issue_titles(r)


def test_repeated_action_verbs_sufficient_words():
    """Bullets need ≥4 words each to pass the filter before verb counting."""
    exp = "\n".join([
        "- Developed a scalable API for users",
        "- Developed a caching layer for performance",
        "- Developed a monitoring dashboard for ops",
        "- Developed a reporting tool for analysts",
    ])
    r = _run(
        "text",
        {"skills": "x", "education": "y", "experience": exp},
        {"email": "a@b.com", "phone": "123"}, word_count=400,
    )
    assert "Repetitive Action Verbs" in issue_titles(r)
