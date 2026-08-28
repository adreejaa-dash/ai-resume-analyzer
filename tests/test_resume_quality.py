import pytest
from src.analysis.resume_quality import analyze_resume_quality

def test_resume_quality_fresher_short():
    text = "John Doe\nSummary: Looking for an entry-level job.\nSkills: Python, Java."
    sections = {"summary": "Looking for an entry-level job.", "skills": "Python, Java."}
    contact = {"email": "john@example.com"}
    word_count = 10
    
    result = analyze_resume_quality(
        text=text,
        sections=sections,
        contact=contact,
        word_count=word_count,
        career_level="Fresher / Student",
        target_role="Software Engineer",
        yoe=None
    )
    
    assert "quality_score" in result
    assert "issues" in result
    
    issues_titles = [i["title"] for i in result["issues"]]
    assert "Missing Core Sections" in issues_titles
    assert "Resume Too Short" in issues_titles

def test_resume_quality_weak_verbs_cliches():
    text = "Jane Doe\nSummary: Hardworking and a team player.\nExperience: \n- Responsible for updating website.\n- Helped with the database."
    sections = {"summary": "Hardworking and a team player.", "experience": "- Responsible for updating website.\n- Helped with the database."}
    contact = {"email": "jane@example.com", "phone": "123-456-7890"}
    word_count = 200
    
    result = analyze_resume_quality(
        text=text,
        sections=sections,
        contact=contact,
        word_count=word_count,
        career_level="Experienced Professional",
        target_role="Frontend Developer",
        yoe="3-5 years"
    )
    
    issues_titles = [i["title"] for i in result["issues"]]
    assert "Generic Soft Skills (Clichés)" in issues_titles
    assert "Weak Phrasing ('responsible for', etc.)" in issues_titles
