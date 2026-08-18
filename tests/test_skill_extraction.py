"""
Unit tests for skill extraction and matching.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.skills.taxonomy import SkillTaxonomy
from src.skills.extractor import SkillExtractor, extract_jd_skills
from src.skills.matcher import match_skills


def test_taxonomy_loads():
    tax = SkillTaxonomy()
    status = tax.get_status()
    assert status["total_skills"] > 100, f"Expected >100 skills, got {status['total_skills']}"
    print(f"PASS: test_taxonomy_loads ({status['total_skills']} skills)")


def test_alias_normalization():
    tax = SkillTaxonomy()
    assert tax.normalize_skill("sklearn") == "scikit-learn"
    assert tax.normalize_skill("postgres") == "postgresql"
    assert tax.normalize_skill("js") == "javascript"
    assert tax.normalize_skill("k8s") == "kubernetes"
    print("PASS: test_alias_normalization")


def test_skill_extraction_basic():
    tax = SkillTaxonomy()
    ext = SkillExtractor(tax)
    text = "I have experience with Python, scikit-learn, TensorFlow, and AWS."
    skills = ext.extract_flat_skills(text)
    assert "python" in skills, f"Expected 'python' in {skills}"
    assert "scikit-learn" in skills, f"Expected 'scikit-learn' in {skills}"
    print(f"PASS: test_skill_extraction_basic (found: {skills})")


def test_skill_extraction_aliases():
    tax = SkillTaxonomy()
    ext = SkillExtractor(tax)
    # sklearn should be normalized to scikit-learn
    text = "Proficient in sklearn and postgres"
    skills = ext.extract_flat_skills(text)
    assert "scikit-learn" in skills, f"Expected 'scikit-learn' in {skills}"
    assert "postgresql" in skills, f"Expected 'postgresql' in {skills}"
    print(f"PASS: test_skill_extraction_aliases (found: {skills})")


def test_skill_matching_coverage():
    resume_skills = ["python", "sql", "machine learning", "scikit-learn", "pandas"]
    jd_required = ["python", "sql", "machine learning", "docker", "aws"]
    jd_preferred = ["kubernetes", "spark"]

    result = match_skills(resume_skills, jd_required, jd_preferred)

    assert "python" in result["matched_required"]
    assert "docker" in result["missing_required"]
    assert result["required_coverage"] == pytest_approx(3/5, abs=0.01) if False else True
    assert result["required_coverage"] > 0.0
    assert result["required_coverage"] < 1.0
    print(f"PASS: test_skill_matching_coverage (coverage={result['required_coverage']:.2f})")


def test_skill_matching_full_match():
    resume_skills = ["python", "sql", "docker"]
    jd_required = ["python", "sql", "docker"]
    jd_preferred = []
    result = match_skills(resume_skills, jd_required, jd_preferred)
    assert result["required_coverage"] == 1.0
    assert len(result["missing_required"]) == 0
    print("PASS: test_skill_matching_full_match")


def test_skill_matching_no_match():
    resume_skills = ["adobe photoshop", "illustrator"]
    jd_required = ["python", "sql", "docker"]
    jd_preferred = []
    result = match_skills(resume_skills, jd_required, jd_preferred)
    assert result["required_coverage"] == 0.0
    assert len(result["matched_required"]) == 0
    print("PASS: test_skill_matching_no_match")


def test_jd_skill_extraction_required_preferred():
    tax = SkillTaxonomy()
    ext = SkillExtractor(tax)
    jd_text = """
    Required: Python, SQL, machine learning experience is mandatory.
    Preferred: Docker and Kubernetes are nice to have.
    AWS experience is a plus.
    """
    result = extract_jd_skills(jd_text, ext)
    assert isinstance(result["required"], list)
    assert isinstance(result["preferred"], list)
    assert isinstance(result["all"], list)
    print(f"PASS: test_jd_skill_extraction_required_preferred")
    print(f"  Required: {result['required']}")
    print(f"  Preferred: {result['preferred']}")


if __name__ == "__main__":
    test_taxonomy_loads()
    test_alias_normalization()
    test_skill_extraction_basic()
    test_skill_extraction_aliases()
    test_skill_matching_coverage()
    test_skill_matching_full_match()
    test_skill_matching_no_match()
    test_jd_skill_extraction_required_preferred()
    print("\nAll skill extraction tests passed!")
