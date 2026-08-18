"""
Unit tests for scoring engine and parsing.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scoring.scorer import calculate_match_score, load_scoring_config
from src.parsing.resume_parser import detect_sections, extract_contact_info, parse_resume
from src.parsing.jd_parser import (
    parse_job_description, extract_experience_requirement,
    extract_education_requirement, classify_requirement_type,
)
from src.analysis.experience_education import match_experience, match_education


# --- Scoring tests ---

def test_scoring_strong_match():
    config = load_scoring_config()
    result = calculate_match_score(
        required_skill_coverage=0.9,
        semantic_similarity=0.85,
        tfidf_similarity=0.75,
        experience_relevance=0.9,
        education_relevance=1.0,
        preferred_skill_coverage=0.7,
        ml_model_probability=0.88,
        config=config,
    )
    assert result["overall_score"] >= 80, f"Expected strong match, got {result['overall_score']}"
    assert result["category"] == "Strong Match"
    print(f"PASS: test_scoring_strong_match (score={result['overall_score']})")


def test_scoring_weak_match():
    config = load_scoring_config()
    result = calculate_match_score(
        required_skill_coverage=0.1,
        semantic_similarity=0.2,
        tfidf_similarity=0.1,
        experience_relevance=0.2,
        education_relevance=0.2,
        preferred_skill_coverage=0.0,
        ml_model_probability=0.2,
        config=config,
    )
    assert result["overall_score"] < 60, f"Expected weak match, got {result['overall_score']}"
    assert result["category"] == "Weak Match"
    print(f"PASS: test_scoring_weak_match (score={result['overall_score']})")


def test_scoring_breakdown_keys():
    config = load_scoring_config()
    result = calculate_match_score(0.5, 0.5, 0.5, config=config)
    assert "overall_score" in result
    assert "category" in result
    assert "breakdown" in result
    assert "weights_used" in result
    print("PASS: test_scoring_breakdown_keys")


def test_scoring_range():
    config = load_scoring_config()
    for val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        result = calculate_match_score(val, val, val, config=config)
        assert 0 <= result["overall_score"] <= 100
    print("PASS: test_scoring_range")


# --- Parsing tests ---

def test_detect_sections():
    text = """
John Doe
john@email.com

SUMMARY
Experienced data scientist with 3 years.

EDUCATION
B.Tech Computer Science

SKILLS
Python, SQL, Machine Learning

EXPERIENCE
Data Scientist at Company (2021-2024)
"""
    sections = detect_sections(text)
    assert "summary" in sections or "education" in sections or "skills" in sections
    print(f"PASS: test_detect_sections (found: {list(sections.keys())})")


def test_extract_contact_info():
    text = "Email: john@example.com | github.com/johnsmith | +1-555-1234"
    contact = extract_contact_info(text)
    assert contact["email"] == "john@example.com"
    assert contact["github"] is not None
    print(f"PASS: test_extract_contact_info ({contact})")


def test_experience_extraction():
    jd = "We require 3+ years of experience in Python machine learning."
    result = extract_experience_requirement(jd)
    assert result is not None
    assert "3" in result
    print(f"PASS: test_experience_extraction (got: {result})")


def test_experience_extraction_not_found():
    jd = "We are looking for a talented engineer to join our team."
    result = extract_experience_requirement(jd)
    # The jd_parser returns "Unable to determine" (string) when not found
    assert result is None or result == "Unable to determine"
    print("PASS: test_experience_extraction_not_found")


def test_education_extraction():
    jd = "Bachelor's degree in Computer Science required."
    result = extract_education_requirement(jd)
    assert result is not None
    print(f"PASS: test_education_extraction (got: {result})")


def test_requirement_classification():
    assert classify_requirement_type("Python is required for this role") == "required"
    assert classify_requirement_type("Docker is preferred but not mandatory") == "preferred"
    assert classify_requirement_type("Experience with data analysis") == "unclassified"
    print("PASS: test_requirement_classification")


def test_experience_match_unable():
    result = match_experience(
        "We need someone with the right attitude.",
        "I built several projects last year."
    )
    assert result["match_indicator"] == 0.5
    print(f"PASS: test_experience_match_unable ({result['note']})")


def test_education_match():
    result = match_education(
        "Bachelor's degree required.",
        "B.Tech in Computer Science 2021"
    )
    assert result["match_indicator"] >= 0.5
    print(f"PASS: test_education_match (indicator={result['match_indicator']})")


if __name__ == "__main__":
    test_scoring_strong_match()
    test_scoring_weak_match()
    test_scoring_breakdown_keys()
    test_scoring_range()
    test_detect_sections()
    test_extract_contact_info()
    test_experience_extraction()
    test_experience_extraction_not_found()
    test_education_extraction()
    test_requirement_classification()
    test_experience_match_unable()
    test_education_match()
    print("\nAll scoring and parsing tests passed!")
