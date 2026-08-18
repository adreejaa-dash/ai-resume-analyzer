"""
Skill matching and gap analysis.

Compares resume skills against JD requirements to produce:
- Matched required skills
- Missing required skills
- Matched preferred skills
- Missing preferred skills
- Coverage ratios
"""
from typing import Dict, List, Set


def match_skills(
    resume_skills: List[str],
    jd_required: List[str],
    jd_preferred: List[str],
) -> Dict:
    """
    Compare resume skills against JD required and preferred skills.
    
    Returns:
        dict with matched/missing skills and coverage ratios
    """
    resume_set = set(s.lower() for s in resume_skills)
    required_set = set(s.lower() for s in jd_required)
    preferred_set = set(s.lower() for s in jd_preferred)
    
    # Required skills
    matched_required = sorted(resume_set & required_set)
    missing_required = sorted(required_set - resume_set)
    
    # Preferred skills
    matched_preferred = sorted(resume_set & preferred_set)
    missing_preferred = sorted(preferred_set - resume_set)
    
    # Coverage ratios
    required_coverage = (
        len(matched_required) / len(required_set) if required_set else 1.0
    )
    preferred_coverage = (
        len(matched_preferred) / len(preferred_set) if preferred_set else 1.0
    )
    total_jd_skills = required_set | preferred_set
    total_matched = resume_set & total_jd_skills
    overall_coverage = (
        len(total_matched) / len(total_jd_skills) if total_jd_skills else 1.0
    )
    
    return {
        "matched_required": matched_required,
        "missing_required": missing_required,
        "matched_preferred": matched_preferred,
        "missing_preferred": missing_preferred,
        "required_coverage": round(required_coverage, 4),
        "preferred_coverage": round(preferred_coverage, 4),
        "overall_coverage": round(overall_coverage, 4),
        "total_resume_skills": len(resume_set),
        "total_jd_required": len(required_set),
        "total_jd_preferred": len(preferred_set),
        "matched_required_count": len(matched_required),
        "missing_required_count": len(missing_required),
    }
