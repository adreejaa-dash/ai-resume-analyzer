"""
Experience and education extraction and matching.

Conservative approach: returns "Unable to determine" rather than fabricating.
"""
import re
from typing import Optional, Dict


DEGREE_LEVELS = {
    "phd": 5, "ph.d": 5, "doctorate": 5, "doctoral": 5,
    "master": 4, "masters": 4, "msc": 4, "ms": 4, "mtech": 4, "mba": 4, "m.tech": 4, "m.s": 4,
    "bachelor": 3, "bachelors": 3, "bsc": 3, "bs": 3, "btech": 3, "b.tech": 3,
    "be": 3, "b.e": 3, "b.s": 3, "undergraduate": 3,
    "associate": 2, "diploma": 1, "certificate": 1,
}

EXPERIENCE_PATTERNS = [
    r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp|work)",
    r"(?:minimum|at\s+least|min\.?)\s+(\d+)\s*(?:years?|yrs?)",
    r"(\d+)\s*[-–]\s*(\d+)\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp)",
    r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:in|with|working\s+(?:in|with))",
]


def extract_experience_years(text: str) -> Optional[str]:
    """
    Extract experience requirement from text.
    Returns formatted string or None if unable to determine.
    """
    for pattern in EXPERIENCE_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            groups = [g for g in m.groups() if g is not None]
            if len(groups) == 2:
                return f"{groups[0]}–{groups[1]} years"
            return f"{groups[0]}+ years"
    return None


def extract_degree_level(text: str) -> Optional[str]:
    """
    Extract highest degree level from text.
    Returns canonical degree name or None.
    """
    text_lower = text.lower()
    best_level = 0
    best_label = None

    for keyword, level in DEGREE_LEVELS.items():
        if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
            if level > best_level:
                best_level = level
                best_label = keyword

    return best_label


def match_experience(jd_text: str, resume_text: str) -> Dict:
    """
    Conservative experience matching.

    Returns indicator value in [0, 1] and explanatory strings.
    Returns 0.5 (neutral) when unable to determine.
    """
    jd_exp = extract_experience_years(jd_text)
    resume_exp = extract_experience_years(resume_text)

    if jd_exp is None:
        return {
            "jd_requirement": "Unable to determine",
            "resume_experience": resume_exp or "Unable to determine",
            "match_indicator": 0.5,
            "note": "JD experience requirement could not be extracted",
        }

    if resume_exp is None:
        return {
            "jd_requirement": jd_exp,
            "resume_experience": "Unable to determine",
            "match_indicator": 0.5,
            "note": "Resume experience could not be extracted",
        }

    # Extract numeric values for comparison
    jd_num = _extract_first_number(jd_exp)
    res_num = _extract_first_number(resume_exp)

    if jd_num is None or res_num is None:
        return {
            "jd_requirement": jd_exp,
            "resume_experience": resume_exp,
            "match_indicator": 0.5,
            "note": "Could not compare numerically",
        }

    if res_num >= jd_num:
        indicator = 1.0
        note = "Meets or exceeds requirement"
    elif res_num >= jd_num * 0.7:
        indicator = 0.7
        note = "Close to requirement"
    else:
        indicator = 0.3
        note = "Below stated requirement"

    return {
        "jd_requirement": jd_exp,
        "resume_experience": resume_exp,
        "match_indicator": indicator,
        "note": note,
    }


def match_education(jd_text: str, resume_text: str) -> Dict:
    """
    Conservative education matching.
    Returns indicator value in [0, 1] and explanatory strings.
    """
    jd_degree = extract_degree_level(jd_text)
    resume_degree = extract_degree_level(resume_text)

    jd_level = DEGREE_LEVELS.get(jd_degree, 0) if jd_degree else 0
    res_level = DEGREE_LEVELS.get(resume_degree, 0) if resume_degree else 0

    if jd_level == 0:
        return {
            "jd_requirement": "Unable to determine",
            "resume_degree": resume_degree or "Unable to determine",
            "match_indicator": 0.5,
            "note": "Education requirement not specified",
        }

    if res_level == 0:
        return {
            "jd_requirement": jd_degree,
            "resume_degree": "Unable to determine",
            "match_indicator": 0.5,
            "note": "Resume education could not be extracted",
        }

    if res_level >= jd_level:
        indicator = 1.0
        note = "Meets or exceeds requirement"
    elif res_level == jd_level - 1:
        indicator = 0.6
        note = "One level below requirement"
    else:
        indicator = 0.2
        note = "Below stated requirement"

    return {
        "jd_requirement": jd_degree,
        "resume_degree": resume_degree,
        "match_indicator": indicator,
        "note": note,
    }


def _extract_first_number(text: str) -> Optional[float]:
    """Extract first number from a string like '3+ years'."""
    m = re.search(r"(\d+(?:\.\d+)?)", text)
    return float(m.group(1)) if m else None
