"""
Resume quality and ATS-oriented checks.

Secondary feature — implemented after core ML pipeline.
These are ATS-ORIENTED CHECKS, not a universal ATS score.
We do NOT claim to reproduce any commercial ATS system.
"""
import re
from typing import Dict, List, Optional


ACTION_VERBS = {
    "developed", "implemented", "built", "designed", "created", "engineered",
    "optimized", "improved", "reduced", "increased", "automated", "deployed",
    "architected", "led", "managed", "collaborated", "analyzed", "researched",
    "evaluated", "trained", "tested", "debugged", "maintained", "delivered",
    "established", "launched", "scaled", "migrated", "refactored", "integrated",
    "streamlined", "accelerated", "spearheaded", "coordinated", "mentored",
}

QUANTIFICATION_PATTERNS = [
    r"\d+%",                   # percentages
    r"\d+x",                   # multipliers
    r"\$[\d,]+",               # dollar amounts
    r"\d+\+?\s*(?:users?|customers?|clients?)",
    r"\d+\s*(?:million|billion|thousand|k\b)",
    r"(?:reduced|improved|increased|decreased)\s+by\s+\d+",
    r"\d+\s*(?:hours?|days?|weeks?|months?)\s*(?:faster|saved)",
]

STANDARD_SECTION_HEADINGS = {
    "summary", "objective", "profile",
    "education", "skills", "experience",
    "projects", "certifications", "achievements",
    "internships", "publications",
}


def analyze_resume_quality(
    text: str,
    sections: Dict[str, str],
    contact: Dict[str, Optional[str]],
    word_count: int,
    page_count: int,
) -> Dict:
    """
    Run ATS-oriented resume quality checks.

    Returns a dict with quality signals — not a claimed universal ATS score.
    """
    text_lower = text.lower()

    # 1. Section completeness
    detected = set(sections.keys()) - {"header"}
    expected = {"education", "skills", "experience", "projects"}
    found_expected = detected & expected
    section_score = len(found_expected) / len(expected)

    # 2. Contact information completeness
    contact_fields = ["email", "phone", "linkedin", "github"]
    contact_found = [f for f in contact_fields if contact.get(f)]
    contact_score = len(contact_found) / len(contact_fields)

    # 3. Quantified achievements
    quantified = []
    for pattern in QUANTIFICATION_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        quantified.extend(matches)
    has_quantification = len(quantified) > 0

    # 4. Action verbs
    words = set(text_lower.split())
    found_verbs = [v for v in ACTION_VERBS if v in words]
    has_action_verbs = len(found_verbs) >= 3

    # 5. Standard section headings (ATS-friendliness)
    heading_overlap = detected & STANDARD_SECTION_HEADINGS
    has_standard_headings = len(heading_overlap) >= 3

    # 6. Length signals
    length_ok = 150 <= word_count <= 1200  # reasonable range

    # 7. GitHub/LinkedIn presence (for tech roles)
    has_github = contact.get("github") is not None
    has_linkedin = contact.get("linkedin") is not None

    return {
        "section_completeness": {
            "detected_sections": sorted(detected),
            "expected_sections": sorted(expected),
            "found_expected": sorted(found_expected),
            "score": round(section_score, 2),
        },
        "contact_info": {
            "found": contact_found,
            "missing": [f for f in contact_fields if not contact.get(f)],
            "score": round(contact_score, 2),
        },
        "quantification": {
            "has_quantified_achievements": has_quantification,
            "examples": quantified[:5],
        },
        "action_verbs": {
            "has_action_verbs": has_action_verbs,
            "found": found_verbs[:8],
        },
        "standard_headings": {
            "has_standard_headings": has_standard_headings,
            "matched": sorted(heading_overlap),
        },
        "length": {
            "word_count": word_count,
            "page_count": page_count,
            "length_ok": length_ok,
        },
        "tech_profile": {
            "has_github": has_github,
            "has_linkedin": has_linkedin,
        },
        # Overall summary — NOT a universal ATS score
        "overall_signals": {
            "sections_ok": section_score >= 0.75,
            "contact_ok": contact_score >= 0.5,
            "quantified": has_quantification,
            "action_verbs": has_action_verbs,
            "standard_headings": has_standard_headings,
            "length_ok": length_ok,
        },
    }
