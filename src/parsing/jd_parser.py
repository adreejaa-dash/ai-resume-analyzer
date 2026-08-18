"""
Job Description parsing and requirement extraction.

Extracts required/preferred skills, experience requirements,
education requirements, and responsibilities.
"""
import re
from typing import Dict, List, Tuple


# Signal words for required vs preferred classification
REQUIRED_SIGNALS = [
    "required", "must have", "must-have", "mandatory", "minimum",
    "essential", "should have", "need", "necessary", "requirement",
]

PREFERRED_SIGNALS = [
    "preferred", "nice to have", "nice-to-have", "bonus", "plus",
    "desirable", "ideal", "advantageous", "optional", "a plus",
]


def parse_job_description(text: str) -> Dict:
    """
    Parse a job description to extract structured information.
    
    Returns:
        dict with:
            - cleaned_text: cleaned JD text
            - experience_required: extracted experience requirement (str or None)
            - education_required: extracted education requirement (str or None)
            - requirement_sections: list of requirement text blocks
            - word_count: word count
    """
    cleaned = text.strip()
    
    return {
        "cleaned_text": cleaned,
        "experience_required": extract_experience_requirement(cleaned),
        "education_required": extract_education_requirement(cleaned),
        "requirement_sections": extract_requirement_blocks(cleaned),
        "word_count": len(cleaned.split()),
    }


def extract_experience_requirement(text: str) -> str:
    """
    Extract experience requirements from JD text.
    Returns a string like "3+ years" or "Unable to determine".
    """
    # Match patterns like "3+ years", "2-5 years", "minimum 3 years"
    patterns = [
        r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:of)?\s*(?:experience|exp)",
        r"(?:minimum|at\s*least|min)\s*(\d+)\s*(?:years?|yrs?)",
        r"(\d+)\s*-\s*(\d+)\s*(?:years?|yrs?)\s*(?:of)?\s*(?:experience|exp)",
        r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:in|with|working)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            groups = match.groups()
            if len(groups) == 2 and groups[1]:
                return f"{groups[0]}-{groups[1]} years"
            return f"{groups[0]}+ years"
    
    return "Unable to determine"


def extract_education_requirement(text: str) -> str:
    """
    Extract education requirements from JD text.
    Returns the requirement string or "Unable to determine".
    """
    patterns = [
        r"((?:bachelor'?s?|master'?s?|ph\.?d\.?|b\.?s\.?|m\.?s\.?|b\.?tech|m\.?tech|mba)\s*(?:degree)?\s*(?:in\s+[\w\s,/&]+)?)",
        r"(degree\s+in\s+[\w\s,/&]+)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return "Unable to determine"


def classify_requirement_type(sentence: str) -> str:
    """
    Classify whether a sentence describes a required or preferred qualification.
    Returns: 'required', 'preferred', or 'unclassified'

    Preferred signals are checked first so that phrases like
    "preferred but not mandatory" are correctly classified as preferred.
    """
    lower = sentence.lower()

    # Check preferred first — it takes priority over negated required signals
    for signal in PREFERRED_SIGNALS:
        if signal in lower:
            return "preferred"

    for signal in REQUIRED_SIGNALS:
        if signal in lower:
            return "required"

    return "unclassified"


def extract_requirement_blocks(text: str) -> List[Dict[str, str]]:
    """
    Split JD into requirement blocks and classify them.
    """
    # Split by bullet points, newlines, or numbered items
    lines = re.split(r"\n|•|●|◦|▪|■|‣|\d+\.\s", text)
    
    blocks = []
    for line in lines:
        line = line.strip()
        if len(line) > 10:  # Skip very short fragments
            req_type = classify_requirement_type(line)
            blocks.append({"text": line, "type": req_type})
    
    return blocks
