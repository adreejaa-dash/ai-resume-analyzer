"""
Resume section detection and structured information extraction.

Detects common resume sections and extracts contact information.
"""
import re
from typing import Dict, List, Optional


# Section heading patterns (case-insensitive)
SECTION_PATTERNS = {
    "summary": r"(?:summary|profile|about\s*me|objective|career\s*objective|professional\s*summary)",
    "education": r"(?:education|academic|qualification|degree)",
    "skills": r"(?:skills|technical\s*skills|core\s*competencies|technologies|tech\s*stack|proficiencies)",
    "experience": r"(?:experience|work\s*experience|employment|professional\s*experience|work\s*history)",
    "projects": r"(?:projects|personal\s*projects|academic\s*projects|key\s*projects)",
    "certifications": r"(?:certifications?|licenses?|professional\s*certifications?)",
    "achievements": r"(?:achievements?|awards?|honors?|accomplishments?)",
    "internships": r"(?:internships?|training|apprenticeship)",
    "publications": r"(?:publications?|papers?|research)",
    "positions": r"(?:positions?\s*of\s*responsibility|leadership|extracurricular|activities)",
}


def detect_sections(text: str) -> Dict[str, str]:
    """
    Detect and extract resume sections from text.
    
    Returns dict mapping section names to their content.
    Not all sections are required — only detected ones are returned.
    """
    lines = text.split("\n")
    sections = {}
    current_section = "header"  # Text before first detected section
    current_content = []
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            current_content.append("")
            continue
        
        # Check if this line is a section heading
        detected = _detect_section_heading(stripped)
        if detected:
            # Save previous section
            content = "\n".join(current_content).strip()
            if content:
                sections[current_section] = content
            current_section = detected
            current_content = []
        else:
            current_content.append(stripped)
    
    # Save last section
    content = "\n".join(current_content).strip()
    if content:
        sections[current_section] = content
    
    return sections


def _detect_section_heading(line: str) -> Optional[str]:
    """
    Check if a line is a section heading.
    Returns the normalized section name or None.
    """
    # Clean the line for matching
    clean = re.sub(r"[:\-–—|_•*#]", " ", line).strip().lower()
    clean = re.sub(r"\s+", " ", clean)
    
    # Skip lines that are too long to be headings
    if len(clean.split()) > 6:
        return None
    
    for section_name, pattern in SECTION_PATTERNS.items():
        if re.search(pattern, clean, re.IGNORECASE):
            return section_name
    
    return None


def extract_contact_info(text: str) -> Dict[str, Optional[str]]:
    """
    Extract contact information from resume text.
    """
    contact = {
        "email": None,
        "phone": None,
        "linkedin": None,
        "github": None,
    }
    
    # Email
    email_match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
    if email_match:
        contact["email"] = email_match.group()
    
    # Phone (various formats)
    phone_match = re.search(
        r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", text
    )
    if phone_match:
        contact["phone"] = phone_match.group()
    
    # LinkedIn
    linkedin_match = re.search(r"linkedin\.com/in/[\w-]+", text, re.IGNORECASE)
    if linkedin_match:
        contact["linkedin"] = linkedin_match.group()
    
    # GitHub
    github_match = re.search(r"github\.com/[\w-]+", text, re.IGNORECASE)
    if github_match:
        contact["github"] = github_match.group()
    
    return contact


def parse_resume(text: str) -> Dict:
    """
    Full resume parsing: sections + contact info.
    
    Returns:
        dict with:
            - sections: dict of detected sections
            - contact: dict of contact info
            - section_count: number of detected sections (excluding header)
    """
    sections = detect_sections(text)
    contact = extract_contact_info(text)
    
    # Count actual resume sections (exclude "header")
    section_names = [k for k in sections.keys() if k != "header"]
    
    return {
        "sections": sections,
        "contact": contact,
        "section_count": len(section_names),
        "detected_sections": section_names,
    }
