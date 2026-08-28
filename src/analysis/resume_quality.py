"""
Resume Quality & Audit Engine
Performs deep heuristic and NLP-based checks for formatting, content, writing quality, and relevance.
"""
import re
from typing import Dict, List, Optional

# --- Dictionaries and Patterns ---

CLICHES = {
    "hardworking", "team player", "motivated", "passionate", "quick learner",
    "detail-oriented", "results-driven", "go-getter", "excellent communication",
    "leadership skills", "problem solver", "highly motivated", "results oriented",
    "detail oriented"
}

ACTION_VERBS = {
    "developed", "implemented", "built", "designed", "created", "engineered",
    "optimized", "improved", "reduced", "increased", "automated", "deployed",
    "architected", "led", "managed", "collaborated", "analyzed", "researched",
    "evaluated", "trained", "tested", "debugged", "maintained", "delivered",
    "established", "launched", "scaled", "migrated", "refactored", "integrated",
    "streamlined", "accelerated", "spearheaded", "coordinated", "mentored",
}

WEAK_VERBS = {
    "helped", "worked", "assisted", "did", "made", "responsible for", "duties included"
}

QUANTIFICATION_PATTERNS = [
    r"\d+%",                   
    r"\d+x",                   
    r"\$[\d,]+",               
    r"\d+\+?\s*(?:users?|customers?|clients?)",
    r"\d+\s*(?:million|billion|thousand|k\b)",
    r"(?:reduced|improved|increased|decreased)\s+by\s+\d+",
]

PERSONAL_INFO_PATTERNS = {
    "age": r"\b(age|dob|date of birth)\s*[:\-]?\s*\d{1,4}\b",
    "marital": r"\b(marital status|married|single|divorced)\b",
}

STANDARD_SECTION_HEADINGS = {
    "summary", "objective", "profile",
    "education", "skills", "experience",
    "projects", "certifications", "achievements",
    "internships", "publications", "contact"
}

def analyze_resume_quality(
    text: str,
    sections: Dict[str, str],
    contact: Dict[str, Optional[str]],
    word_count: int,
    career_level: str,
    target_role: str,
    yoe: str
) -> Dict:
    """
    Main audit function.
    Returns:
      - quality_score: int
      - strengths: list of strings
      - issues: list of dicts (title, severity, location, why, solution)
    """
    text_lower = text.lower()
    issues = []
    strengths = []
    
    score = 100

    def add_issue(title, severity, location, why, solution, penalty=5):
        nonlocal score
        issues.append({
            "title": title,
            "severity": severity,
            "location": location,
            "why": why,
            "solution": solution
        })
        if severity == "Critical":
            score -= 15
        elif severity == "High":
            score -= 10
        elif severity == "Medium":
            score -= 5
        else:
            score -= 2

    # 1. Contact Information
    contact_fields = ["email", "phone"]
    missing_contact = [f for f in contact_fields if not contact.get(f)]
    if missing_contact:
        add_issue(
            "Missing Essential Contact Info", "Critical", "Header / Contact Info",
            f"Missing: {', '.join(missing_contact)}. Recruiters cannot contact you.",
            "Add your email and phone number clearly at the top."
        )
    else:
        strengths.append("Essential contact information is present.")
        
    if not contact.get("linkedin"):
        add_issue(
            "Missing LinkedIn", "Medium", "Header / Contact Info",
            "A LinkedIn profile provides additional professional context and credibility.",
            "Add a link to your LinkedIn profile if you have one."
        )

    # Personal Info Check
    for info_type, pattern in PERSONAL_INFO_PATTERNS.items():
        if re.search(pattern, text_lower):
            add_issue(
                f"Unnecessary Personal Info ({info_type})", "Minor", "General",
                "Personal details like age, marital status, or DOB are generally discouraged in corporate resumes to prevent bias.",
                "Consider removing this unless specifically required for the role."
            )

    # 2. Section Completeness & Structure
    detected_sections = set(sections.keys())
    missing_sections = []
    if "experience" not in detected_sections and "projects" not in detected_sections:
        missing_sections.append("Experience or Projects")
    if "skills" not in detected_sections:
        missing_sections.append("Skills")
    if "education" not in detected_sections:
        missing_sections.append("Education")
        
    if missing_sections:
        add_issue(
            "Missing Core Sections", "High", "Structure",
            f"Your resume is missing fundamental sections: {', '.join(missing_sections)}.",
            "Ensure you have clearly labeled headers for Skills, Education, and Experience/Projects."
        )
    else:
        strengths.append("Core resume sections (Skills, Education, Experience) are present.")
        
    # Non-standard headings
    non_standard = detected_sections - STANDARD_SECTION_HEADINGS - {"header"}
    if len(non_standard) > 1:
        add_issue(
            "Non-standard Section Headings", "Medium", "Structure",
            f"Found unusual headings: {', '.join(list(non_standard)[:2])}. ATS parsers might struggle to classify these.",
            "Rename sections to standard terms like 'Experience', 'Education', 'Projects', 'Skills'."
        )

    # 3. Resume Length
    if word_count < 150:
        add_issue(
            "Resume Too Short", "High", "Overall Length",
            "Your resume has very little content, making it hard to assess your fit.",
            "Add more detail about your projects, coursework, or responsibilities."
        )
    elif word_count > 1000 and career_level == "Fresher / Student":
        add_issue(
            "Resume Too Dense (Fresher)", "Medium", "Overall Length",
            "For a student/fresher, a resume over ~1000 words is usually too dense and hard to scan.",
            "Condense verbose descriptions and remove older, irrelevant experiences."
        )
    elif word_count > 1500:
        add_issue(
            "Resume Excessively Long", "High", "Overall Length",
            "Recruiters scan resumes in seconds. Highly verbose resumes hide your best achievements.",
            "Edit down to the most impactful, relevant points for the target role."
        )
    else:
        strengths.append("Appropriate overall word count and density.")

    # 4. Content & Writing Issues (Bullets)
    # Check bullets in experience/projects
    all_bullets = []
    if "experience" in sections:
        all_bullets.extend(sections["experience"].split('\n'))
    if "projects" in sections:
        all_bullets.extend(sections["projects"].split('\n'))
        
    all_bullets = [b.strip() for b in all_bullets if len(b.strip()) > 20]
    
    # Check for weak verbs
    weak_verbs_found = False
    for bullet in all_bullets[:10]: # Check sample
        bl = bullet.lower()
        if any(wv in bl for wv in WEAK_VERBS):
            weak_verbs_found = True
            break
            
    if weak_verbs_found:
        add_issue(
            "Weak Phrasing ('responsible for', etc.)", "Medium", "Experience / Projects",
            "Phrases like 'responsible for' or 'helped with' describe duties, not achievements.",
            "Start bullet points with strong action verbs (e.g., Developed, Optimized, Managed) and state what you accomplished."
        )
        
    # Check for quantification, but don't strictly penalize.
    quantified = 0
    for bullet in all_bullets:
        for pattern in QUANTIFICATION_PATTERNS:
            if re.search(pattern, bullet, re.IGNORECASE):
                quantified += 1
                break
                
    if len(all_bullets) > 0 and quantified == 0:
        add_issue(
            "Missing Measurable Impact", "Medium", "Experience / Projects",
            "Your bullets describe what you did, but not the scale or outcome of your work.",
            "Add a measurable result if you have one (e.g., 'reduced latency by 20%', 'served 10k users'). Do not invent numbers."
        )
    elif quantified > 2:
        strengths.append("Good use of quantified achievements to demonstrate impact.")

    # 5. Clichés
    found_cliches = [c for c in CLICHES if c in text_lower]
    if len(found_cliches) >= 2:
        add_issue(
            "Generic Soft Skills (Clichés)", "Medium", "Summary / Skills",
            f"Found generic terms: '{found_cliches[0]}', '{found_cliches[1]}'. These are claims without evidence.",
            "Replace generic soft skills with concrete examples demonstrating them in your experience section."
        )

    # 6. Repetition Detection
    words = text_lower.split()
    # Simple check for repeated action verbs at start of bullets
    verb_starts = []
    for bullet in all_bullets:
        parts = bullet.split()
        if parts:
            first_word = parts[0].lower().strip("*-•")
            if first_word in ACTION_VERBS:
                verb_starts.append(first_word)
    
    from collections import Counter
    verb_counts = Counter(verb_starts)
    repeated_verbs = [v for v, c in verb_counts.items() if c > 3]
    if repeated_verbs:
        add_issue(
            "Repetitive Action Verbs", "Minor", "Experience / Projects",
            f"You started {verb_counts[repeated_verbs[0]]} bullet points with '{repeated_verbs[0]}'. This makes reading repetitive.",
            "Vary your action verbs (e.g., implemented, designed, engineered, created)."
        )
        
    # Final score boundary
    score = max(0, min(100, score))

    return {
        "quality_score": score,
        "strengths": strengths,
        "issues": issues
    }

