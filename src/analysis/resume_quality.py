"""
Resume Quality & Audit Engine
Performs deep heuristic and NLP-based checks for formatting, content, writing quality, and relevance.
"""
import re
from collections import Counter
from typing import Dict, List, Optional


# ── Dictionaries and Patterns ─────────────────────────────────────────────────

CLICHES = {
    "hardworking", "team player", "motivated", "passionate", "quick learner",
    "detail-oriented", "results-driven", "go-getter", "excellent communication",
    "leadership skills", "problem solver", "highly motivated", "results oriented",
    "detail oriented", "excellent communicator", "strong communicator",
    "self-motivated", "self motivated",
}

CLICHE_GUIDANCE = {
    "hardworking": "Unsupported claim. Add an example of a demanding project that demonstrates work ethic instead.",
    "team player": "Unsupported claim. Mention a specific collaborative project and your role in it.",
    "motivated": "Vague. Demonstrated motivation shows up in projects and contributions, not stated.",
    "passionate": "Vague. Show passion through side projects, contributions, or continuous learning.",
    "quick learner": "Unsupported claim. Name a specific technology you picked up quickly and where you used it.",
    "detail-oriented": "Unsupported claim. Show attention to detail through testing, code reviews, or documentation work.",
    "results-driven": "Unsupported claim. Use actual results (metrics, outcomes) to demonstrate this.",
    "excellent communication": "Unsupported claim. Describe a presentation, client interaction, or documentation you produced.",
    "leadership skills": "Unsupported claim. Mention who you led, on what project, and what the outcome was.",
    "problem solver": "Unsupported claim. Describe a specific hard problem you solved and how.",
    "highly motivated": "Vague — no meaningful signal for recruiters without evidence.",
    "self-motivated": "Vague. Show it through independent projects or contributions.",
}

ACTION_VERBS = {
    "developed", "implemented", "built", "designed", "created", "engineered",
    "optimized", "improved", "reduced", "increased", "automated", "deployed",
    "architected", "led", "managed", "analyzed", "researched", "evaluated",
    "trained", "tested", "debugged", "maintained", "delivered", "established",
    "launched", "scaled", "migrated", "refactored", "integrated", "streamlined",
    "accelerated", "spearheaded", "coordinated", "mentored", "configured",
    "published", "authored", "presented", "initiated", "contributed", "extended",
}

WEAK_VERB_PATTERNS = [
    r"\bresponsible for\b",
    r"\bduties included?\b",
    r"\bduties were\b",
    r"\bassisted with\b",
    r"\binvolved in\b",
    r"\bparticipated in\b",
    r"\bworked on\b",
    r"\bhelped with\b",
    r"\bhelped to\b",
    r"\bwas tasked with\b",
]

QUANTIFICATION_PATTERNS = [
    r"\d+%",
    r"\d+x\b",
    r"\$[\d,]+",
    r"\d+\+?\s*(?:users?|customers?|clients?)",
    r"\d+\s*(?:million|billion|thousand|k\b)",
    r"(?:reduced|improved|increased|decreased)\s+by\s+\d+",
    r"\d+\s*(?:hours?|days?|weeks?|months?)\s*(?:faster|saved)",
]

PERSONAL_INFO_PATTERNS = {
    "date of birth / age": r"\b(?:age|dob|date\s+of\s+birth)\s*[:\-]?\s*\d{1,4}\b",
    "marital status": r"\b(?:marital\s+status|married|single|divorced|widowed)\b",
    "gender": r"\b(?:gender|sex)\s*[:\-]?\s*(?:male|female|other)\b",
    "nationality field": r"\bnationality\s*[:\-]",
    "parents' information": r"\b(?:father(?:'s)?\s+name|mother(?:'s)?\s+name|guardian(?:'s)?\s+name)\b",
}

STANDARD_SECTION_HEADINGS = {
    "summary", "objective", "profile", "about", "education", "skills",
    "experience", "projects", "certifications", "achievements", "internships",
    "publications", "contact", "positions", "awards", "honors", "activities",
    "extracurricular", "languages", "volunteer", "courses",
}

TENSE_PAST = r"\b(was|were|had|built|developed|created|designed|implemented|deployed|reduced|improved|led|managed|delivered|established|launched|scaled|migrated|automated)\b"
TENSE_PRESENT = r"\b(is|are|have|build|develop|create|design|implement|deploy|reduce|improve|lead|manage|deliver|establish|launch|scale|migrate|automate)\b"

DATE_FORMATS = [
    (r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\b", "Abbrev Month+Year"),
    (r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b", "Full Month+Year"),
    (r"\b\d{1,2}/\d{4}\b", "MM/YYYY"),
    (r"\b\d{1,2}-\d{4}\b", "MM-YYYY"),
]

STOP_BIGRAMS = {
    "of the", "in the", "to the", "and the", "for the", "with the", "at the",
    "on the", "in a", "of a", "is a", "as a", "i have", "and to", "to a",
    "in my", "my own", "a the", "the a", "and i", "i am", "to be",
}


def analyze_resume_quality(
    text: str,
    sections: Dict[str, str],
    contact: Dict[str, Optional[str]],
    word_count: int,
    career_level: str,
    target_role: str,
    yoe: str,
) -> Dict:
    """
    Main audit function.
    Returns:
      - quality_score: int (0-100)
      - strengths: list of strings
      - issues: list of dicts {title, severity, location, why, solution}
    """
    text_lower = text.lower()
    issues: List[Dict] = []
    strengths: List[str] = []
    score = 100
    PENALTIES = {"Critical": 15, "High": 10, "Medium": 5, "Minor": 2}

    def add_issue(title: str, severity: str, location: str, why: str, solution: str):
        nonlocal score
        issues.append({"title": title, "severity": severity, "location": location, "why": why, "solution": solution})
        score -= PENALTIES.get(severity, 5)

    # ── 1. Contact Information ────────────────────────────────────────────
    missing_contact = [f for f in ["email", "phone"] if not contact.get(f)]
    if missing_contact:
        add_issue(
            "Missing Essential Contact Info", "Critical", "Header / Contact Info",
            f"Missing: {', '.join(missing_contact)}. Recruiters cannot contact you.",
            "Add your email and phone number clearly at the top of the resume.",
        )
    else:
        strengths.append("Essential contact information (email and phone) is present.")

    if not contact.get("linkedin"):
        add_issue(
            "Missing LinkedIn Profile", "Medium", "Header / Contact Info",
            "A LinkedIn profile is now commonly expected. It provides additional professional context and credibility.",
            "Add your LinkedIn profile URL (e.g., linkedin.com/in/yourname).",
        )
    else:
        strengths.append("LinkedIn profile is included.")

    # ── 2. Unnecessary Personal Information ──────────────────────────────
    for info_type, pattern in PERSONAL_INFO_PATTERNS.items():
        if re.search(pattern, text_lower):
            add_issue(
                f"Unnecessary Personal Info ({info_type})", "Minor", "Header / Contact Info",
                f"Personal details like {info_type} are generally discouraged in corporate resumes "
                "and can introduce bias during screening.",
                "Consider removing this unless specifically required for the role.",
            )

    # ── 3. Section Completeness & Structure ──────────────────────────────
    detected = set(sections.keys()) - {"header"}
    missing = []
    if "experience" not in detected and "projects" not in detected and "internships" not in detected:
        missing.append("Experience or Projects")
    if "skills" not in detected:
        missing.append("Skills")
    if "education" not in detected:
        missing.append("Education")

    if missing:
        add_issue(
            "Missing Core Sections", "High", "Structure",
            f"Resume is missing these fundamental sections: {', '.join(missing)}.",
            "Add clearly labeled section headers: Skills, Education, and at least one of Experience or Projects.",
        )
    else:
        strengths.append("Core resume sections (Skills, Education, Experience/Projects) are present.")

    if career_level == "Experienced Professional" and "summary" not in detected and "objective" not in detected:
        add_issue(
            "Missing Professional Summary", "Medium", "Summary",
            "Experienced professionals are generally expected to have a concise summary framing their expertise.",
            "Add a 2–3 sentence summary explaining who you are and what you offer for this specific role.",
        )

    # Non-standard headings
    non_standard = detected - STANDARD_SECTION_HEADINGS
    if non_standard:
        ns_list = ", ".join(f'"{s}"' for s in sorted(non_standard)[:3])
        add_issue(
            "Non-standard Section Headings", "Minor", "Structure",
            f"Unusual section headings detected: {ns_list}. ATS parsers may fail to correctly classify "
            "sections with non-standard names.",
            "Rename to standard labels: 'Experience', 'Education', 'Projects', 'Skills', 'Certifications'.",
        )

    # ── 4. Resume Length ─────────────────────────────────────────────────
    is_fresher = (career_level == "Fresher / Student")
    if word_count < 100:
        add_issue(
            "Resume Extremely Short", "Critical", "Overall",
            f"Only {word_count} words. Recruiters have almost no information to evaluate your fit.",
            "Expand with project descriptions, technical skills, education, relevant coursework, and responsibilities.",
        )
    elif word_count < 200:
        add_issue(
            "Resume Too Short", "High", "Overall",
            f"At {word_count} words, the resume is too sparse.",
            "Add more detail about your work, projects, or coursework.",
        )
    elif word_count > 1200 and is_fresher:
        add_issue(
            "Resume Too Long for Fresher / Student", "Medium", "Overall",
            f"At {word_count} words, this is unusually long for a student or recent graduate. "
            "Recruiters expect concise resumes from candidates with limited experience.",
            "Trim verbose descriptions and remove low-relevance items. Aim for a focused single-page resume.",
        )
    elif word_count > 2000:
        add_issue(
            "Resume Excessively Long", "High", "Overall",
            f"At {word_count} words, the resume is too dense. Recruiters skim resumes quickly.",
            "Remove older or low-relevance experiences. Keep the most impactful roles with concise bullets.",
        )
    else:
        strengths.append("Appropriate resume length.")

    # ── 5. Bullet Analysis ───────────────────────────────────────────────
    raw_lines: List[str] = []
    for sec in ["experience", "projects", "internships"]:
        if sec in sections:
            raw_lines.extend(sections[sec].split("\n"))

    bullets = [re.sub(r"^[\s•\-–*▸▶►]+", "", b).strip() for b in raw_lines]
    bullets = [b for b in bullets if len(b.split()) >= 4]

    if bullets:
        # a) Weak / duty-based phrasing
        weak_bullets = [b for b in bullets if any(re.search(p, b, re.IGNORECASE) for p in WEAK_VERB_PATTERNS)]
        if weak_bullets:
            ex = weak_bullets[0][:100]
            add_issue(
                "Duty-based Phrasing in Bullets", "High", "Experience / Projects",
                "Phrases like 'responsible for', 'helped with', or 'assisted' describe job duties, not your personal "
                "contribution or impact. Recruiters prefer achievement-oriented language.",
                f'Start bullets with a strong action verb. For example, change: "{ex}…" → '
                f'"Developed / Built / Engineered [what you built] to achieve [outcome]."',
            )

        # b) Overly long bullets
        long_bullets = [b for b in bullets if len(b.split()) > 40]
        if long_bullets:
            add_issue(
                "Overly Long Bullet Points", "Minor", "Experience / Projects",
                f"{len(long_bullets)} bullet(s) exceed 40 words and are difficult to scan quickly.",
                "Break long bullets into two shorter ones, or trim filler. One bullet = one clear accomplishment.",
            )

        # c) Stub / incomplete bullets
        stub_bullets = [b for b in bullets if 4 <= len(b.split()) <= 6]
        if len(stub_bullets) >= 2:
            add_issue(
                "Incomplete / Stub Bullet Points", "Minor", "Experience / Projects",
                f"{len(stub_bullets)} bullet(s) are only 4–6 words — too brief to convey meaningful context.",
                "Expand these to describe what you built, how you built it, and what the outcome was.",
            )

        # d) Vague bullets without scale — only flag if ALL bullets lack numbers AND average length is short
        quantified = sum(1 for b in bullets if any(re.search(p, b, re.IGNORECASE) for p in QUANTIFICATION_PATTERNS))
        avg_words = sum(len(b.split()) for b in bullets) / len(bullets)
        if quantified == 0 and avg_words < 18:
            add_issue(
                "Bullets Lack Scale or Outcome (Consider Adding)", "Minor", "Experience / Projects",
                "Your bullets are brief and don't convey the scope or outcome of your work. "
                "Readers cannot judge your impact.",
                "Where you have them, add measurable results (e.g., 'reduced load time by 30%', "
                "'built for 5,000+ daily users'). Only add numbers you can support — do not invent them.",
            )
        elif quantified >= 2:
            strengths.append("Good use of measurable outcomes in bullets.")

        # e) Repeated action verbs
        verb_starts = []
        for b in bullets:
            first = b.split()[0].lower().rstrip(".,;:") if b.split() else ""
            if first in ACTION_VERBS:
                verb_starts.append(first)
        verb_counts = Counter(verb_starts)
        repeated = [(v, c) for v, c in verb_counts.most_common() if c >= 3]
        if repeated:
            v, c = repeated[0]
            add_issue(
                "Repetitive Action Verbs", "Minor", "Experience / Projects",
                f'The verb "{v.capitalize()}" starts {c} bullets. Repetitive verbs make the resume sound monotonous.',
                f'Vary your verbs. Replace some uses of "{v}" with: implemented, engineered, designed, '
                f'optimized, automated, delivered, or coordinated — whichever is most accurate.',
            )
        elif bullets:
            strengths.append("Good variety of action verbs across bullet points.")

    else:
        # Sections found but no bullets parsed — possible parsing issue
        if any(s in detected for s in ["experience", "projects", "internships"]):
            add_issue(
                "No Parseable Bullet Points Detected", "High", "Experience / Projects",
                "Experience or project sections were found but no bullet points were readable. "
                "This often indicates content in tables, text boxes, columns, or images that cannot be parsed.",
                "Ensure all bullet points are plain text, not embedded in tables, text boxes, or multi-column layouts.",
            )

    # ── 6. Clichés / Generic Soft Skills ─────────────────────────────────
    found_cliches = [c for c in CLICHES if re.search(r"\b" + re.escape(c) + r"\b", text_lower)]
    if found_cliches:
        guidance = CLICHE_GUIDANCE.get(found_cliches[0], "Replace with a concrete example demonstrating this skill.")
        extras = ""
        if len(found_cliches) > 1:
            extras = f" Also found: {', '.join(found_cliches[1:3])}."
        add_issue(
            "Generic Soft Skills / Clichés", "Medium", "Summary / Skills",
            f'Found unsupported generic claim(s): "{found_cliches[0]}".{extras} '
            "These phrases are meaningless without supporting evidence and are widely overused.",
            guidance,
        )
    elif bullets:
        strengths.append("No common cliché phrases detected.")

    # ── 7. Duplicate / Repeated Skills ───────────────────────────────────
    if "skills" in sections:
        skill_tokens = re.findall(r"\b[a-z][a-z0-9\+\#\.]{2,}\b", sections["skills"].lower())
        filler = {"and", "or", "the", "with", "for", "using", "such", "other", "etc", "tools", "tech", "including"}
        skill_counts = Counter(t for t in skill_tokens if t not in filler)
        dupes = [w for w, c in skill_counts.items() if c >= 2]
        if dupes:
            add_issue(
                "Duplicate Skills Listed", "Minor", "Skills",
                f"Skill(s) appear more than once in the Skills section: {', '.join(dupes[:4])}.",
                "Review the Skills section and remove duplicate entries. Each skill should appear once.",
            )

    # ── 8. Date Format Inconsistency ─────────────────────────────────────
    matched_formats = []
    for pattern, label in DATE_FORMATS:
        if re.search(pattern, text, re.IGNORECASE):
            matched_formats.append(label)

    if len(matched_formats) >= 3:
        add_issue(
            "Inconsistent Date Formatting", "Minor", "Experience / Education",
            f"Multiple date formats detected: {', '.join(matched_formats)}. "
            "Mixing formats (e.g., 'Jan 2023', '01/2023') looks unpolished.",
            "Pick one format and apply it consistently throughout. "
            "Recommended: 'Month YYYY – Month YYYY' (e.g., 'Jan 2021 – Mar 2023').",
        )

    # ── 9. Tense Inconsistency ────────────────────────────────────────────
    if "experience" in sections:
        exp_text = sections["experience"]
        past = len(re.findall(TENSE_PAST, exp_text, re.IGNORECASE))
        present = len(re.findall(TENSE_PRESENT, exp_text, re.IGNORECASE))
        # Flag only when both are meaningfully present
        if past >= 3 and present >= 3:
            ratio = min(past, present) / max(past, present)
            if ratio > 0.3:  # both sides non-trivial
                add_issue(
                    "Tense Inconsistency in Experience Section", "Minor", "Experience",
                    f"The Experience section mixes past-tense ({past} instances) and present-tense "
                    f"({present} instances) verbs. This looks inconsistent.",
                    "Use past tense for previous roles ('Developed', 'Managed'). "
                    "Use present tense only for your current position.",
                )

    # ── 10. Repeated Phrases (Bigram Check) ──────────────────────────────
    clean_words = re.sub(r"[^a-z\s]", " ", text_lower).split()
    bigrams = [f"{clean_words[i]} {clean_words[i+1]}" for i in range(len(clean_words) - 1)]
    bigram_counts = Counter(
        bg for bg in bigrams
        if bg not in STOP_BIGRAMS and all(len(w) > 2 for w in bg.split())
    )
    repeated_phrases = [(bg, c) for bg, c in bigram_counts.most_common(5) if c >= 4]
    if repeated_phrases:
        examples = "; ".join(f'"{bg}" (×{c})' for bg, c in repeated_phrases[:2])
        add_issue(
            "Repeated Phrases Across Resume", "Minor", "General",
            f"Phrases appear 4 or more times: {examples}. "
            "Heavy repetition may signal copy-pasting or keyword stuffing.",
            "Vary your phrasing. Avoid repeating the same two-word phrases across multiple sections.",
        )

    # ── Score Clamp ───────────────────────────────────────────────────────
    score = max(0, min(100, score))

    return {
        "quality_score": score,
        "strengths": strengths,
        "issues": issues,
    }
