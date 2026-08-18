"""
Hybrid skill extraction from text.

Three-layer approach:
1. Taxonomy matching against ESCO/O*NET vocabulary
2. Pattern-based phrase matching
3. Alias normalization

Extracts and categorizes skills from both resumes and job descriptions.
"""
import re
from typing import Dict, List, Set, Tuple
from src.skills.taxonomy import SkillTaxonomy


class SkillExtractor:
    """
    Extracts skills from text using taxonomy matching and pattern detection.
    """
    
    def __init__(self, taxonomy: SkillTaxonomy = None):
        self.taxonomy = taxonomy or SkillTaxonomy()
        self._build_patterns()
    
    def _build_patterns(self):
        """Build regex patterns from taxonomy for efficient matching."""
        # Get all skill phrases sorted by length (longest first for greedy matching)
        phrases = self.taxonomy.get_all_skill_phrases()
        phrases.sort(key=len, reverse=True)
        
        # Build compiled patterns for multi-word skills
        self.multi_word_patterns = []
        self.single_word_skills = set()
        
        for phrase in phrases:
            if " " in phrase or "-" in phrase or "." in phrase or "+" in phrase or "/" in phrase:
                # Multi-word or special-char skill — use regex
                escaped = re.escape(phrase)
                # Allow flexible whitespace/hyphen matching
                escaped = escaped.replace(r"\ ", r"[\s\-]")
                try:
                    self.multi_word_patterns.append(
                        (re.compile(r"\b" + escaped + r"\b", re.IGNORECASE), phrase)
                    )
                except re.error:
                    # Skip invalid patterns
                    pass
            else:
                self.single_word_skills.add(phrase.lower())
    
    def extract_skills(self, text: str) -> Dict[str, List[str]]:
        """
        Extract skills from text and categorize them.
        
        Returns:
            dict mapping category names to lists of normalized skill names
        """
        found_skills = set()
        text_lower = text.lower()
        
        # Layer 1: Multi-word pattern matching (longest first)
        for pattern, skill_name in self.multi_word_patterns:
            if pattern.search(text_lower):
                normalized = self.taxonomy.normalize_skill(skill_name)
                found_skills.add(normalized)
        
        # Layer 2: Single-word matching
        words = set(re.findall(r"\b[a-zA-Z+#.]+\b", text_lower))
        for word in words:
            normalized = self.taxonomy.normalize_skill(word)
            if normalized in self.single_word_skills or word in self.single_word_skills:
                found_skills.add(normalized)
        
        # Layer 3: Additional pattern-based extraction
        found_skills.update(self._extract_version_skills(text_lower))
        
        # Categorize
        categorized = {}
        for skill in found_skills:
            category = self.taxonomy.get_category(skill)
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(skill)
        
        # Sort within categories
        for cat in categorized:
            categorized[cat].sort()
        
        return categorized
    
    def extract_flat_skills(self, text: str) -> List[str]:
        """Extract skills as a flat sorted list."""
        categorized = self.extract_skills(text)
        all_skills = []
        for skills in categorized.values():
            all_skills.extend(skills)
        return sorted(set(all_skills))
    
    def _extract_version_skills(self, text: str) -> Set[str]:
        """Extract skills with version numbers (e.g., Python 3.x, Java 11)."""
        found = set()
        version_patterns = [
            (r"python\s*[23]\.\d+", "python"),
            (r"java\s*\d+", "java"),
            (r"node\s*\d+", "node.js"),
            (r"angular\s*\d+", "angular"),
            (r"react\s*\d+", "react"),
            (r"vue\s*[23]", "vue"),
            (r"\.net\s*\d+", ".net"),
            (r"spring\s*boot", "spring boot"),
        ]
        for pattern, skill in version_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                found.add(skill)
        return found


def extract_jd_skills(
    text: str,
    extractor: SkillExtractor,
) -> Dict[str, List[str]]:
    """
    Extract skills from a job description with required/preferred classification.
    
    Uses signal words to classify each sentence's skills as required or preferred.
    
    Returns:
        dict with 'required', 'preferred', and 'all' skill lists
    """
    from src.parsing.jd_parser import REQUIRED_SIGNALS, PREFERRED_SIGNALS
    
    all_skills = extractor.extract_flat_skills(text)
    
    # Try to classify skills by the context they appear in
    required_skills = set()
    preferred_skills = set()
    
    # Split into sentences/bullets
    sentences = re.split(r"[.\n•●◦▪■‣]|\d+\.\s", text)
    
    for sentence in sentences:
        sentence_lower = sentence.lower().strip()
        if not sentence_lower:
            continue
        
        # Determine if this sentence is about required or preferred
        is_required = any(signal in sentence_lower for signal in REQUIRED_SIGNALS)
        is_preferred = any(signal in sentence_lower for signal in PREFERRED_SIGNALS)
        
        # Extract skills from this sentence
        sentence_skills = extractor.extract_flat_skills(sentence)
        
        if is_preferred and not is_required:
            preferred_skills.update(sentence_skills)
        elif is_required:
            required_skills.update(sentence_skills)
    
    # Skills not classified go to required by default (conservative)
    unclassified = set(all_skills) - required_skills - preferred_skills
    required_skills.update(unclassified)
    
    return {
        "required": sorted(required_skills),
        "preferred": sorted(preferred_skills),
        "all": sorted(set(all_skills)),
    }
