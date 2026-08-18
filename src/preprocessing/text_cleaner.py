"""
Text preprocessing for resume and job description text.

Adapted from the original complaint-classifier's preprocess.py.
Key differences:
- Preserves technical terms (C++, C#, scikit-learn, etc.)
- Handles hyphens intelligently
- Does NOT strip numbers (experience years matter)
- Provides both aggressive (TF-IDF) and gentle (embedding) cleaning modes
"""
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure NLTK data is available
for resource in ['stopwords', 'punkt', 'punkt_tab']:
    try:
        nltk.data.find(f'tokenizers/{resource}' if 'punkt' in resource else f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

STOP_WORDS = set(stopwords.words("english"))

# Technical terms to preserve during preprocessing
PRESERVE_TERMS = {
    "c++", "c#", ".net", "node.js", "vue.js", "react.js", "angular.js",
    "scikit-learn", "scikit learn", "sci-kit", "t-sql", "pl/sql",
    "objective-c", "f#", "asp.net", "vb.net",
}


def clean_for_tfidf(text: str) -> str:
    """
    Clean text for TF-IDF vectorization.
    
    Adapted from original complaint-classifier clean_text():
    - Lowercases
    - Removes URLs
    - Preserves technical terms with special characters
    - Removes most punctuation
    - Removes stopwords
    - Keeps numbers (important for experience years)
    """
    text = str(text).lower().strip()
    
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    
    # Remove email addresses
    text = re.sub(r"\S+@\S+\.\S+", "", text)
    
    # Preserve C++, C# by replacing with tokens
    text = text.replace("c++", "cplusplus")
    text = text.replace("c#", "csharp")
    text = text.replace(".net", "dotnet")
    text = text.replace("node.js", "nodejs")
    text = text.replace("vue.js", "vuejs")
    text = text.replace("react.js", "reactjs")
    
    # Replace hyphens between words with space (but keep scikit-learn as one token)
    text = text.replace("scikit-learn", "scikitlearn")
    text = text.replace("scikit learn", "scikitlearn")
    
    # Remove special characters but keep alphanumeric and spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and very short tokens
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    
    return " ".join(tokens)


def clean_for_embedding(text: str) -> str:
    """
    Lighter cleaning for sentence embedding models (SBERT).
    Preserves more natural language structure since SBERT benefits from context.
    """
    text = str(text).strip()
    
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    
    # Remove email addresses
    text = re.sub(r"\S+@\S+\.\S+", " ", text)
    
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    # Remove very long sequences of special characters
    text = re.sub(r"[^\w\s.,;:!?+#\-/()]{3,}", " ", text)
    
    return text


def clean_text(text: str) -> str:
    """
    Backward-compatible wrapper. Uses TF-IDF cleaning by default.
    This preserves API compatibility with the original complaint-classifier.
    """
    return clean_for_tfidf(text)
