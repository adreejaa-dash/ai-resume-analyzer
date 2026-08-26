"""
Unit tests for text preprocessing (text_cleaner module).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.text_cleaner import clean_for_tfidf, clean_for_embedding, clean_text


def test_clean_for_tfidf_basic():
    text = "Hello World! This is a test 123."
    cleaned = clean_for_tfidf(text)
    assert "hello" in cleaned or "world" in cleaned
    assert "!" not in cleaned
    print("PASS: test_clean_for_tfidf_basic")


def test_clean_for_tfidf_stopwords():
    text = "The quick brown fox jumps over the lazy dog."
    cleaned = clean_for_tfidf(text)
    assert "the" not in cleaned.split()
    assert "quick" in cleaned
    print("PASS: test_clean_for_tfidf_stopwords")


def test_clean_for_tfidf_urls():
    text = "Check my work at http://github.com/user and www.portfolio.com"
    cleaned = clean_for_tfidf(text)
    assert "http" not in cleaned
    assert "www" not in cleaned
    print("PASS: test_clean_for_tfidf_urls")


def test_clean_for_tfidf_preserves_python():
    text = "Expert in Python and scikit-learn"
    cleaned = clean_for_tfidf(text)
    assert "python" in cleaned
    assert "scikitlearn" in cleaned  # hyphen replaced
    print("PASS: test_clean_for_tfidf_preserves_python")


def test_clean_for_embedding_preserves_structure():
    text = "I built predictive models for customer churn prediction."
    cleaned = clean_for_embedding(text)
    # Should keep the sentence more intact
    assert "predictive" in cleaned
    assert "churn" in cleaned
    assert len(cleaned) > 20
    print("PASS: test_clean_for_embedding_preserves_structure")


def test_clean_text_backward_compat():
    """Ensure clean_text() wrapper is backward-compatible with original."""
    text = "My order hasn't arrived. I need a refund."
    result = clean_text(text)
    assert isinstance(result, str)
    assert len(result) > 0
    print("PASS: test_clean_text_backward_compat")


if __name__ == "__main__":
    test_clean_for_tfidf_basic()
    test_clean_for_tfidf_stopwords()
    test_clean_for_tfidf_urls()
    test_clean_for_tfidf_preserves_python()
    test_clean_for_embedding_preserves_structure()
    test_clean_text_backward_compat()
    print("\nAll preprocessing tests passed!")
