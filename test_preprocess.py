from preprocess import clean_text

def test_clean_text_basic():
    text = "Hello World! This is a test 123."
    cleaned = clean_text(text)
    assert "hello" in cleaned
    assert "world" in cleaned
    assert "test" in cleaned
    assert "123" not in cleaned
    assert "!" not in cleaned

def test_clean_text_stopwords():
    text = "The quick brown fox jumps over the lazy dog."
    cleaned = clean_text(text)
    assert "the" not in cleaned
    assert "quick" in cleaned
    assert "fox" in cleaned

def test_clean_text_urls():
    text = "Check out my website at http://example.com and www.google.com"
    cleaned = clean_text(text)
    assert "http" not in cleaned
    assert "www" not in cleaned
    assert "website" in cleaned

if __name__ == "__main__":
    test_clean_text_basic()
    test_clean_text_stopwords()
    test_clean_text_urls()
    print("All preprocess tests passed!")
