import pytest
from unittest.mock import MagicMock, patch

try:
    import fitz
except ImportError:
    fitz = None

if fitz is None:
    pytest.skip("PyMuPDF (fitz) is not installed in this environment", allow_module_level=True)

from src.parsing.pdf_parser import extract_text_from_pdf, extract_text_from_pdf_bytes
from src.parsing.resume_parser import extract_contact_info

def create_mock_doc(text, links=None):
    if links is None:
        links = []
    
    mock_page = MagicMock()
    mock_page.get_text.return_value = text
    mock_page.get_links.return_value = links
    
    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 1
    mock_doc.__iter__.return_value = iter([mock_page])
    return mock_doc


@patch("src.parsing.pdf_parser.fitz.open")
def test_visible_linkedin(mock_fitz_open):
    """TEST 1: Resume with visible LinkedIn URL as plain text."""
    mock_fitz_open.return_value = create_mock_doc("Experience at Corp. linkedin.com/in/my-profile")
    
    res = extract_text_from_pdf("dummy.pdf")
    contact = extract_contact_info(res["text"])
    assert contact.get("linkedin") == "linkedin.com/in/my-profile"


@patch("src.parsing.pdf_parser.fitz.open")
def test_embedded_linkedin(mock_fitz_open):
    """TEST 2: Clickable LinkedIn hyperlink where the URL is embedded."""
    mock_fitz_open.return_value = create_mock_doc(
        text="Experience at Corp. Connect with me on LinkedIn",
        links=[{"kind": fitz.LINK_URI, "uri": "https://linkedin.com/in/embedded-profile"}]
    )
    
    res = extract_text_from_pdf("dummy.pdf")
    contact = extract_contact_info(res["text"])
    assert "linkedin.com/in/embedded-profile" in contact.get("linkedin", "")


@patch("src.parsing.pdf_parser.fitz.open")
def test_embedded_github(mock_fitz_open):
    """TEST 3: Clickable GitHub hyperlink."""
    mock_fitz_open.return_value = create_mock_doc(
        text="Check my projects on GitHub.",
        links=[{"kind": fitz.LINK_URI, "uri": "https://github.com/my-github"}]
    )
    
    res = extract_text_from_pdf("dummy.pdf")
    contact = extract_contact_info(res["text"])
    assert "github.com/my-github" in contact.get("github", "")


@patch("src.parsing.pdf_parser.fitz.open")
def test_neither_linkedin_nor_github(mock_fitz_open):
    """TEST 4: Resume with neither LinkedIn nor GitHub."""
    mock_fitz_open.return_value = create_mock_doc(
        text="Software Engineer",
        links=[{"kind": fitz.LINK_URI, "uri": "https://google.com"}]
    )
    
    res = extract_text_from_pdf("dummy.pdf")
    contact = extract_contact_info(res["text"])
    assert contact.get("linkedin") is None
    assert contact.get("github") is None


@patch("src.parsing.pdf_parser.fitz.open")
def test_word_linkedin_no_url(mock_fitz_open):
    """TEST 5: Resume containing the word 'LinkedIn' but no actual URL/hyperlink."""
    mock_fitz_open.return_value = create_mock_doc(
        text="Find me on LinkedIn and GitHub if you can.",
        links=[]
    )
    
    res = extract_text_from_pdf("dummy.pdf")
    contact = extract_contact_info(res["text"])
    assert contact.get("linkedin") is None
    assert contact.get("github") is None
