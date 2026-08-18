"""
PDF text extraction using PyMuPDF (fitz).

Handles multi-page resumes and provides useful errors
for non-extractable PDFs.
"""
import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_path: str) -> dict:
    """
    Extract text from a PDF file.
    
    Returns:
        dict with keys:
            - text: full extracted text
            - pages: list of per-page text
            - page_count: number of pages
            - word_count: approximate word count
            - has_text: whether meaningful text was extracted
            - error: error message if extraction failed, else None
    """
    result = {
        "text": "",
        "pages": [],
        "page_count": 0,
        "word_count": 0,
        "has_text": False,
        "error": None,
    }
    
    try:
        doc = fitz.open(pdf_path)
        result["page_count"] = len(doc)
        
        all_text = []
        for page in doc:
            page_text = page.get_text("text")
            result["pages"].append(page_text)
            all_text.append(page_text)
        
        doc.close()
        
        full_text = "\n".join(all_text).strip()
        result["text"] = full_text
        result["word_count"] = len(full_text.split())
        result["has_text"] = len(full_text.strip()) > 50
        
        if not result["has_text"]:
            result["error"] = (
                "The PDF contains very little or no extractable text. "
                "It may be a scanned image. OCR is not supported in this version."
            )
    
    except Exception as e:
        result["error"] = f"Failed to read PDF: {str(e)}"
    
    return result


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> dict:
    """
    Extract text from PDF bytes (e.g., from Streamlit file uploader).
    """
    result = {
        "text": "",
        "pages": [],
        "page_count": 0,
        "word_count": 0,
        "has_text": False,
        "error": None,
    }
    
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        result["page_count"] = len(doc)
        
        all_text = []
        for page in doc:
            page_text = page.get_text("text")
            result["pages"].append(page_text)
            all_text.append(page_text)
        
        doc.close()
        
        full_text = "\n".join(all_text).strip()
        result["text"] = full_text
        result["word_count"] = len(full_text.split())
        result["has_text"] = len(full_text.strip()) > 50
        
        if not result["has_text"]:
            result["error"] = (
                "The PDF contains very little or no extractable text. "
                "It may be a scanned image. OCR is not supported in this version."
            )
    
    except Exception as e:
        result["error"] = f"Failed to read PDF: {str(e)}"
    
    return result
