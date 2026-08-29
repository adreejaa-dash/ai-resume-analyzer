import io
try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

def extract_text_from_image_bytes(image_bytes: bytes) -> dict:
    if not OCR_AVAILABLE:
        return {"text": "", "word_count": 0, "error": "OCR dependencies (Pillow, pytesseract) not installed."}
    
    try:
        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image)
        word_count = len(text.split())
        return {"text": text.strip(), "word_count": word_count, "error": None}
    except Exception as e:
        return {"text": "", "word_count": 0, "error": str(e)}
