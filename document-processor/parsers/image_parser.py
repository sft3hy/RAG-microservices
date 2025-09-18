# parsers/image_parser.py

import logging
from typing import Optional
from ocr_engine import OCRProcessor

logger = logging.getLogger(__name__)


def extract_text_from_image(
    file_content: bytes, filename: str, ocr_processor: OCRProcessor
) -> Optional[str]:
    """Extract text from image files using OCR."""
    try:
        ocr_result = ocr_processor.process_image(file_content, filename)
        if ocr_result.get("success"):
            return ocr_result.get("text")
        return None
    except Exception as e:
        logger.error(f"Failed to extract text from image {filename}: {e}")
        return None
