# utils.py

import logging
import chardet

logger = logging.getLogger(__name__)


def detect_encoding(file_content: bytes) -> str:
    """Detect file encoding using chardet."""
    try:
        result = chardet.detect(file_content)
        encoding = result.get("encoding", "utf-8")
        confidence = result.get("confidence", 0)

        # Fallback to utf-8 if confidence is too low
        if confidence < 0.7:
            return "utf-8"

        return encoding if encoding else "utf-8"
    except Exception as e:
        logger.warning(f"Encoding detection failed: {e}")
        return "utf-8"
