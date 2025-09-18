# parsers/text_parser.py

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def extract_text_from_plain_text(
    self, file_content: bytes, filename: str
) -> Optional[str]:
    """Extract text from plain text files with better encoding handling."""
    try:
        # Try to detect encoding first
        encoding = self.detect_encoding(file_content)

        # Decode with detected encoding
        text_content = file_content.decode(encoding, errors="ignore")

        # Clean up the text using ftfy for better unicode handling
        try:
            import ftfy

            text_content = ftfy.fix_text(text_content)
        except ImportError:
            # ftfy not available, continue without it
            pass

        return text_content if text_content.strip() else None

    except Exception as e:
        logger.error(f"Failed to extract text from plain text file {filename}: {e}")
        return None
