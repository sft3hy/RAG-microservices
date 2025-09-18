# parsers/html_parser.py

import logging
from typing import Optional
from ocr_engine import OCRProcessor

logger = logging.getLogger(__name__)


def extract_text_from_html(self, file_content: bytes, filename: str) -> Optional[str]:
    """Extract text from HTML files."""
    try:
        from bs4 import BeautifulSoup

        encoding = self.detect_encoding(file_content)
        html_content = file_content.decode(encoding, errors="ignore")

        soup = BeautifulSoup(html_content, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text and clean it up
        text = soup.get_text()

        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text_content = " ".join(chunk for chunk in chunks if chunk)

        return text_content if text_content.strip() else None

    except Exception as e:
        logger.error(f"Failed to extract text from HTML {filename}: {e}")
        return None
