# parsers/email_parser.py

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def extract_text_from_email(self, file_content: bytes, filename: str) -> Optional[str]:
    """Extract text from email files."""
    try:
        import email
        from email import policy
        from bs4 import BeautifulSoup

        # Parse email content
        msg = email.message_from_bytes(file_content, policy=policy.default)

        text_content = f"Email: {filename}\n\n"

        # Extract headers
        if msg["subject"]:
            text_content += f"Subject: {msg['subject']}\n"
        if msg["from"]:
            text_content += f"From: {msg['from']}\n"
        if msg["to"]:
            text_content += f"To: {msg['to']}\n"
        if msg["date"]:
            text_content += f"Date: {msg['date']}\n"

        text_content += "\n--- Email Body ---\n\n"

        # Extract body content
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_content()
                    if body:
                        text_content += body + "\n"
                elif part.get_content_type() == "text/html":
                    # Fallback to HTML if no plain text
                    html_body = part.get_content()
                    if html_body and "text/plain" not in [
                        p.get_content_type() for p in msg.walk()
                    ]:
                        soup = BeautifulSoup(html_body, "html.parser")
                        text_content += soup.get_text() + "\n"
        else:
            # Single part message
            body = msg.get_content()
            if body:
                text_content += body + "\n"

        return text_content if text_content.strip() else None

    except Exception as e:
        logger.error(f"Failed to extract text from email {filename}: {e}")
        return None
