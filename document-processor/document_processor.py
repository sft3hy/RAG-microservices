import os
import logging
from typing import Dict, Any, Optional

from settings import settings
from ocr_engine import OCRProcessor
from parsers import (
    data_parser,
    document_parser,
    email_parser,
    html_parser,
    image_parser,
    pdf_parser,
    text_parser,
)

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Orchestrates text extraction from various file types."""

    def __init__(self):
        self.ocr_processor = OCRProcessor(
            {
                "use_tesseract": True,
                "use_easyocr": True,
                "use_keras_ocr": False,
                "min_confidence": 0.3,
                "max_image_size": (2000, 2000),
            }
        )
        self.supported_extensions = settings.SUPPORTED_EXTENSIONS

    def extract_text_from_file(
        self, file_content: bytes, filename: str
    ) -> Optional[str]:
        """Routes file content to the appropriate parser based on extension."""
        _, ext = os.path.splitext(filename.lower())

        if ext == ".pdf":
            return pdf_parser.extract_text_from_pdf(
                file_content, filename, self.ocr_processor
            )
        elif ext in {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}:
            return image_parser.extract_text_from_image(
                file_content, filename, self.ocr_processor
            )
        elif ext in {".docx", ".doc"}:
            return document_parser.extract_text_from_docx(file_content, filename)
        elif ext in {".pptx", ".ppt"}:
            return document_parser.extract_text_from_pptx(file_content, filename)
        elif ext in {".xlsx", ".xls"}:
            return data_parser.extract_text_from_excel(file_content, filename)
        elif ext in {".csv", ".tsv"}:
            return data_parser.extract_text_from_csv(file_content, filename)
        elif ext in {".html", ".htm"}:
            return html_parser.extract_text_from_html(file_content, filename)
        elif ext == ".eml":
            return email_parser.extract_text_from_email(file_content, filename)
        else:
            return text_parser.extract_text_from_plain_text(file_content, filename)

    def process_file_content(
        self, file_content: bytes, filename: str, max_size_mb: int = 10
    ) -> Dict[str, Any]:
        """Process file content directly without mock objects."""
        result = {
            "success": False,
            "filename": filename,
            "text_content": None,
            "file_size": len(file_content),
            "file_type": None,
            "file_type_description": None,
            "error_message": None,
        }

        try:
            # Get file extension and description
            _, ext = os.path.splitext(filename.lower())
            result["file_type"] = ext
            result["file_type_description"] = self.get_file_type_description(filename)

            # Validate file size
            if not self.validate_file_size(result["file_size"], max_size_mb):
                result["error_message"] = (
                    f"File size ({result['file_size']/1024/1024:.1f} MB) exceeds maximum allowed size ({max_size_mb} MB)"
                )
                return result

            # Check if file type is supported
            if not self.is_supported_file(filename):
                result["error_message"] = (
                    f"File type '{ext}' is not supported. Supported types: {', '.join(sorted(self.supported_extensions))}"
                )
                return result

            # Extract text content
            text_content = self.extract_text_from_file(file_content, filename)

            if text_content is None:
                result["error_message"] = "Failed to extract text content from file"
                return result

            if len(text_content.strip()) == 0:
                result["error_message"] = (
                    "File appears to be empty or contains no extractable text"
                )
                return result

            result["text_content"] = text_content
            result["success"] = True

            logger.info(
                f"Successfully processed {result['file_type_description']}: {filename} ({result['file_size']} bytes)"
            )

        except Exception as e:
            result["error_message"] = f"Error processing file: {str(e)}"
            logger.error(f"Error processing file {filename}: {e}")

        return result

    def is_supported_file(self, filename: str) -> bool:
        """Check if file type is supported (including images)."""
        _, ext = os.path.splitext(filename.lower())
        return ext in self.supported_extensions

    def get_file_type_description(self, filename: str) -> str:
        """Get human-readable file type description including images."""
        _, ext = os.path.splitext(filename.lower())

        type_descriptions = {
            # Documents
            ".pdf": "PDF Document",
            ".docx": "Word Document",
            ".doc": "Word Document (Legacy)",
            ".pptx": "PowerPoint Presentation",
            ".ppt": "PowerPoint Presentation (Legacy)",
            ".xlsx": "Excel Spreadsheet",
            ".xls": "Excel Spreadsheet (Legacy)",
            ".csv": "CSV Data File",
            ".tsv": "Tab-Separated Values",
            ".html": "HTML Document",
            ".htm": "HTML Document",
            ".eml": "Email Message",
            ".txt": "Text File",
            ".md": "Markdown Document",
            ".json": "JSON Data File",
            ".xml": "XML Document",
            ".yaml": "YAML Configuration",
            ".yml": "YAML Configuration",
            # Images
            ".png": "PNG Image",
            ".jpg": "JPEG Image",
            ".jpeg": "JPEG Image",
            ".tiff": "TIFF Image",
            ".tif": "TIFF Image",
            ".bmp": "Bitmap Image",
            ".gif": "GIF Image",
            ".webp": "WebP Image",
        }

        return type_descriptions.get(
            ext, f"{ext.upper()} File" if ext else "Unknown File Type"
        )

    def validate_file_size(self, file_size: int, max_size_mb: int = 10) -> bool:
        """Validate file size."""
        max_size_bytes = max_size_mb * 1024 * 1024
        return file_size <= max_size_bytes
