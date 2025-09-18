# parsers/pdf_parser.py

import io
import logging
from typing import Optional, List, Dict

import fitz  # PyMuPDF
import pdf2image

from ocr_engine import OCRProcessor

logger = logging.getLogger(__name__)


def extract_text_from_pdf(
    file_content: bytes, filename: str, ocr_processor: OCRProcessor
) -> Optional[str]:
    """Extract text from PDF using text extraction and an OCR fallback."""
    try:
        # First try extracting text directly from PDF
        pdf_document = fitz.open(stream=file_content, filetype="pdf")
        text_content = ""
        for page in pdf_document:
            text_content += page.get_text() + "\n"
        pdf_document.close()

        # If substantial text is found, return it
        if len(text_content.strip()) > 100:
            return text_content

        # Otherwise, fall back to OCR
        logger.info(f"PDF {filename} has minimal text, using OCR fallback.")
        images_text = _extract_images_from_pdf_for_ocr(
            file_content, filename, ocr_processor
        )

        if images_text:
            ocr_text = "\n\n".join([img["text"] for img in images_text])
            return text_content + "\n\n--- OCR Content ---\n\n" + ocr_text

        return text_content if text_content.strip() else None

    except Exception as e:
        logger.error(f"Failed to extract text from PDF {filename}: {e}")
        return None


def _extract_images_from_pdf_for_ocr(
    self, file_content: bytes, filename: str, ocr_processor: OCRProcessor
) -> List[Dict]:
    """Extract images from PDF and process them with OCR."""
    images_text = []
    try:
        pdf_document = fitz.open(stream=file_content, filetype="pdf")
        for page_num in range(min(len(pdf_document), 5)):
            page = pdf_document[page_num]
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(pdf_document, xref)
                    if pix.n - pix.alpha < 4:
                        img_data = pix.tobytes("png")
                        ocr_result = self.ocr_processor.process_image(
                            img_data,
                            f"{filename}_page_{page_num+1}_img_{img_index+1}.png",
                        )
                        if ocr_result["success"] and ocr_result["text"].strip():
                            images_text.append(
                                {
                                    "page": page_num + 1,
                                    "image_index": img_index + 1,
                                    "text": ocr_result["text"],
                                    "ocr_details": ocr_result,
                                }
                            )
                    pix = None
                except Exception as e:
                    logger.warning(
                        f"Failed to process image {img_index} on page {page_num}: {e}"
                    )
        pdf_document.close()

        if not images_text:
            pages = pdf2image.convert_from_bytes(
                file_content, dpi=200, first_page=1, last_page=3
            )
            for page_num, page_image in enumerate(pages):
                img_bytes = io.BytesIO()
                page_image.save(img_bytes, format="PNG")
                img_data = img_bytes.getvalue()
                ocr_result = self.ocr_processor.process_image(
                    img_data, f"{filename}_page_{page_num+1}_full.png"
                )
                if ocr_result["success"] and ocr_result["text"].strip():
                    images_text.append(
                        {
                            "page": page_num + 1,
                            "image_index": "full_page",
                            "text": ocr_result["text"],
                            "ocr_details": ocr_result,
                        }
                    )
    except Exception as e:
        logger.error(f"Failed to extract images from PDF {filename}: {e}")
    return images_text
