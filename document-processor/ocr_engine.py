# ocr_engine.py

import io
import logging
from typing import Dict, Any, Optional, Tuple

import cv2
import numpy as np
import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)


class OCRProcessor:
    """
    A synchronous processor for OCR tasks. All its methods run directly.
    Designed to be used inside a dedicated worker process.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.use_tesseract = self.config.get("use_tesseract", True)
        self.use_easyocr = self.config.get("use_easyocr", True)
        self.use_keras_ocr = self.config.get("use_keras_ocr", False)
        self._ocr_engines: Dict[str, Any] = {}

    @property
    def easyocr_engine(self):
        """Lazy loader for the EasyOCR engine."""
        if "easyocr" not in self._ocr_engines:
            if self.use_easyocr:
                try:
                    logger.info("Initializing EasyOCR. This may take a while...")
                    import easyocr

                    self._ocr_engines["easyocr"] = easyocr.Reader(["en"], gpu=False)
                    logger.info("EasyOCR initialized successfully.")
                except Exception as e:
                    logger.error(f"Failed to initialize EasyOCR: {e}")
                    self._ocr_engines["easyocr"] = None
            else:
                self._ocr_engines["easyocr"] = None
        return self._ocr_engines.get("easyocr")

    @property
    def keras_ocr_engine(self):
        """Lazy loader for the Keras-OCR engine."""
        if "keras_ocr" not in self._ocr_engines:
            if self.use_keras_ocr:
                try:
                    logger.info("Initializing Keras-OCR. This may take a while...")
                    import keras_ocr

                    self._ocr_engines["keras_ocr"] = keras_ocr.pipeline.Pipeline()
                    logger.info("Keras-OCR initialized successfully.")
                except Exception as e:
                    logger.error(f"Failed to initialize Keras-OCR: {e}")
                    self._ocr_engines["keras_ocr"] = None
            else:
                self._ocr_engines["keras_ocr"] = None
        return self._ocr_engines.get("keras_ocr")

    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Apply image preprocessing to improve OCR accuracy."""
        preprocessing_info = {}
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            denoised = cv2.fastNlMeansDenoising(gray)
            thresh = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            kernel = np.ones((1, 1), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            corrected = self._correct_skew(cleaned)
            preprocessing_info = {
                "applied_denoising": True,
                "applied_thresholding": True,
                "applied_morphology": True,
                "applied_skew_correction": True,
            }
            return corrected, preprocessing_info
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}")
            return image, {"error": str(e)}

    def _correct_skew(self, image: np.ndarray) -> np.ndarray:
        """Correct skew in the image."""
        try:
            edges = cv2.Canny(image, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(
                edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10
            )
            if lines is not None:
                angles = [
                    np.arctan2(y2 - y1, x2 - x1)
                    for line in lines
                    for x1, y1, x2, y2 in line
                ]
                median_angle = np.median(angles)
                angle_deg = np.degrees(median_angle)
                if abs(angle_deg) > 0.5:
                    (h, w) = image.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
                    corrected = cv2.warpAffine(
                        image,
                        M,
                        (w, h),
                        flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REPLICATE,
                    )
                    return corrected
            return image
        except Exception as e:
            logger.warning(f"Skew correction failed: {e}")
            return image

    def extract_text_tesseract(
        self, image: np.ndarray, config: str = "--psm 6"
    ) -> Dict:
        """Extract text using Tesseract OCR."""
        try:
            data = pytesseract.image_to_data(
                image, config=config, output_type=pytesseract.Output.DICT
            )
            min_confidence = self.config.get("min_confidence", 30)
            confident_text = []
            for i, conf in enumerate(data["conf"]):
                if int(conf) > min_confidence:
                    text = data["text"][i].strip()
                    if text:
                        confident_text.append(
                            {
                                "text": text,
                                "confidence": int(conf),
                                "bbox": (
                                    data["left"][i],
                                    data["top"][i],
                                    data["width"][i],
                                    data["height"][i],
                                ),
                            }
                        )
            full_text = " ".join([item["text"] for item in confident_text])
            return {
                "text": full_text,
                "words": confident_text,
                "engine": "tesseract",
                "success": True,
            }
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return {
                "text": "",
                "words": [],
                "engine": "tesseract",
                "success": False,
                "error": str(e),
            }

    def extract_text_easyocr(self, image: np.ndarray) -> Dict:
        """Extract text using EasyOCR."""
        engine = self.easyocr_engine
        try:
            if not engine:
                return {
                    "text": "",
                    "words": [],
                    "engine": "easyocr",
                    "success": False,
                    "error": "EasyOCR not initialized.",
                }
            results = engine.readtext(image)
            words = []
            full_text_parts = []
            for bbox, text, confidence in results:
                if confidence > self.config.get("min_confidence", 0.3):
                    words.append(
                        {"text": text, "confidence": float(confidence), "bbox": bbox}
                    )
                    full_text_parts.append(text)
            return {
                "text": " ".join(full_text_parts),
                "words": words,
                "engine": "easyocr",
                "success": True,
            }
        except Exception as e:
            logger.error(f"EasyOCR failed: {e}")
            return {
                "text": "",
                "words": [],
                "engine": "easyocr",
                "success": False,
                "error": str(e),
            }

    def extract_text_keras_ocr(self, image: np.ndarray) -> Dict:
        """Extract text using Keras-OCR."""
        engine = self.keras_ocr_engine
        try:
            if not engine:
                return {
                    "text": "",
                    "words": [],
                    "engine": "keras_ocr",
                    "success": False,
                    "error": "Keras-OCR not initialized.",
                }
            if len(image.shape) == 2:  # Convert grayscale to RGB
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            prediction_groups = engine.recognize([image])
            words = [
                {"text": text, "confidence": 1.0, "bbox": bbox.tolist()}
                for text, bbox in prediction_groups[0]
            ]
            full_text = " ".join([word["text"] for word in words])
            return {
                "text": full_text,
                "words": words,
                "engine": "keras_ocr",
                "success": True,
            }
        except Exception as e:
            logger.error(f"Keras-OCR failed: {e}")
            return {
                "text": "",
                "words": [],
                "engine": "keras_ocr",
                "success": False,
                "error": str(e),
            }

    def simple_layout_analysis(self, image: np.ndarray) -> Dict:
        """Simple layout analysis using OpenCV."""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Find contours for text regions
            _, binary = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            text_regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 10 and h > 10:  # Filter small noise
                    text_regions.append({"x": x, "y": y, "width": w, "height": h})

            return {
                "text_regions": text_regions,
                "num_regions": len(text_regions),
                "success": True,
            }
        except Exception as e:
            logger.error(f"Layout analysis failed: {e}")
            return {"success": False, "error": str(e)}

    def ensemble_ocr(self, image: np.ndarray) -> Dict:
        """Run multiple OCR engines and combine results."""
        results = []
        if self.use_tesseract:
            tesseract_result = self.extract_text_tesseract(image)
            if tesseract_result["success"]:
                results.append(tesseract_result)
        if self.use_easyocr:
            easyocr_result = self.extract_text_easyocr(image)
            if easyocr_result["success"]:
                results.append(easyocr_result)
        if self.use_keras_ocr:
            keras_result = self.extract_text_keras_ocr(image)
            if keras_result["success"]:
                results.append(keras_result)
        if not results:
            return {"text": "", "success": False, "error": "All OCR engines failed"}
        best_result = max(results, key=lambda x: len(x["text"]))
        best_result["ensemble_results"] = [r["engine"] for r in results]
        return best_result

    def process_image(self, image_data: bytes, filename: str = "") -> Dict:
        """Main synchronous method for processing a single image."""
        try:
            image = Image.open(io.BytesIO(image_data))
            image_np = np.array(image)
            max_size = self.config.get("max_image_size", (2000, 2000))
            if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                image.thumbnail(max_size, Image.Resampling.LANCZOS)
                image_np = np.array(image)

            preprocessed_image, preprocess_info = self.preprocess_image(image_np)
            layout_info = self.simple_layout_analysis(preprocessed_image)
            ocr_result = self.ensemble_ocr(preprocessed_image)

            return {
                "success": ocr_result.get("success", False),
                "text": ocr_result.get("text", ""),
                "filename": filename,
                "preprocessing": preprocess_info,
                "layout_analysis": layout_info,
                "ocr_details": ocr_result,
                "image_size": image.size,
            }
        except Exception as e:
            logger.error(f"Image processing failed for {filename}: {e}", exc_info=True)
            return {"success": False, "text": "", "filename": filename, "error": str(e)}
