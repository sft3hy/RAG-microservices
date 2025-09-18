import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # Model Configuration
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    LLM_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

    # Chunking Configuration
    CHILD_CHUNK_SIZE = 250
    PARENT_CHUNK_SIZE = 2500
    CONTEXTUAL_HEADER_SIZE = 100
    CHUNK_OVERLAP = 50

    # Retrieval Configuration
    INITIAL_RETRIEVAL_COUNT = 200
    FILTERED_CHILD_CHUNKS = 150
    PARENT_CHUNKS_COUNT = 20
    FINAL_CHUNKS_COUNT = 30
    MIN_RELEVANCE_SCORE = 0.1

    # Database
    DATABASE_PATH = "data/rag_database.db"

    # UI Configuration
    MAX_UPLOAD_SIZE = 200  # MB

    # Locally or deployed
    TEST = os.getenv("TEST_LOCAL")

    # OCR Configuration
    OCR_CONFIG = {
        "use_tesseract": True,
        "use_easyocr": True,
        "use_keras_ocr": False,  # Disable by default (slower)
        "min_confidence": 0.3,
        "max_image_size": (2000, 2000),  # Resize large images
        "preprocess_images": True,
        "pdf_dpi": 200,  # DPI for PDF to image conversion
    }

    SUPPORTED_EXTENSIONS = {
        # Text files
        ".txt",
        ".md",
        ".rst",
        ".log",
        ".cfg",
        ".ini",
        ".conf",
        # Code files
        ".py",
        ".js",
        ".html",
        ".css",
        ".json",
        ".xml",
        ".yaml",
        ".yml",
        ".java",
        ".cpp",
        ".c",
        ".h",
        ".php",
        ".rb",
        ".go",
        ".rs",
        ".sql",
        # Office documents
        ".pdf",
        ".docx",
        ".doc",
        ".pptx",
        ".ppt",
        # Spreadsheets
        ".xlsx",
        ".xls",
        ".csv",
        ".tsv",
        # Email and web
        ".eml",
        ".htm",
        ".xhtml",
        # Other structured formats
        ".rtf",
        ".odt",
        ".ods",
        ".odp",
        ".png",
        # Images
        ".jpg",
        ".jpeg",
        ".tiff",
        ".tif",
        ".bmp",
        ".gif",
        ".webp",
    }


settings = Settings()
