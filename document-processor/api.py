# api.py

import logging
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse

# Import the main processor class from your refactored code
from document_processor import DocumentProcessor

# --- Basic Configuration ---

# Configure logging to see outputs in the console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# --- FastAPI Application Initialization ---

app = FastAPI(
    title="Document Processing Microservice",
    description="An API to extract text content from various document and image types.",
    version="1.0.0",
)

# Create a single, reusable instance of the DocumentProcessor.
# This is efficient as it initializes the OCR engines only once when the app starts.
try:
    processor = DocumentProcessor()
    logger.info("DocumentProcessor initialized successfully.")
except Exception as e:
    logger.critical(
        f"Fatal error during DocumentProcessor initialization: {e}", exc_info=True
    )
    # If the processor can't start, the API is not functional.
    processor = None


# --- API Health Check Endpoint ---


@app.get("/health", tags=["Health Check"])
async def health_check():
    """
    Simple health check endpoint to confirm the service is running
    and the processor is initialized.
    """
    if processor is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "error",
                "message": "Document processor failed to initialize.",
            },
        )
    return {"status": "ok", "message": "Service is up and running."}


# --- Main Document Processing Endpoint ---


@app.post("/process-document/", tags=["Document Processing"])
async def process_document(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Processes an uploaded file (document, image, etc.) to extract its text content.

    **Supported File Types**: PDF, DOCX, PPTX, XLSX, CSV, EML, HTML,
    PNG, JPG, TIFF, and more.

    **Returns**: A JSON object containing the extracted text and metadata on success.
    On failure, it returns a descriptive HTTP error.
    """
    if processor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The document processing engine is not available. Please check server logs.",
        )

    logger.info(f"Received file for processing: {file.filename} ({file.content_type})")

    try:
        # Read the file content into memory.
        file_content = await file.read()
        filename = file.filename

        # Use the DocumentProcessor to handle all the complex logic
        result = processor.process_file_content(
            file_content=file_content,
            filename=filename,
            max_size_mb=200,  # You can configure the max file size here
        )

        # Check the result from the processor and return the appropriate HTTP response
        if not result.get("success"):
            error_message = result.get(
                "error_message", "An unknown error occurred during processing."
            )
            logger.warning(f"Processing failed for {filename}: {error_message}")
            # Use 422 for validation errors like wrong file type or size
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_message,
            )

        logger.info(f"Successfully processed file: {filename}")

        # Return a clean, successful JSON response
        return {
            "filename": result.get("filename"),
            "file_type": result.get("file_type_description"),
            "file_size_bytes": result.get("file_size"),
            "extracted_text": result.get("text_content"),
        }

    except HTTPException as http_exc:
        # Re-raise exceptions that are already formatted for HTTP response
        raise http_exc
    except Exception as e:
        # Catch any other unexpected errors and return a generic 500 error
        logger.error(
            f"An unexpected error occurred while processing {file.filename}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal server error occurred. Please check the logs.",
        )


@app.get("/")
def hello():
    return "Hello, World!"


# -------------------------
# Health Check Endpoints (Required by Weaviate)
# -------------------------
@app.get("/.well-known/ready")
def ready():
    """
    Health check endpoint that Weaviate uses to verify the service is ready.
    """
    if processor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The document processing engine is not available. Please check server logs.",
        )
    return {"ready": True}
