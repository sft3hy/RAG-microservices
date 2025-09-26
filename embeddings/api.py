# main.py
import logging
from typing import List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from embedding import EmbeddingManager

# -------------------------
# Logging Configuration
# -------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("embedding_api")


# -------------------------
# Handle startup and shutdown
# -------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the manager
    global embedding_manager
    embedding_manager = EmbeddingManager(model_name="all-MiniLM-L6-v2")

    # Eagerly load the model at startup by encoding a dummy text
    logger.info("Warming up the embedding model...")
    embedding_manager.encode(["Pre-warming the model at startup."])
    logger.info("Embedding model loaded and ready.")

    yield

    # Clean up the model and release the resources
    logger.info("Shutting down API.")


# -------------------------
# FastAPI App
# -------------------------
app = FastAPI(
    title="Embedding API",
    description="API for generating text embeddings using SentenceTransformers.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow CORS (adjust origins for your environment)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for production, restrict to your domain(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Initialize Embedding Manager
# -------------------------
embedding_manager: EmbeddingManager | None = None


# -------------------------
# Schemas
# -------------------------
class EmbeddingRequest(BaseModel):
    texts: List[str]


class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    dimension: int


# -------------------------
# Health Check Endpoints (Required by Weaviate)
# -------------------------
@app.get("/.well-known/ready")
def ready():
    """
    Health check endpoint that Weaviate uses to verify the service is ready.
    """
    if embedding_manager is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return {"ready": True}


@app.get("/.well-known/live")
def live():
    """
    Liveness check endpoint.
    """
    return {"live": True}


@app.get("/health")
def health():
    """
    Additional health check endpoint.
    """
    return {"status": "healthy"}


# -------------------------
# Embedding Endpoints
# -------------------------
@app.post("/embed", response_model=EmbeddingResponse, tags=["Embeddings"])
def embed(request: EmbeddingRequest):
    """
    Generate embeddings for a list of input texts.
    """
    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided.")

    try:
        vectors = embedding_manager.encode(request.texts)
        embeddings = vectors.tolist()
        dim = embedding_manager.get_embedding_dimension()

        return EmbeddingResponse(embeddings=embeddings, dimension=dim)

    except Exception as e:
        logger.error(f"Embedding failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Embedding generation failed")


@app.post("/v1/embeddings")
async def hf_embeddings(request: Request):
    """
    HuggingFace-compatible embeddings endpoint for Weaviate.
    """
    # Log the incoming request for debugging
    logger.info(f"Received embedding request from {request.client.host}")

    try:
        body = await request.json()
        logger.info(f"Request body: {body}")
    except Exception as e:
        logger.error(f"Failed to parse JSON body: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # Weaviate expects `input`, OpenAI expects `input`, HuggingFace TGI expects `inputs`
    texts = body.get("input") or body.get("inputs")
    logger.info(f"Extracted texts: {texts}")

    if not texts or not isinstance(texts, list):
        logger.error(f"Invalid texts format: {texts}")
        raise HTTPException(
            status_code=400,
            detail="Request must contain `input` or `inputs` as a list of strings.",
        )

    try:
        logger.info(f"Generating embeddings for {len(texts)} texts")
        vectors = embedding_manager.encode(texts).tolist()
        response = {"data": [{"embedding": vec} for vec in vectors]}
        logger.info(f"Successfully generated {len(vectors)} embeddings")
        return response
    except Exception as e:
        logger.error(f"Embedding failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Embedding generation failed")


@app.get("/meta")
def get_meta():
    """
    Metadata endpoint that Weaviate calls to get model information.
    """
    try:
        # The model is now eagerly loaded at startup, so no check is needed here.
        dimension = embedding_manager.get_embedding_dimension()

        return {
            "model": "all-MiniLM-L6-v2",
            "dimension": dimension,
            "max_seq_length": 256,  # typical for this model
            "tokenizer": "sentence-transformers",
        }
    except Exception as e:
        logger.error(f"Meta endpoint failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get model metadata")


@app.post("/embed/vectors")
async def embed_vectors(request: Request):
    """
    Vectors endpoint that Weaviate calls for embeddings.
    This is the text2vec-transformers format.
    """
    logger.info(f"Received vectors request from {request.client.host}")

    try:
        body = await request.json()
        logger.info(f"Vectors request body: {body}")

        # text2vec-transformers sends different formats
        # Check for all possible text fields
        text_input = None
        if "text" in body:
            text_input = body["text"]
        elif "texts" in body:
            text_input = body["texts"]
        elif "input" in body:
            text_input = body["input"]
        elif isinstance(body, dict) and len(body) == 1:
            # Sometimes it might be a single key-value pair
            key, value = next(iter(body.items()))
            text_input = value
        else:
            # If body is a list, it might be the texts directly
            if isinstance(body, list):
                text_input = body

        logger.info(f"Extracted text input: {text_input}")

        # Handle different input formats
        if isinstance(text_input, str):
            texts = [text_input]
        elif isinstance(text_input, list):
            texts = text_input
        else:
            logger.error(f"Unexpected input format: {text_input}")
            raise HTTPException(
                status_code=400, detail="No valid text provided for vectorization"
            )

        logger.info(f"Processing {len(texts)} texts for vectorization")

        if not texts:
            raise HTTPException(
                status_code=400, detail="No text provided for vectorization"
            )

        vectors = embedding_manager.encode(texts).tolist()
        logger.info(
            f"Generated {len(vectors)} vectors, each with {len(vectors[0])} dimensions"
        )

        # Try different response formats that text2vec-transformers might expect
        response_formats = [
            {"vectors": vectors},  # Simple format
            {"data": [{"embedding": vec} for vec in vectors]},  # HF format
            vectors,  # Direct list format
        ]

        # Use the simple format first
        response = {"vectors": vectors}
        logger.info(f"Returning response with {len(vectors)} vectors")
        return response

    except Exception as e:
        logger.error(f"Vectors endpoint failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Vectorization failed: {str(e)}")


@app.get("/test-embedding")
def test_embedding():
    """
    Test endpoint to verify embedding generation works.
    """
    try:
        test_texts = ["Hello world", "This is a test"]
        vectors = embedding_manager.encode(test_texts).tolist()
        return {
            "success": True,
            "embeddings_count": len(vectors),
            "dimension": len(vectors[0]) if vectors else 0,
            "sample_embedding": vectors[0][:5] if vectors else None,  # First 5 dims
        }
    except Exception as e:
        logger.error(f"Test embedding failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


# @app.get("/embed/.well-known/ready")
# def test_embedding():
#     return {}


@app.get("/")
def hello():
    return "Hello, World!"
