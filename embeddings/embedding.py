# embeddings/embedding.py
import numpy as np
from typing import List, Union, Any
import logging

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    A synchronous, lazy-loading manager for SentenceTransformer models.
    Designed to be used inside a dedicated worker process.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model: Any = None
        self._dimension: int = None

    @property
    def model(self) -> Any:
        """Lazy loader for the SentenceTransformer model."""
        if self._model is None:
            self._load_model()
        return self._model

    def _load_model(self):
        """
        Loads the sentence transformer model.
        This is called only when the model is first accessed.
        """
        try:
            # Import heavy libraries here, inside the method.
            from sentence_transformers import SentenceTransformer

            logger.info(
                f"Loading embedding model for the first time: {self.model_name}"
            )
            self._model = SentenceTransformer(self.model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(
                f"Embedding model loaded successfully. Dimension: {self._dimension}"
            )
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}", exc_info=True)
            raise

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Synchronously encodes texts into embeddings using the lazy-loaded model.
        """
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.array([])
        try:
            # Accessing self.model will trigger the lazy load if needed.
            return self.model.encode(texts, convert_to_numpy=True)
        except Exception as e:
            logger.error(f"Failed to encode texts: {e}", exc_info=True)
            raise

    def get_embedding_dimension(self) -> int:
        """Gets the embedding dimension, loading the model if necessary."""
        if self._dimension is None:
            # Trigger the model load via the property, which sets the dimension.
            _ = self.model
        return self._dimension
