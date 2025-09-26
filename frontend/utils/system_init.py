# utils/system_init.py

import os
import logging
import traceback
import streamlit as st

# Import project modules
from config.settings import settings
from rag.chunking import DocumentChunker
from rag.retrieval import WeaviateRetriever
from database.weav_operations import WeaviateManager

if settings.TEST == "True":
    from database.models_local import DatabaseManager
    from database.operations_local import (
        DocumentOperations,
        QueryOperations,
        UserOperations,
    )
    from rag.llm_client import GroqLLMClient as LLMClient
else:
    from database.models import DatabaseManager
    from database.operations import (
        DocumentOperations,
        QueryOperations,
        UserOperations,
    )
    from rag.llm_client import SanctuaryLLMClient as LLMClient

# External microservices clients
from utils.microservice_clients import DocumentProcessorClient

logger = logging.getLogger(__name__)


@st.cache_resource
def initialize_system():
    """Initialize all system components and cache them."""
    try:
        if not settings.GROQ_API_KEY and settings.TEST == "True":
            st.error("GROQ_API_KEY environment variable is not set!")
            st.stop()

        os.makedirs("data", exist_ok=True)

        # Database setup
        if settings.TEST == "True":
            db_manager = DatabaseManager(settings.DATABASE_PATH)
        else:
            db_manager = DatabaseManager()

        # Core database operations
        doc_ops = DocumentOperations(db_manager)
        query_ops = QueryOperations(db_manager)
        user_ops = UserOperations(db_manager)

        # Microservice-based components
        document_processor = DocumentProcessorClient(base_url="http://localhost:8002")

        # Local components
        document_chunker = DocumentChunker(
            child_chunk_size=settings.CHILD_CHUNK_SIZE,
            parent_chunk_size=settings.PARENT_CHUNK_SIZE,
            contextual_header_size=settings.CONTEXTUAL_HEADER_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        llm_client = LLMClient()

        # Retrieval system
        retriever = WeaviateRetriever(
            WeaviateManager(), config=settings, llm_client=llm_client
        )

        logger.info("System initialized successfully")
        return {
            "db_manager": db_manager,
            "doc_ops": doc_ops,
            "query_ops": query_ops,
            "user_ops": user_ops,
            "document_chunker": document_chunker,
            "llm_client": llm_client,
            "document_processor": document_processor,
            "retriever": retriever,
        }

    except Exception as e:
        error_msg = f"Failed to initialize system: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        st.error(f"{error_msg}\n\n**Traceback:**\n```\n{traceback.format_exc()}\n```")
        st.stop()
