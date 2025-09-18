import weaviate
import weaviate.classes.config as wc
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import Filter, MetadataQuery
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import json
import os
import logging
from datetime import datetime, timezone
from config.settings import settings


class WeaviateManager:
    """
    Handles all Weaviate operations for document chunks and embeddings.
    Supports hierarchical parent-child chunk relationships.
    """

    def __init__(
        self,
    ):
        """
        Initialize Weaviate client and setup collections.

        Args:
            weaviate_url: Weaviate instance URL (defaults to env var WEAVIATE_URL)
            weaviate_api_key: Weaviate API key (defaults to env var WEAVIATE_API_KEY)
            openai_api_key: OpenAI API key for vectorization (defaults to env var OPENAI_API_KEY)
        """
        self.weaviate_host = os.getenv("WEAVIATE_HOST")
        self.weaviate_port = int(os.getenv("WEAVIATE_PORT", "8080"))
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize client
        self.client = self._connect()

        # Collection names
        self.parent_collection_name = "DocumentParentChunk"
        self.child_collection_name = "DocumentChildChunk"

        # Setup collections
        self._setup_collections()

    def _connect(self) -> weaviate.WeaviateClient:
        """Establish connection to Weaviate."""
        try:

            client = weaviate.connect_to_local(
                host=self.weaviate_host,
                port=self.weaviate_port,
            )

            self.logger.info(
                f"Connected to Weaviate at {self.weaviate_host}:{self.weaviate_port}"
            )
            return client

        except Exception as e:
            self.logger.error(f"Failed to connect to Weaviate: {e}")
            raise

    def _setup_collections(self):
        """Setup Weaviate collections for parent and child chunks."""
        try:
            # Parent chunks collection
            if not self.client.collections.exists(self.parent_collection_name):
                self.client.collections.create(
                    name=self.parent_collection_name,
                    properties=[
                        Property(name="document_id", data_type=DataType.INT),
                        Property(name="user_id", data_type=DataType.TEXT),
                        Property(name="chunk_text", data_type=DataType.TEXT),
                        Property(name="contextual_header", data_type=DataType.TEXT),
                        Property(name="chunk_index", data_type=DataType.INT),
                        Property(name="document_name", data_type=DataType.TEXT),
                        Property(name="file_type", data_type=DataType.TEXT),
                        Property(name="created_at", data_type=DataType.DATE),
                    ],
                    # Configure the vectorizer for the Hugging Face model
                    vectorizer_config=Configure.Vectorizer.text2vec_transformers(
                        # Point to your custom embedding API endpoint
                        inference_url="http://embedding_api:8001/embed",
                        # Optional: specify model-related options
                        pooling_strategy="masked_mean",
                    ),
                    vector_index_config=Configure.VectorIndex.hnsw(
                        distance_metric=wc.VectorDistances.COSINE,
                        ef_construction=200,
                        max_connections=64,
                    ),
                )
                self.logger.info(f"Created collection: {self.parent_collection_name}")

            # Child chunks collection
            if not self.client.collections.exists(self.child_collection_name):
                self.client.collections.create(
                    name=self.child_collection_name,
                    properties=[
                        Property(name="document_id", data_type=DataType.INT),
                        Property(name="user_id", data_type=DataType.TEXT),
                        Property(
                            name="parent_chunk_id", data_type=DataType.TEXT
                        ),  # UUID of parent chunk
                        Property(name="chunk_text", data_type=DataType.TEXT),
                        Property(name="contextual_header", data_type=DataType.TEXT),
                        Property(name="chunk_index", data_type=DataType.INT),
                        Property(name="document_name", data_type=DataType.TEXT),
                        Property(name="file_type", data_type=DataType.TEXT),
                        Property(name="created_at", data_type=DataType.DATE),
                    ],
                    # Configure the vectorizer for the Hugging Face model
                    vectorizer_config=Configure.Vectorizer.text2vec_transformers(
                        # Point to your custom embedding API endpoint
                        inference_url="http://embedding_api:8001/embed",
                        # Optional: specify model-related options
                        pooling_strategy="masked_mean",
                    ),
                    vector_index_config=Configure.VectorIndex.hnsw(
                        distance_metric=wc.VectorDistances.COSINE,
                        ef_construction=200,
                        max_connections=64,
                    ),
                )
                self.logger.info(f"Created collection: {self.child_collection_name}")

        except Exception as e:
            self.logger.error(f"Failed to setup collections: {e}")
            raise

    def insert_parent_chunk(
        self,
        document_id: int,
        user_id: str,
        chunk_text: str,
        contextual_header: str,
        chunk_index: int,
        document_name: str,
        file_type: str,
    ) -> str:
        """
        Insert a parent chunk into Weaviate.

        Args:
            document_id: ID of the document this chunk belongs to
            user_id: ID of the user who owns the document
            chunk_text: The actual text content of the chunk
            contextual_header: Contextual header for the chunk
            chunk_index: Index of this chunk within the document
            document_name: Name of the source document
            file_type: Type of the source file

        Returns:
            UUID of the created chunk
        """
        try:
            collection = self.client.collections.get(self.parent_collection_name)

            chunk_data = {
                "document_id": document_id,
                "user_id": user_id,
                "chunk_text": chunk_text,
                "contextual_header": contextual_header,
                "chunk_index": chunk_index,
                "document_name": document_name,
                "file_type": file_type,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

            uuid = collection.data.insert(chunk_data)
            self.logger.info(f"Inserted parent chunk {uuid} for document {document_id}")
            return str(uuid)

        except Exception as e:
            self.logger.error(f"Failed to insert parent chunk: {e}")
            raise

    def insert_child_chunk(
        self,
        document_id: int,
        user_id: str,
        parent_chunk_id: str,
        chunk_text: str,
        contextual_header: str,
        chunk_index: int,
        document_name: str,
        file_type: str,
    ) -> str:
        """
        Insert a child chunk into Weaviate.

        Args:
            document_id: ID of the document this chunk belongs to
            user_id: ID of the user who owns the document
            parent_chunk_id: UUID of the parent chunk
            chunk_text: The actual text content of the chunk
            contextual_header: Contextual header for the chunk
            chunk_index: Index of this chunk within the document
            document_name: Name of the source document
            file_type: Type of the source file

        Returns:
            UUID of the created chunk
        """
        try:
            collection = self.client.collections.get(self.child_collection_name)

            chunk_data = {
                "document_id": document_id,
                "user_id": user_id,
                "parent_chunk_id": parent_chunk_id,
                "chunk_text": chunk_text,
                "contextual_header": contextual_header,
                "chunk_index": chunk_index,
                "document_name": document_name,
                "file_type": file_type,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

            uuid = collection.data.insert(chunk_data)
            self.logger.info(f"Inserted child chunk {uuid} for document {document_id}")
            return str(uuid)

        except Exception as e:
            self.logger.error(f"Failed to insert child chunk: {e}")
            raise

    def batch_insert_chunks(
        self, chunks: List[Dict[str, Any]], chunk_type: str = "child"
    ) -> List[str]:
        """
        Batch insert multiple chunks for better performance.

        Args:
            chunks: List of chunk dictionaries
            chunk_type: Either "parent" or "child"

        Returns:
            List of UUIDs for inserted chunks
        """
        try:
            collection_name = (
                self.parent_collection_name
                if chunk_type == "parent"
                else self.child_collection_name
            )
            collection = self.client.collections.get(collection_name)

            # Prepare batch data
            batch_data = []
            for chunk in chunks:
                chunk_data = {
                    "document_id": chunk["document_id"],
                    "user_id": chunk["user_id"],
                    "chunk_text": chunk["chunk_text"],
                    "contextual_header": chunk["contextual_header"],
                    "chunk_index": chunk["chunk_index"],
                    "document_name": chunk["document_name"],
                    "file_type": chunk["file_type"],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }

                if chunk_type == "child" and "parent_chunk_id" in chunk:
                    chunk_data["parent_chunk_id"] = chunk["parent_chunk_id"]

                batch_data.append(chunk_data)

            # Insert batch
            result = collection.data.insert_many(batch_data)

            uuids = [str(uuid) for uuid in result.uuids]
            self.logger.info(f"Batch inserted {len(uuids)} {chunk_type} chunks")

            if result.errors:
                self.logger.warning(f"Batch insert had {len(result.errors)} errors")
                for error in result.errors:
                    self.logger.error(f"Batch insert error: {error}")

            return uuids

        except Exception as e:
            self.logger.error(f"Failed to batch insert chunks: {e}")
            raise

    def search_similar_chunks(
        self,
        query_text: str,
        user_id: str,
        limit: int = 10,
        chunk_type: str = "child",
        document_ids: Optional[List[int]] = None,
        min_score: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using vector similarity.

        Args:
            query_text: Text to search for
            user_id: User ID to filter results
            limit: Maximum number of results to return
            chunk_type: Either "parent" or "child"
            document_ids: Optional list of document IDs to filter by
            min_score: Minimum similarity score threshold

        Returns:
            List of similar chunks with scores
        """
        try:
            collection_name = (
                self.parent_collection_name
                if chunk_type == "parent"
                else self.child_collection_name
            )
            collection = self.client.collections.get(collection_name)

            # Build filter
            where_filter = Filter.by_property("user_id").equal(user_id)

            if document_ids:
                doc_filter = Filter.by_property("document_id").contains_any(
                    document_ids
                )
                where_filter = where_filter & doc_filter

            # Perform vector search
            response = collection.query.near_text(
                query=query_text,
                limit=limit,
                where=where_filter,
                return_metadata=MetadataQuery(score=True, explain_score=True),
            )

            # Process results
            results = []
            for obj in response.objects:
                score = obj.metadata.score if obj.metadata.score else 0.0

                # Filter by minimum score
                if score < min_score:
                    continue

                result = {
                    "chunk_id": str(obj.uuid),
                    "chunk_text": obj.properties.get("chunk_text", ""),
                    "contextual_header": obj.properties.get("contextual_header", ""),
                    "document_id": obj.properties.get("document_id"),
                    "document_name": obj.properties.get("document_name", ""),
                    "chunk_index": obj.properties.get("chunk_index"),
                    "score": score,
                    "chunk_type": chunk_type,
                }

                if chunk_type == "child":
                    result["parent_chunk_id"] = obj.properties.get("parent_chunk_id")

                results.append(result)

            self.logger.info(
                f"Found {len(results)} similar {chunk_type} chunks for query"
            )
            return results

        except Exception as e:
            self.logger.error(f"Failed to search similar chunks: {e}")
            raise

    def get_chunk_by_id(
        self, chunk_id: str, chunk_type: str = "child"
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific chunk by its ID.

        Args:
            chunk_id: UUID of the chunk
            chunk_type: Either "parent" or "child"

        Returns:
            Chunk data or None if not found
        """
        try:
            collection_name = (
                self.parent_collection_name
                if chunk_type == "parent"
                else self.child_collection_name
            )
            collection = self.client.collections.get(collection_name)

            obj = collection.query.fetch_object_by_id(chunk_id)

            if obj:
                result = {
                    "chunk_id": str(obj.uuid),
                    "chunk_text": obj.properties.get("chunk_text", ""),
                    "contextual_header": obj.properties.get("contextual_header", ""),
                    "document_id": obj.properties.get("document_id"),
                    "user_id": obj.properties.get("user_id", ""),
                    "document_name": obj.properties.get("document_name", ""),
                    "chunk_index": obj.properties.get("chunk_index"),
                    "file_type": obj.properties.get("file_type", ""),
                    "created_at": obj.properties.get("created_at"),
                    "chunk_type": chunk_type,
                }

                if chunk_type == "child":
                    result["parent_chunk_id"] = obj.properties.get("parent_chunk_id")

                return result

            return None

        except Exception as e:
            self.logger.error(f"Failed to get chunk by ID: {e}")
            raise

    def get_child_chunks_by_parent(self, parent_chunk_id: str) -> List[Dict[str, Any]]:
        """
        Get all child chunks for a given parent chunk.

        Args:
            parent_chunk_id: UUID of the parent chunk

        Returns:
            List of child chunks
        """
        try:
            collection = self.client.collections.get(self.child_collection_name)

            response = collection.query.fetch_objects(
                where=Filter.by_property("parent_chunk_id").equal(parent_chunk_id),
                limit=1000,  # High limit to get all children
            )

            results = []
            for obj in response.objects:
                result = {
                    "chunk_id": str(obj.uuid),
                    "chunk_text": obj.properties.get("chunk_text", ""),
                    "contextual_header": obj.properties.get("contextual_header", ""),
                    "document_id": obj.properties.get("document_id"),
                    "parent_chunk_id": obj.properties.get("parent_chunk_id"),
                    "chunk_index": obj.properties.get("chunk_index"),
                }
                results.append(result)

            # Sort by chunk_index
            results.sort(key=lambda x: x["chunk_index"] or 0)

            return results

        except Exception as e:
            self.logger.error(f"Failed to get child chunks: {e}")
            raise

    def get_parent_chunk_by_child(
        self, child_chunk_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get the parent chunk for a given child chunk.

        Args:
            child_chunk_id: UUID of the child chunk

        Returns:
            Parent chunk data or None if not found
        """
        try:
            # First get the child chunk to find parent ID
            child_chunk = self.get_chunk_by_id(child_chunk_id, "child")

            if not child_chunk or not child_chunk.get("parent_chunk_id"):
                return None

            # Get the parent chunk
            return self.get_chunk_by_id(child_chunk["parent_chunk_id"], "parent")

        except Exception as e:
            self.logger.error(f"Failed to get parent chunk: {e}")
            raise

    def delete_chunks_by_document(
        self, document_id: int, user_id: str
    ) -> Dict[str, int]:
        """
        Delete all chunks (parent and child) for a specific document.

        Args:
            document_id: ID of the document
            user_id: User ID (for security)

        Returns:
            Dictionary with count of deleted parent and child chunks
        """
        try:
            deleted_counts = {"parent_chunks": 0, "child_chunks": 0}

            # Delete parent chunks
            parent_collection = self.client.collections.get(self.parent_collection_name)
            parent_filter = Filter.by_property("document_id").equal(
                document_id
            ) & Filter.by_property("user_id").equal(user_id)

            parent_result = parent_collection.data.delete_many(where=parent_filter)
            deleted_counts["parent_chunks"] = parent_result.successful

            # Delete child chunks
            child_collection = self.client.collections.get(self.child_collection_name)
            child_filter = Filter.by_property("document_id").equal(
                document_id
            ) & Filter.by_property("user_id").equal(user_id)

            child_result = child_collection.data.delete_many(where=child_filter)
            deleted_counts["child_chunks"] = child_result.successful

            self.logger.info(
                f"Deleted chunks for document {document_id}: {deleted_counts}"
            )
            return deleted_counts

        except Exception as e:
            self.logger.error(f"Failed to delete chunks for document: {e}")
            raise

    def delete_chunks_by_user(self, user_id: str) -> Dict[str, int]:
        """
        Delete all chunks for a specific user.

        Args:
            user_id: User ID

        Returns:
            Dictionary with count of deleted parent and child chunks
        """
        try:
            deleted_counts = {"parent_chunks": 0, "child_chunks": 0}

            # Delete parent chunks
            parent_collection = self.client.collections.get(self.parent_collection_name)
            parent_result = parent_collection.data.delete_many(
                where=Filter.by_property("user_id").equal(user_id)
            ).successful

            # Delete child chunks
            child_collection = self.client.collections.get(self.child_collection_name)
            child_result = child_collection.data.delete_many(
                where=Filter.by_property("user_id").equal(user_id)
            )
            deleted_counts["child_chunks"] = child_result.successful

            self.logger.info(f"Deleted all chunks for user {user_id}: {deleted_counts}")
            return deleted_counts

        except Exception as e:
            self.logger.error(f"Failed to delete chunks for user: {e}")
            raise

    def get_document_chunks(
        self, document_id: int, user_id: str, chunk_type: str = "child"
    ) -> List[Dict[str, Any]]:
        """
        Get all chunks for a specific document.

        Args:
            document_id: ID of the document
            user_id: User ID (for security)
            chunk_type: Either "parent", "child", or "both"

        Returns:
            List of chunks
        """
        try:
            results = []

            filter_condition = Filter.by_property("document_id").equal(
                document_id
            ) & Filter.by_property("user_id").equal(user_id)

            if chunk_type in ["parent", "both"]:
                parent_collection = self.client.collections.get(
                    self.parent_collection_name
                )
                parent_response = parent_collection.query.fetch_objects(
                    where=filter_condition, limit=10000
                )

                for obj in parent_response.objects:
                    result = {
                        "chunk_id": str(obj.uuid),
                        "chunk_text": obj.properties.get("chunk_text", ""),
                        "contextual_header": obj.properties.get(
                            "contextual_header", ""
                        ),
                        "chunk_index": obj.properties.get("chunk_index"),
                        "chunk_type": "parent",
                    }
                    results.append(result)

            if chunk_type in ["child", "both"]:
                child_collection = self.client.collections.get(
                    self.child_collection_name
                )
                child_response = child_collection.query.fetch_objects(
                    where=filter_condition, limit=10000
                )

                for obj in child_response.objects:
                    result = {
                        "chunk_id": str(obj.uuid),
                        "chunk_text": obj.properties.get("chunk_text", ""),
                        "contextual_header": obj.properties.get(
                            "contextual_header", ""
                        ),
                        "parent_chunk_id": obj.properties.get("parent_chunk_id"),
                        "chunk_index": obj.properties.get("chunk_index"),
                        "chunk_type": "child",
                    }
                    results.append(result)

            # Sort by chunk_index
            results.sort(key=lambda x: x["chunk_index"] or 0)

            return results

        except Exception as e:
            self.logger.error(f"Failed to get document chunks: {e}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the Weaviate database.

        Returns:
            Dictionary with database statistics
        """
        try:
            stats = {}

            # Parent chunks count
            parent_collection = self.client.collections.get(self.parent_collection_name)
            parent_response = parent_collection.aggregate.over_all(total_count=True)
            stats["total_parent_chunks"] = parent_response.total_count

            # Child chunks count
            child_collection = self.client.collections.get(self.child_collection_name)
            child_response = child_collection.aggregate.over_all(total_count=True)
            stats["total_child_chunks"] = child_response.total_count

            stats["total_chunks"] = (
                stats["total_parent_chunks"] + stats["total_child_chunks"]
            )

            return stats

        except Exception as e:
            self.logger.error(f"Failed to get stats: {e}")
            raise

    def nuke_all_chunks(self) -> Dict[str, int]:
        """
        Delete all chunks from both collections. Use with caution!

        Returns:
            Dictionary with count of deleted chunks
        """
        try:
            deleted_counts = {"parent_chunks": 0, "child_chunks": 0}

            # Delete all parent chunks
            parent_collection = self.client.collections.get(self.parent_collection_name)
            parent_result = parent_collection.data.delete_many(
                where=Filter.by_property("user_id").like("*")
            )
            deleted_counts["parent_chunks"] = parent_result.successful

            # Delete all child chunks
            child_collection = self.client.collections.get(self.child_collection_name)
            child_result = child_collection.data.delete_many(
                where=Filter.by_property("user_id").like("*")
            )
            deleted_counts["child_chunks"] = child_result.successful

            self.logger.warning(f"NUKED all chunks from Weaviate: {deleted_counts}")
            return deleted_counts

        except Exception as e:
            self.logger.error(f"Failed to nuke all chunks: {e}")
            raise

    def close(self):
        """Close the Weaviate client connection."""
        try:
            self.client.close()
            self.logger.info("Closed Weaviate client connection")
        except Exception as e:
            self.logger.error(f"Error closing connection: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Example usage and testing functions
def test_weaviate_operations():
    """Test function to demonstrate Weaviate operations."""

    # Initialize Weaviate manager
    weaviate_manager = WeaviateManager()

    try:
        # Test inserting parent chunk
        parent_id = weaviate_manager.insert_parent_chunk(
            document_id=1,
            user_id="test@example.com",
            chunk_text="This is a parent chunk containing overview information.",
            contextual_header="Document Overview",
            chunk_index=0,
            document_name="test_document.pdf",
            file_type="pdf",
        )
        print(f"Inserted parent chunk: {parent_id}")

        # Test inserting child chunks
        child_chunks = [
            {
                "document_id": 1,
                "user_id": "test@example.com",
                "parent_chunk_id": parent_id,
                "chunk_text": "This is the first child chunk with detailed information.",
                "contextual_header": "Document Overview - Detail 1",
                "chunk_index": 1,
                "document_name": "test_document.pdf",
                "file_type": "pdf",
            },
            {
                "document_id": 1,
                "user_id": "test@example.com",
                "parent_chunk_id": parent_id,
                "chunk_text": "This is the second child chunk with more specific details.",
                "contextual_header": "Document Overview - Detail 2",
                "chunk_index": 2,
                "document_name": "test_document.pdf",
                "file_type": "pdf",
            },
        ]

        child_ids = weaviate_manager.batch_insert_chunks(child_chunks, "child")
        print(f"Inserted child chunks: {child_ids}")

        # Test searching
        results = weaviate_manager.search_similar_chunks(
            query_text="detailed information", user_id="test@example.com", limit=5
        )
        print(f"Search results: {len(results)} chunks found")
        for result in results:
            print(
                f"  - Score: {result['score']:.3f}, Text: {result['chunk_text'][:50]}..."
            )

        # Test getting stats
        stats = weaviate_manager.get_stats()
        print(f"Database stats: {stats}")

    except Exception as e:
        print(f"Test failed: {e}")
    finally:
        weaviate_manager.close()


if __name__ == "__main__":
    # Set environment variables for local Weaviate instance
    test_weaviate_operations()
