import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)


class ReRanker:
    def __init__(self, embedding_manager):
        self.embedding_manager = embedding_manager

    def compute_relevance_scores(self, query: str, documents: List[str]) -> List[float]:
        """Compute relevance scores between query and documents."""
        if not documents:
            return []

        # Get embeddings
        query_embedding = self.embedding_manager.encode_single(query)
        doc_embeddings = self.embedding_manager.encode(documents)

        # Compute cosine similarity
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1), doc_embeddings
        )[0]

        return similarities.tolist()

    def rerank_documents(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        text_key: str = "text",
        min_score: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """Rerank documents based on relevance to query."""
        if not documents:
            return []

        # Extract text content
        doc_texts = [doc[text_key] for doc in documents]

        # Compute relevance scores
        scores = self.compute_relevance_scores(query, doc_texts)

        # Combine documents with scores
        docs_with_scores = list(zip(documents, scores))

        # Filter by minimum score
        filtered_docs = [
            (doc, score) for doc, score in docs_with_scores if score >= min_score
        ]

        # Sort by score (descending)
        filtered_docs.sort(key=lambda x: x[1], reverse=True)

        # Return documents with scores added
        reranked_docs = []
        for doc, score in filtered_docs:
            doc_copy = doc.copy()
            doc_copy["relevance_score"] = score
            reranked_docs.append(doc_copy)

        logger.info(
            f"Reranked {len(documents)} documents, {len(reranked_docs)} passed minimum score threshold"
        )

        return reranked_docs

    def diversify_results(
        self,
        documents: List[Dict[str, Any]],
        diversity_threshold: float = 0.8,
        text_key: str = "text",
    ) -> List[Dict[str, Any]]:
        """Remove highly similar documents to improve diversity."""
        if len(documents) <= 1:
            return documents

        # Get embeddings for all documents
        doc_texts = [doc[text_key] for doc in documents]
        embeddings = self.embedding_manager.encode(doc_texts)

        # Start with the highest scored document
        diversified_docs = [documents[0]]
        diversified_embeddings = [embeddings[0]]

        # Add documents that are sufficiently different
        for i, doc in enumerate(documents[1:], 1):
            current_embedding = embeddings[i]

            # Check similarity with already selected documents
            similarities = cosine_similarity(
                current_embedding.reshape(1, -1), np.vstack(diversified_embeddings)
            )[0]

            # If not too similar to any selected document, add it
            if max(similarities) < diversity_threshold:
                diversified_docs.append(doc)
                diversified_embeddings.append(current_embedding)

        logger.info(
            f"Diversified results: {len(documents)} -> {len(diversified_docs)} documents"
        )

        return diversified_docs
