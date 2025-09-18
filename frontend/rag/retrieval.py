import logging
import re
import time
from typing import Dict, List, Tuple

from weaviate.classes.query import Filter, MetadataQuery

# Assuming your WeaviateManager class is in a file named `weaviate_manager.py`
from database.weav_operations import WeaviateManager


logger = logging.getLogger(__name__)


class EnhancedMultiQueryRAGRetriever:
    """
    A RAG retriever that leverages Weaviate for efficient chunk retrieval and
    an LLM for advanced query understanding and answer generation.
    """

    def __init__(self, weaviate_manager: WeaviateManager, llm_client: any, config: any):
        """
        Initialize the retriever.

        Args:
            weaviate_manager: An instance of the WeaviateManager class for DB operations.
            llm_client: A client for interacting with a large language model.
            config: A configuration object with parameters like MIN_RELEVANCE_SCORE.
        """
        self.weaviate_manager = weaviate_manager
        self.llm_client = llm_client
        self.config = config
        self.total_tokens_used = 0

    def decompose_complex_query(self, query: str) -> Tuple[List[str], Dict[str, int]]:
        """Break down complex queries into sub-questions using an LLM."""
        prompt = f"""Break down this complex question into 2-4 simpler, independent sub-questions that together would fully answer the original question. Each sub-question should focus on a specific aspect or entity.

Original question: {query}

Sub-questions (one per line):"""
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that breaks down complex questions into simpler sub-questions. Each sub-question should be specific and focused.",
                },
                {"role": "user", "content": prompt},
            ]
            response = self.llm_client._make_chat_completion(
                messages=messages, max_tokens=300, temperature=0.3
            )
            decomposition = self.llm_client._extract_content(response).strip()
            token_usage = self.llm_client.get_token_usage(response)

            sub_questions = [
                re.sub(r"^\d+[\.\)]\s*", "", line).strip()
                for line in decomposition.split("\n")
                if line.strip() and len(line.strip()) > 10
            ]
            logger.info(
                f"Decomposed query into {len(sub_questions)} sub-questions: {sub_questions}"
            )
            return sub_questions, token_usage
        except Exception as e:
            logger.error(f"Failed to decompose query: {e}")
            return [query], {"total_tokens": 0}

    def retrieve_child_chunks(
        self, query: str, user_id: str, top_k: int = 50
    ) -> List[Dict]:
        """Retrieve initial child chunks using Weaviate's vector search."""
        try:
            return self.weaviate_manager.search_similar_chunks(
                query_text=query,
                user_id=user_id,
                limit=top_k,
                chunk_type="child",
                min_score=self.config.MIN_RELEVANCE_SCORE,
            )
        except Exception as e:
            logger.error(f"Error retrieving child chunks via Weaviate: {e}")
            return []

    def get_parent_chunks_from_children(self, child_chunks: List[Dict]) -> List[Dict]:
        """Get unique parent chunks corresponding to the retrieved child chunks."""
        if not child_chunks:
            return []

        parent_chunk_ids = {
            child.get("parent_chunk_id")
            for child in child_chunks
            if child.get("parent_chunk_id")
        }
        if not parent_chunk_ids:
            return []

        parent_chunks = []
        try:
            for parent_id in parent_chunk_ids:
                parent_chunk = self.weaviate_manager.get_chunk_by_id(
                    chunk_id=parent_id, chunk_type="parent"
                )
                if parent_chunk:
                    parent_chunks.append(parent_chunk)
            return parent_chunks
        except Exception as e:
            logger.error(f"Error getting parent chunks from Weaviate: {e}")
            return []

    def diversified_retrieval(
        self, query: str, user_id: str
    ) -> Tuple[List[Dict], Dict[str, int]]:
        """
        Perform a diversified retrieval strategy by decomposing the query and
        generating alternative phrasings to find a comprehensive set of chunks.
        """
        logger.info("Starting diversified retrieval...")
        total_tokens = {"total_tokens": 0}

        try:
            sub_questions, decomp_tokens = self.decompose_complex_query(query)
            total_tokens["total_tokens"] += decomp_tokens.get("total_tokens", 0)

            alternative_query, alt_tokens = self.llm_client.generate_multi_query(query)
            total_tokens["total_tokens"] += alt_tokens.get("total_tokens", 0)

            all_queries = [query, alternative_query] + sub_questions
            all_parent_chunks = []
            seen_chunk_ids = set()

            for i, search_query in enumerate(all_queries):
                logger.info(
                    f"Retrieving for query {i+1}/{len(all_queries)}: '{search_query[:70]}...'"
                )
                child_chunks = self.retrieve_child_chunks(search_query, user_id)
                parent_chunks = self.get_parent_chunks_from_children(child_chunks)

                for parent_chunk in parent_chunks:
                    if parent_chunk["chunk_id"] not in seen_chunk_ids:
                        parent_chunk["retrieved_by_query"] = search_query
                        parent_chunk["query_type"] = (
                            "original"
                            if i == 0
                            else "alternative" if i == 1 else "sub_question"
                        )
                        all_parent_chunks.append(parent_chunk)
                        seen_chunk_ids.add(parent_chunk["chunk_id"])

            logger.info(
                f"Diversified retrieval found {len(all_parent_chunks)} unique parent chunks."
            )
            return all_parent_chunks, total_tokens
        except Exception as e:
            logger.error(f"Error in diversified retrieval: {e}")
            return [], total_tokens

    def final_reranking(
        self, original_query: str, parent_chunks: List[Dict]
    ) -> List[Dict]:
        """
        Reranks a list of candidate chunks against the original query using Weaviate
        to ensure final results are highly relevant.
        """
        if not parent_chunks:
            logger.warning("No parent chunks to rerank.")
            return []

        parent_chunk_ids = [chunk["chunk_id"] for chunk in parent_chunks]
        if not parent_chunk_ids:
            return []

        try:
            collection = self.weaviate_manager.client.collections.get(
                self.weaviate_manager.parent_collection_name
            )
            # Filter search to only our candidate chunks
            where_filter = Filter.by_id().contains_any(parent_chunk_ids)

            response = collection.query.near_text(
                query=original_query,
                limit=len(parent_chunk_ids),
                where=where_filter,
                return_metadata=MetadataQuery(score=True),
            )

            original_chunk_lookup = {
                chunk["chunk_id"]: chunk for chunk in parent_chunks
            }
            reranked_chunks = []
            for obj in response.objects:
                chunk_id_str = str(obj.uuid)
                original_chunk = original_chunk_lookup.get(chunk_id_str)
                if not original_chunk:
                    continue

                reranked_chunk = original_chunk.copy()
                reranked_chunk["relevance_score"] = obj.metadata.score or 0.0
                reranked_chunks.append(reranked_chunk)

            reranked_chunks.sort(key=lambda x: x["relevance_score"], reverse=True)

            final_chunks = [
                chunk
                for chunk in reranked_chunks
                if chunk["relevance_score"] >= self.config.MIN_RELEVANCE_SCORE
            ][: self.config.FINAL_CHUNKS_COUNT]

            logger.info(f"Final reranking produced {len(final_chunks)} chunks.")
            return final_chunks
        except Exception as e:
            logger.error(f"Error in final reranking with Weaviate: {e}")
            return []

    def retrieve_and_generate(
        self, query: str, user_id: str
    ) -> Tuple[str, List[Dict], float, int]:
        """
        Execute the full RAG pipeline: retrieve, rerank, and generate an answer.
        """
        start_time = time.time()
        logger.info(f"Starting RAG pipeline for query: '{query[:100]}...'")

        try:
            parent_chunks, retrieval_tokens = self.diversified_retrieval(query, user_id)

            if not parent_chunks:
                logger.warning("No relevant chunks found after diversified retrieval.")
                return (
                    "I couldn't find relevant information to answer your question. Please try rephrasing or uploading more documents.",
                    [],
                    time.time() - start_time,
                    retrieval_tokens.get("total_tokens", 0),
                )

            final_chunks = self.final_reranking(query, parent_chunks)

            if not final_chunks:
                logger.warning("No chunks passed the final reranking stage.")
                return (
                    "I found some information, but it wasn't relevant enough to construct a reliable answer.",
                    [],
                    time.time() - start_time,
                    retrieval_tokens.get("total_tokens", 0),
                )

            answer, answer_tokens = self.llm_client.generate_answer(query, final_chunks)
            total_tokens = retrieval_tokens.get("total_tokens", 0) + answer_tokens.get(
                "total_tokens", 0
            )
            processing_time = time.time() - start_time
            logger.info(
                f"RAG pipeline completed in {processing_time:.2f}s. Tokens used: {total_tokens}"
            )

            return answer, final_chunks, processing_time, total_tokens
        except Exception as e:
            logger.error(f"Error in retrieve_and_generate pipeline: {e}")
            return (
                f"An error occurred while processing your query: {e}",
                [],
                time.time() - start_time,
                0,
            )
