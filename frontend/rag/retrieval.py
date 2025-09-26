import logging
import time
from typing import Dict, List, Tuple, Optional
import traceback
import re

from weaviate.classes.query import Filter, MetadataQuery

# Assuming your WeaviateManager class is in a file named `weaviate_manager.py`
from database.weav_operations import WeaviateManager


logger = logging.getLogger(__name__)


class WeaviateRetriever:
    """
    A RAG retriever that leverages Weaviate
    for efficient question answering and chunk retrieval.
    """

    def __init__(self, weaviate_manager: WeaviateManager, llm_client, config: any):
        """
        Initialize the retriever.

        Args:
            weaviate_manager: An instance of the WeaviateManager class for DB operations.
            llm_client: LLM client for query decomposition and answer generation.
            config: A configuration object with parameters like MIN_RELEVANCE_SCORE.
        """
        self.weaviate_manager = weaviate_manager
        self.llm_client = llm_client
        self.config = config

    def decompose_complex_query(self, query: str) -> Tuple[List[str], Dict[str, int]]:
        """Break down complex queries into sub-questions."""
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

            # Parse sub-questions (simple line-based parsing)
            sub_questions = []
            for line in decomposition.split("\n"):
                line = line.strip()
                # Remove numbering and clean up
                line = re.sub(r"^\d+[\.\)]\s*", "", line)
                if line and len(line) > 10:  # Filter out very short lines
                    sub_questions.append(line)

            logger.info(
                f"Decomposed query into {len(sub_questions)} sub-questions: {sub_questions}"
            )
            return sub_questions, token_usage

        except Exception as e:
            logger.error(f"Failed to decompose query: {e}")
            return [query], {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

    def retrieve_child_chunks(
        self, query: str, user_id: str, top_k: int = 200
    ) -> List[Dict]:
        """Retrieve initial child chunks based on query similarity using Weaviate."""
        try:
            # Use Weaviate's hybrid search to get child chunks with scores
            child_chunks = self.weaviate_manager.search_similar_chunks(
                query_text=query,
                user_id=user_id,
                limit=top_k,
                chunk_type="child",
                min_score=0.0,  # We'll filter later with our own threshold
            )

            if not child_chunks:
                logger.warning(f"No child chunks found for user {user_id}")
                return []

            # Convert Weaviate results to our expected format and apply manual filtering
            processed_chunks = []
            for chunk in child_chunks:
                # Apply our minimum relevance score threshold
                if chunk.get("score", 0) >= self.config.MIN_RELEVANCE_SCORE:
                    processed_chunk = {
                        "chunk_id": chunk["chunk_id"],
                        "chunk_text": chunk["chunk_text"],
                        "text": chunk["chunk_text"],  # For compatibility
                        "contextual_header": chunk.get("contextual_header", ""),
                        "relevance_score": chunk.get("score", 0),
                        "parent_chunk_id": chunk.get("parent_chunk_id"),
                    }
                    processed_chunks.append(processed_chunk)

            # Manual reranking by score (Weaviate already provides scores)
            processed_chunks.sort(key=lambda x: x["relevance_score"], reverse=True)

            logger.info(
                f"Retrieved {len(processed_chunks)} child chunks after filtering"
            )
            return processed_chunks

        except Exception as e:
            logger.error(f"Error retrieving child chunks: {e}")
            return []

    def get_parent_chunks_from_children(self, child_chunks: List[Dict]) -> List[Dict]:
        """Get parent chunks corresponding to the child chunks."""
        if not child_chunks:
            return []

        parent_chunks = []
        seen_parent_ids = set()

        try:
            for child_chunk in child_chunks:
                parent_chunk_id = child_chunk.get("parent_chunk_id")

                if parent_chunk_id and parent_chunk_id not in seen_parent_ids:
                    parent_chunk = self.weaviate_manager.get_chunk_by_id(
                        parent_chunk_id, chunk_type="parent"
                    )

                    if parent_chunk:
                        # Normalize the chunk format
                        if "text" not in parent_chunk and "chunk_text" in parent_chunk:
                            parent_chunk["text"] = parent_chunk["chunk_text"]
                        elif (
                            "chunk_text" not in parent_chunk and "text" in parent_chunk
                        ):
                            parent_chunk["chunk_text"] = parent_chunk["text"]

                        parent_chunks.append(parent_chunk)
                        seen_parent_ids.add(parent_chunk_id)

            logger.info(f"Retrieved {len(parent_chunks)} unique parent chunks")
            return parent_chunks

        except Exception as e:
            logger.error(f"Error getting parent chunks: {e}")
            return []

    def multi_question_retrieval(
        self, sub_questions: List[str], user_id: str
    ) -> Tuple[List[Dict], Dict[str, int]]:
        """Retrieve chunks for multiple sub-questions and combine results."""
        logger.info(f"Performing retrieval for {len(sub_questions)} sub-questions")

        all_parent_chunks = []
        seen_chunk_ids = set()
        total_tokens = {"total_tokens": 0}

        try:
            for i, sub_question in enumerate(sub_questions):
                logger.info(f"Processing sub-question {i+1}: {sub_question[:50]}...")

                # Retrieve child chunks for this sub-question
                child_chunks = self.retrieve_child_chunks(
                    sub_question,
                    user_id,
                    self.config.INITIAL_RETRIEVAL_COUNT // len(sub_questions),
                )

                # Get corresponding parent chunks
                parent_chunks = self.get_parent_chunks_from_children(child_chunks)

                # Add unique parent chunks to collection
                for parent_chunk in parent_chunks[
                    : self.config.PARENT_CHUNKS_COUNT // len(sub_questions)
                ]:
                    if parent_chunk["chunk_id"] not in seen_chunk_ids:
                        # Add metadata about which sub-question retrieved this chunk
                        parent_chunk["retrieved_by_question"] = sub_question
                        parent_chunk["question_index"] = i
                        all_parent_chunks.append(parent_chunk)
                        seen_chunk_ids.add(parent_chunk["chunk_id"])

            logger.info(
                f"Retrieved {len(all_parent_chunks)} unique parent chunks from all sub-questions"
            )
            return all_parent_chunks, total_tokens

        except Exception as e:
            logger.error(f"Error in multi-question retrieval: {e}")
            return [], total_tokens

    def diversified_retrieval(
        self, query: str, user_id: str
    ) -> Tuple[List[Dict], Dict[str, int]]:
        """Enhanced retrieval that combines query decomposition with traditional multi-query approach."""
        logger.info("Starting diversified retrieval")
        total_tokens = {"total_tokens": 0}

        try:
            # Step 1: Decompose the query into sub-questions
            sub_questions, decomp_tokens = self.decompose_complex_query(query)
            total_tokens["total_tokens"] += decomp_tokens.get("total_tokens", 0)

            # Step 2: Generate alternative phrasing for the original query
            alternative_query, alt_tokens = self.llm_client.generate_multi_query(query)
            total_tokens["total_tokens"] += alt_tokens.get("total_tokens", 0)

            # Step 3: Create comprehensive query list
            all_queries = [query, alternative_query] + sub_questions

            # Step 4: Retrieve chunks for all queries with MORE generous limits
            all_parent_chunks = []
            seen_chunk_ids = set()

            for i, search_query in enumerate(all_queries):
                logger.info(
                    f"Retrieving for query {i+1}/{len(all_queries)}: {search_query[:50]}..."
                )

                # MORE GENEROUS retrieval - don't divide by query count initially
                retrieval_count = max(150, self.config.INITIAL_RETRIEVAL_COUNT)

                child_chunks = self.retrieve_child_chunks(
                    search_query, user_id, retrieval_count
                )
                parent_chunks = self.get_parent_chunks_from_children(child_chunks)

                # Take MORE chunks per query - be generous here
                parent_count = max(20, self.config.PARENT_CHUNKS_COUNT)

                for parent_chunk in parent_chunks[:parent_count]:
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
                f"Diversified retrieval found {len(all_parent_chunks)} unique chunks"
            )
            return all_parent_chunks, total_tokens

        except Exception as e:
            logger.error(f"Error in diversified retrieval: {e}")
            return [], total_tokens

    def final_reranking_with_coverage(
        self, original_query: str, parent_chunks: List[Dict]
    ) -> List[Dict]:
        """Enhanced reranking using Weaviate scores with coverage consideration."""
        if not parent_chunks:
            logger.warning("No parent chunks to rerank")
            return []

        try:
            # Get fresh similarity scores for parent chunks against original query
            enhanced_chunks = []

            # We'll use Weaviate to get fresh scores for parent chunks
            # against the original query
            parent_chunk_texts = [
                chunk.get("chunk_text", "") or chunk.get("text", "")
                for chunk in parent_chunks
            ]

            if not any(parent_chunk_texts):
                logger.warning("No parent texts found for reranking")
                return []

            # Get scores using Weaviate search on parent chunks
            parent_search_results = self.weaviate_manager.search_similar_chunks(
                query_text=original_query,
                user_id=parent_chunks[0].get(
                    "user_id", ""
                ),  # Assume same user for all chunks
                limit=len(parent_chunks) * 2,  # Get more than we need
                chunk_type="parent",
                min_score=0.0,
            )

            # Create a mapping of chunk_id to Weaviate score
            score_mapping = {}
            for result in parent_search_results:
                score_mapping[result["chunk_id"]] = result.get("score", 0)

            # Enhanced scoring that considers diversity
            for chunk in parent_chunks:
                enhanced_chunk = chunk.copy()
                chunk_id = chunk["chunk_id"]

                # Get Weaviate similarity score
                weaviate_score = score_mapping.get(chunk_id, 0)
                enhanced_chunk["relevance_score"] = weaviate_score

                # Boost score slightly if retrieved by sub-question (promotes diversity)
                if chunk.get("query_type") == "sub_question":
                    enhanced_chunk["relevance_score"] *= 1.1

                enhanced_chunks.append(enhanced_chunk)

            # Sort by enhanced relevance score
            enhanced_chunks.sort(key=lambda x: x["relevance_score"], reverse=True)

            # Filter by minimum score
            filtered_chunks = [
                chunk
                for chunk in enhanced_chunks
                if chunk["relevance_score"] >= self.config.MIN_RELEVANCE_SCORE
            ]

            # Return final top chunks
            final_chunks = filtered_chunks[: self.config.FINAL_CHUNKS_COUNT]
            logger.info(
                f"Final reranking with coverage produced {len(final_chunks)} chunks"
            )

            return final_chunks

        except Exception as e:
            logger.error(f"Error in final reranking with coverage: {e}")
            # Fallback to simple sorting if reranking fails
            parent_chunks.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            return parent_chunks[: self.config.FINAL_CHUNKS_COUNT]

    def retrieve_and_generate(
        self, query: str, user_id: str
    ) -> Tuple[str, List[Dict], float, int]:
        """Complete enhanced RAG pipeline with improved multi-document retrieval."""
        start_time = time.time()
        total_tokens = 0

        logger.info(f"Starting enhanced RAG retrieval for query: {query[:100]}...")

        try:
            # Step 1: Diversified retrieval (combines decomposition + multi-query)
            parent_chunks, retrieval_tokens = self.diversified_retrieval(query, user_id)

            # Step 1.5: Fallback retrieval if we don't have enough chunks
            if len(parent_chunks) < 10:
                logger.warning(
                    f"Only got {len(parent_chunks)} chunks from diversified retrieval, trying direct retrieval"
                )

                # Direct retrieval with high limits
                direct_child_chunks = self.retrieve_child_chunks(
                    query, user_id, top_k=400
                )
                direct_parent_chunks = self.get_parent_chunks_from_children(
                    direct_child_chunks
                )

                # Add to existing chunks
                seen_chunk_ids = {chunk["chunk_id"] for chunk in parent_chunks}
                for chunk in direct_parent_chunks:
                    if chunk["chunk_id"] not in seen_chunk_ids:
                        chunk["query_type"] = "direct_fallback"
                        parent_chunks.append(chunk)
                        seen_chunk_ids.add(chunk["chunk_id"])

                logger.info(
                    f"After fallback retrieval: {len(parent_chunks)} total chunks"
                )

            if not parent_chunks:
                logger.warning("No relevant chunks found")
                processing_time = time.time() - start_time
                return (
                    "I couldn't find relevant information to answer your question. Please try rephrasing your query or upload more documents.",
                    [],
                    processing_time,
                    retrieval_tokens.get("total_tokens", 0),
                )

            # Step 2: Enhanced final reranking with coverage consideration
            final_chunks = self.final_reranking_with_coverage(query, parent_chunks)

            if not final_chunks:
                logger.warning("No chunks passed final reranking")
                processing_time = time.time() - start_time
                return (
                    "I couldn't find sufficiently relevant information to answer your question.",
                    [],
                    processing_time,
                    retrieval_tokens.get("total_tokens", 0),
                )

            # Step 3: Generate comprehensive answer
            answer, answer_tokens = self.llm_client.generate_answer(query, final_chunks)

            # Calculate total tokens used
            total_tokens = retrieval_tokens.get("total_tokens", 0) + answer_tokens.get(
                "total_tokens", 0
            )

            processing_time = time.time() - start_time
            logger.info(
                f"Enhanced RAG retrieval completed in {processing_time:.2f} seconds"
            )
            logger.info(f"Total tokens used: {total_tokens}")
            logger.info(f"Final chunks passed to LLM: {len(final_chunks)}")

            return answer, final_chunks, processing_time, total_tokens

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error in enhanced retrieve_and_generate: {e}")
            return (
                f"An error occurred while processing your query: {str(e)}",
                [],
                processing_time,
                0,
            )
