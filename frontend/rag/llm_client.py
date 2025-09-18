import os
import requests
from groq import Groq
from typing import List, Dict, Any, Tuple
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


def limit_model_input(input_str: str, max_len: int = 44000):
    """Truncates the input string to a maximum length."""
    if len(input_str) > max_len:
        return input_str[:max_len]
    return input_str


class BaseLLMClient(ABC):
    """Base class for LLM clients with shared functionality."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name

    @abstractmethod
    def _make_chat_completion(
        self, messages: List[Dict], max_tokens: int = 2000, temperature: float = 0.3
    ) -> Any:
        """Make a chat completion request. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _extract_content(self, response: Any) -> str:
        """Extract content from response. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def get_token_usage(self, response: Any) -> Dict[str, int]:
        """Extract token usage from response. Must be implemented by subclasses."""
        pass

    def generate_multi_query(self, original_query: str) -> Tuple[str, Dict[str, int]]:
        """Generate an alternative query for multi-query retrieval."""
        prompt = f"""Given the following user question, generate ONE alternative question that asks for the same information but uses different words and phrasing. The alternative question should help retrieve relevant documents that might not match the original query exactly.

Original question: {original_query}

Alternative question:"""

        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that generates alternative search queries.",
                },
                {
                    "role": "user",
                    "content": limit_model_input(prompt, max_len=2000),
                },
            ]

            response = self._make_chat_completion(
                messages=messages, max_tokens=100, temperature=0.7
            )

            alternative_query = self._extract_content(response).strip()
            token_usage = self.get_token_usage(response)

            logger.info(f"Generated alternative query: {alternative_query}")
            return alternative_query, token_usage

        except Exception as e:
            logger.error(f"Failed to generate alternative query: {e}")
            return original_query, {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

    def generate_answer(
        self, query: str, context_chunks: List[Dict[str, Any]]
    ) -> Tuple[str, Dict[str, int]]:
        """Generate answer based on query and retrieved context chunks."""

        # Define a maximum character limit for the entire prompt payload
        MAX_PROMPT_LEN = 44000

        # Define the prompt structure with placeholders for context and query
        prompt_template = """Based on the provided context, answer the following question. If the context doesn't contain enough information to answer the question completely, say so and provide what information you can find.

Context:
{context_text}

Question: {query}

Answer:"""

        # Calculate the length of the prompt's boilerplate (everything EXCEPT the context)
        boilerplate_prompt = prompt_template.format(context_text="", query=query)
        boilerplate_len = len(boilerplate_prompt)

        # Calculate the maximum character length available for the context
        max_len_for_context = MAX_PROMPT_LEN - boilerplate_len

        # Iteratively build the context, stopping when it's full
        context_text = ""
        for i, chunk in enumerate(context_chunks, 1):
            header = chunk.get("contextual_header", "")
            text = chunk.get("text", "")
            chunk_str = f"[Source {i}] {header}\n{text}\n\n"

            # If adding the next chunk exceeds the available space, stop
            if len(context_text) + len(chunk_str) > max_len_for_context:
                logger.warning(
                    f"Truncating context after {i-1} chunks to fit model input limits."
                )
                break

            context_text += chunk_str

        # Assemble the final prompt with the (potentially truncated) context
        final_prompt = prompt_template.format(context_text=context_text, query=query)

        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on provided context. Always cite which sources you used in your answer using [Source X] format.",
                },
                {"role": "user", "content": final_prompt},
            ]

            response = self._make_chat_completion(
                messages=messages, max_tokens=2000, temperature=0.3
            )

            answer = self._extract_content(response).strip()
            token_usage = self.get_token_usage(response)
            logger.info(f"Generated answer successfully")
            return answer, token_usage

        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return (
                "I apologize, but I encountered an error while generating the answer. Please try again.",
                {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            )


class GroqLLMClient(BaseLLMClient):
    def __init__(
        self,
        api_key: str = os.getenv("GROQ_API_KEY"),
        model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct",
    ):
        super().__init__(model_name)
        self.client = Groq(api_key=api_key)

    def _make_chat_completion(
        self, messages: List[Dict], max_tokens: int = 2000, temperature: float = 0.3
    ) -> Any:
        """Make a chat completion request using Groq."""
        logger.info(f"Using {self.model_name}")
        return self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def _extract_content(self, response: Any) -> str:
        """Extract content from Groq response."""
        return response.choices[0].message.content

    def get_token_usage(self, response) -> Dict[str, int]:
        """Extract token usage from Groq response."""
        if hasattr(response, "usage") and response.usage:
            return {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


class SanctuaryLLMClient(BaseLLMClient):
    def __init__(
        self,
        api_key: str = os.getenv("SANCTUARY_API_KEY"),
        base_url: str = "https://api-sanctuary.i2cv.io",
        model_name: str = "bedrock-claude-3-5-sonnet-v1",
    ):
        super().__init__(model_name)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        )

    def _make_chat_completion(
        self, messages: List[Dict], max_tokens: int = 2000, temperature: float = 0.3
    ) -> Any:
        """Make a chat completion request using Sanctuary API."""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        logger.info(f"Using {self.model_name}")
        response = self.session.post(
            f"{self.base_url}/v1/chat/completions", json=payload
        )
        response.raise_for_status()
        return response.json()

    def _extract_content(self, response: Any) -> str:
        """Extract content from Sanctuary response."""
        return response["choices"][0]["message"]["content"]

    def get_token_usage(self, response) -> Dict[str, int]:
        """Extract token usage from Sanctuary response."""
        if "usage" in response and response["usage"]:
            usage = response["usage"]
            return {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
