"""
OpenAI GPT-5 Client Module.

Provides a wrapper for OpenAI API with support for:
- GPT-5 chat completions with structured output
- Text embeddings for semantic matching
- Rate limiting and error handling
"""

import os
import time
from typing import Optional, Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel

# * Configuration
DEFAULT_MODEL = "gpt-5"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
MAX_RETRIES = 3
RETRY_DELAY = 1.0


T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """
    OpenAI GPT-5 client with structured output support.

    Supports:
    - Chat completions with JSON structured output
    - Text embeddings for semantic similarity
    - Automatic retries with exponential backoff
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ):
        """
        Initialize the LLM client.

        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            model: Model to use for chat completions (default: gpt-5).
            embedding_model: Model to use for embeddings.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key to the constructor."
            )

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.embedding_model = embedding_model

    def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """
        Send a chat completion request.

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature (0-2).
            max_tokens: Maximum tokens in response.

        Returns:
            Model response text.
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                return response.choices[0].message.content or ""

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY * (2 ** attempt)
                    print(f"? Retry {attempt + 1}/{MAX_RETRIES} after {delay}s: {e}")
                    time.sleep(delay)
                else:
                    raise RuntimeError(f"Failed after {MAX_RETRIES} attempts: {e}") from e

        return ""

    def chat_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
    ) -> T:
        """
        Send a chat completion request with structured JSON output.

        Args:
            prompt: User prompt.
            response_model: Pydantic model class for the response.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature (lower for more deterministic).

        Returns:
            Parsed response as Pydantic model instance.
        """
        messages = []

        # * Build system prompt with JSON schema
        schema_json = response_model.model_json_schema()
        json_instruction = (
            f"You must respond with valid JSON matching this schema:\n"
            f"{schema_json}\n\n"
            "Do not include any text outside the JSON object."
        )

        full_system_prompt = json_instruction
        if system_prompt:
            full_system_prompt = f"{system_prompt}\n\n{json_instruction}"

        messages.append({"role": "system", "content": full_system_prompt})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=4000,
                    response_format={"type": "json_object"},
                )

                content = response.choices[0].message.content or "{}"

                # * Parse and validate with Pydantic
                return response_model.model_validate_json(content)

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY * (2 ** attempt)
                    print(f"? Retry {attempt + 1}/{MAX_RETRIES} after {delay}s: {e}")
                    time.sleep(delay)
                else:
                    raise RuntimeError(f"Failed after {MAX_RETRIES} attempts: {e}") from e

        # * Should not reach here, but satisfy type checker
        raise RuntimeError("Unexpected error in chat_structured")

    def get_embedding(self, text: str) -> list[float]:
        """
        Get embedding vector for text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as list of floats.
        """
        # * Truncate very long texts
        max_chars = 8000
        if len(text) > max_chars:
            text = text[:max_chars]

        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=text,
                )

                return response.data[0].embedding

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY * (2 ** attempt)
                    print(f"? Retry {attempt + 1}/{MAX_RETRIES} after {delay}s: {e}")
                    time.sleep(delay)
                else:
                    raise RuntimeError(f"Failed after {MAX_RETRIES} attempts: {e}") from e

        return []

    def get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Get embedding vectors for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        # * Truncate and filter empty texts
        max_chars = 8000
        processed_texts = []
        for text in texts:
            if text.strip():
                processed_texts.append(text[:max_chars] if len(text) > max_chars else text)

        if not processed_texts:
            return []

        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=processed_texts,
                )

                # * Return embeddings in same order as input
                return [item.embedding for item in response.data]

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY * (2 ** attempt)
                    print(f"? Retry {attempt + 1}/{MAX_RETRIES} after {delay}s: {e}")
                    time.sleep(delay)
                else:
                    raise RuntimeError(f"Failed after {MAX_RETRIES} attempts: {e}") from e

        return []


def get_client(api_key: Optional[str] = None) -> LLMClient:
    """
    Get a configured LLM client instance.

    Args:
        api_key: Optional API key override.

    Returns:
        Configured LLMClient instance.
    """
    return LLMClient(api_key=api_key)

