"""
OpenAI API Client Module.

Provides a wrapper for OpenAI API with support for:
- Structured outputs using Responses API (new models: gpt-4o-mini, gpt-4o-2024-08-06+)
- Text embeddings for semantic matching
- Automatic retries with exponential backoff
- Async methods for parallel execution
"""

import asyncio
import logging
import os
import time
from typing import Optional, Type, TypeVar, Union

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

# * Configuration
# * DEFAULT_MODEL: Must support Responses API with Structured Outputs
# * Supported models: gpt-4o-mini, gpt-4o-2024-08-06, gpt-5-mini, and later models
DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
MAX_RETRIES = 3
RETRY_DELAY = 1.0


T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger("resume_matcher.llm")


class LLMClient:
    """
    OpenAI API client with structured output and embedding support.

    Supports:
    - Structured outputs using Responses API (gpt-4o-mini, gpt-4o-2024-08-06+)
    - Text embeddings for semantic similarity (text-embedding-3-small)
    - Automatic retries with exponential backoff
    - Async methods for parallel execution
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
            model: Model to use for structured outputs (default: gpt-4o-2024-08-06).
                  Must support Structured Outputs (gpt-4o-mini, gpt-4o-2024-08-06+).
            embedding_model: Model to use for embeddings.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key to the constructor."
            )

        # * Sync and async clients
        self.client = OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)
        self.model = model
        self.embedding_model = embedding_model

    def chat_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> T:
        """
        Send a request with structured output using Responses API (sync).

        Uses Responses API parse() method with Structured Outputs for new models (gpt-4o-mini, gpt-4o-2024-08-06+)
        which ensures schema adherence and type-safety. The parse() method is specifically designed
        for structured outputs with Pydantic models.

        Args:
            prompt: User prompt.
            response_model: Pydantic model class for the response.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature (lower for more deterministic).

        Returns:
            Parsed response as Pydantic model instance.
        """
        messages = self._build_messages(prompt, response_model, system_prompt)

        for attempt in range(MAX_RETRIES):
            try:
                start = time.perf_counter()
                
                # * Use Responses API parse() method for structured outputs with Pydantic models
                # * responses.parse() is specifically designed for structured outputs
                response = self.client.responses.parse(
                    model=self.model,
                    input=messages,
                    text_format=response_model,
                    temperature=temperature,
                )
                duration = time.perf_counter() - start

                # * Responses API parse() returns structured output in response.output[0].content[0].parsed
                # * Check response structure with proper None handling
                if not hasattr(response, "output") or not response.output:
                    raise ValueError("Responses API returned no output")
                
                if len(response.output) == 0:
                    raise ValueError("Responses API returned empty output array")
                
                message = response.output[0]
                
                # * Check if content exists and is not None
                if not hasattr(message, "content") or message.content is None:
                    # * Check for refusal message
                    refusal = getattr(message, "refusal", None)
                    if refusal:
                        raise ValueError(f"Model refused: {refusal}")
                    raise ValueError("Responses API returned no content (content is None)")
                
                if len(message.content) == 0:
                    raise ValueError("Responses API returned empty content array")
                
                text = message.content[0]
                
                if hasattr(text, "parsed") and text.parsed is not None:
                    usage = getattr(response, "usage", None)
                    token_summary = ""
                    if usage:
                        prompt_tokens = getattr(usage, "prompt_tokens", None)
                        completion_tokens = getattr(usage, "completion_tokens", None)
                        total_tokens = getattr(usage, "total_tokens", None)
                        token_summary = (
                            f" prompt={prompt_tokens} completion={completion_tokens} total={total_tokens}"
                        )
                    
                    logger.info(
                        "LLM structured success model=%s duration=%.3fs%s",
                        self.model,
                        duration,
                        token_summary,
                    )
                    return text.parsed
                else:
                    # * Check for refusal message
                    refusal = getattr(text, "refusal", None)
                    if refusal:
                        raise ValueError(f"Model refused: {refusal}")
                    raise ValueError("Responses API returned no parsed output")

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY * (2 ** attempt)
                    logger.warning(
                        "LLM structured retry %s/%s in %.1fs due to: %s",
                        attempt + 1,
                        MAX_RETRIES,
                        delay,
                        e,
                    )
                    time.sleep(delay)
                else:
                    logger.error("LLM structured call failed after retries: %s", e, exc_info=True)
                    raise RuntimeError(f"Failed after {MAX_RETRIES} attempts: {e}") from e

        raise RuntimeError("Unexpected error in chat_structured")

    async def chat_structured_async(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> T:
        """
        Send a request with structured output using Responses API (async).

        Uses Responses API parse() method with Structured Outputs for new models (gpt-4o-mini, gpt-4o-2024-08-06+)
        which ensures schema adherence and type-safety. The parse() method is specifically designed
        for structured outputs with Pydantic models.

        Args:
            prompt: User prompt.
            response_model: Pydantic model class for the response.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature (lower for more deterministic).

        Returns:
            Parsed response as Pydantic model instance.
        """
        messages = self._build_messages(prompt, response_model, system_prompt)

        for attempt in range(MAX_RETRIES):
            try:
                start = time.perf_counter()
                
                # * Use Responses API parse() method for structured outputs with Pydantic models
                # * responses.parse() is specifically designed for structured outputs
                response = await self.async_client.responses.parse(
                    model=self.model,
                    input=messages,
                    text_format=response_model,
                    temperature=temperature,
                )
                duration = time.perf_counter() - start

                # * Responses API parse() returns structured output in response.output[0].content[0].parsed
                # * Check response structure with proper None handling
                if not hasattr(response, "output") or not response.output:
                    logger.error("Response structure: has output=%s", hasattr(response, "output"))
                    raise ValueError("Responses API returned no output")
                
                if len(response.output) == 0:
                    raise ValueError("Responses API returned empty output array")
                
                message = response.output[0]
                
                # * Check if content exists and is not None
                if not hasattr(message, "content") or message.content is None:
                    # * Log response structure for debugging
                    logger.error(
                        "Response structure error: message.type=%s, has content=%s, content=%s",
                        getattr(message, "type", None),
                        hasattr(message, "content"),
                        message.content if hasattr(message, "content") else "N/A",
                    )
                    # * Check for refusal message
                    refusal = getattr(message, "refusal", None)
                    if refusal:
                        raise ValueError(f"Model refused: {refusal}")
                    raise ValueError("Responses API returned no content (content is None)")
                
                if len(message.content) == 0:
                    raise ValueError("Responses API returned empty content array")
                
                text = message.content[0]
                
                if hasattr(text, "parsed") and text.parsed is not None:
                    usage = getattr(response, "usage", None)
                    token_summary = ""
                    if usage:
                        prompt_tokens = getattr(usage, "prompt_tokens", None)
                        completion_tokens = getattr(usage, "completion_tokens", None)
                        total_tokens = getattr(usage, "total_tokens", None)
                        token_summary = (
                            f" prompt={prompt_tokens} completion={completion_tokens} total={total_tokens}"
                        )
                    
                    logger.info(
                        "LLM structured async success model=%s duration=%.3fs%s",
                        self.model,
                        duration,
                        token_summary,
                    )
                    return text.parsed
                else:
                    # * Check for refusal message
                    refusal = getattr(text, "refusal", None)
                    if refusal:
                        raise ValueError(f"Model refused: {refusal}")
                    raise ValueError("Responses API returned no parsed output")

            except Exception as e:
                # * Log more details about the error
                logger.error(
                    "LLM structured async error attempt %s/%s: %s (type: %s)",
                    attempt + 1,
                    MAX_RETRIES,
                    e,
                    type(e).__name__,
                    exc_info=True,
                )
                
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY * (2 ** attempt)
                    logger.warning(
                        "LLM structured async retry %s/%s in %.1fs due to: %s",
                        attempt + 1,
                        MAX_RETRIES,
                        delay,
                        e,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error("LLM structured async failed after retries: %s", e, exc_info=True)
                    raise RuntimeError(f"Failed after {MAX_RETRIES} attempts: {e}") from e

        raise RuntimeError("Unexpected error in chat_structured_async")

    def _build_messages(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: Optional[str],
    ) -> list[dict]:
        """
        Build messages list for Responses API.

        Responses API parse() method accepts input as either:
        - A string (simple text input)
        - A list of message objects with role and content

        Note: With Structured Outputs, we don't need JSON formatting instructions
        in the prompt - the schema is handled automatically by the API via text_format parameter.
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        return messages

    def get_embedding(self, text: str) -> list[float]:
        """
        Get embedding vector for text using OpenAI embeddings (sync).

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
                start = time.perf_counter()
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=text,
                )
                duration = time.perf_counter() - start

                logger.info(
                    "Embedding success model=%s duration=%.3fs",
                    self.embedding_model,
                    duration,
                )

                return response.data[0].embedding

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY * (2 ** attempt)
                    logger.warning(
                        "Embedding retry %s/%s in %.1fs due to: %s",
                        attempt + 1,
                        MAX_RETRIES,
                        delay,
                        e,
                    )
                    time.sleep(delay)
                else:
                    logger.error("Embedding failed after retries: %s", e, exc_info=True)
                    raise RuntimeError(f"Failed after {MAX_RETRIES} attempts: {e}") from e

        return []

    async def get_embedding_async(self, text: str) -> list[float]:
        """
        Get embedding vector for text using OpenAI embeddings (async).

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
                start = time.perf_counter()
                response = await self.async_client.embeddings.create(
                    model=self.embedding_model,
                    input=text,
                )
                duration = time.perf_counter() - start

                logger.info(
                    "Embedding async success model=%s duration=%.3fs",
                    self.embedding_model,
                    duration,
                )

                return response.data[0].embedding

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY * (2 ** attempt)
                    logger.warning(
                        "Embedding async retry %s/%s in %.1fs due to: %s",
                        attempt + 1,
                        MAX_RETRIES,
                        delay,
                        e,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error("Embedding async failed after retries: %s", e, exc_info=True)
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
