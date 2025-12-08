"""
OpenAI API Client Module.

Provides a wrapper for OpenAI API with support for:
- Chat completions with structured JSON output
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
    - Chat completions with JSON structured output (gpt-4o-mini)
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
            model: Model to use for chat completions (default: gpt-4o-mini).
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
        Send a chat completion request with structured JSON output (sync).

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
                request_kwargs = self._build_chat_kwargs(messages, temperature)

                response = self.client.chat.completions.create(**request_kwargs)
                duration = time.perf_counter() - start

                return self._parse_chat_response(response, response_model, duration)

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
        Send a chat completion request with structured JSON output (async).

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
                request_kwargs = self._build_chat_kwargs(messages, temperature)

                response = await self.async_client.chat.completions.create(**request_kwargs)
                duration = time.perf_counter() - start

                return self._parse_chat_response(response, response_model, duration)

            except Exception as e:
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
        """Build messages list for chat completion."""
        schema_json = response_model.model_json_schema()
        json_instruction = (
            f"You must respond with valid JSON matching this schema:\n"
            f"{schema_json}\n\n"
            "Do not include any text outside the JSON object."
        )

        full_system_prompt = json_instruction
        if system_prompt:
            full_system_prompt = f"{system_prompt}\n\n{json_instruction}"

        return [
            {"role": "system", "content": full_system_prompt},
            {"role": "user", "content": prompt},
        ]

    def _build_chat_kwargs(
        self,
        messages: list[dict],
        temperature: Optional[float],
    ) -> dict:
        """Build kwargs for chat completion request."""
        request_kwargs = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": 4000,
            "response_format": {"type": "json_object"},
        }

        if temperature is not None:
            request_kwargs["temperature"] = temperature

        return request_kwargs

    def _parse_chat_response(
        self,
        response,
        response_model: Type[T],
        duration: float,
    ) -> T:
        """Parse chat completion response into Pydantic model."""
        usage = getattr(response, "usage", None)
        token_summary = ""
        if usage:
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None)
            total_tokens = getattr(usage, "total_tokens", None)
            token_summary = (
                f" prompt={prompt_tokens} completion={completion_tokens} total={total_tokens}"
            )

        message = response.choices[0].message
        raw_content: Union[str, dict, BaseModel, None] = getattr(message, "content", None)
        parsed = getattr(message, "parsed", None)

        if parsed is not None:
            if isinstance(parsed, BaseModel):
                logger.info(
                    "LLM structured success model=%s duration=%.3fs%s",
                    self.model,
                    duration,
                    token_summary,
                )
                return response_model.model_validate(parsed.model_dump())
            if isinstance(parsed, dict):
                logger.info(
                    "LLM structured success model=%s duration=%.3fs%s",
                    self.model,
                    duration,
                    token_summary,
                )
                return response_model.model_validate(parsed)
            if isinstance(parsed, str):
                raw_content = parsed

        if raw_content is None or (isinstance(raw_content, str) and not raw_content.strip()):
            raise ValueError("Received empty content from LLM")

        if isinstance(raw_content, dict):
            logger.info(
                "LLM structured success model=%s duration=%.3fs%s",
                self.model,
                duration,
                token_summary,
            )
            return response_model.model_validate(raw_content)

        logger.info(
            "LLM structured success model=%s duration=%.3fs%s",
            self.model,
            duration,
            token_summary,
        )
        return response_model.model_validate_json(str(raw_content))

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
