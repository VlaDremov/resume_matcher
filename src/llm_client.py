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
import re
import time
from dataclasses import dataclass
from typing import Optional, Type, TypeVar, Union

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

# * Configuration
# * DEFAULT_MODEL: Must support Responses API with Structured Outputs
# * Supported models: gpt-4o-mini, gpt-4o-2024-08-06, gpt-5-mini, and later models
DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_REASONING_EFFORT = "minimal"
MAX_RETRIES = 3
RETRY_DELAY = 1.0


T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger("resume_matcher.llm")


def _normalize_model_key(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", model).upper()


def _read_float_env(key: str) -> Optional[float]:
    raw = os.getenv(key)
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid numeric value for %s=%r", key, raw)
        return None


def _pricing_env_keys(model: str) -> tuple[str, str]:
    model_key = _normalize_model_key(model)
    return (
        f"OPENAI_PRICE_{model_key}_INPUT_PER_1M",
        f"OPENAI_PRICE_{model_key}_OUTPUT_PER_1M",
    )


def _get_pricing_for_model(model: str) -> tuple[Optional[float], Optional[float]]:
    model_input_key, model_output_key = _pricing_env_keys(model)
    input_price = _read_float_env(model_input_key)
    output_price = _read_float_env(model_output_key)
    if input_price is None:
        input_price = _read_float_env("OPENAI_PRICE_INPUT_PER_1M")
    if output_price is None:
        output_price = _read_float_env("OPENAI_PRICE_OUTPUT_PER_1M")
    return input_price, output_price


def _default_reasoning_effort() -> Optional[str]:
    raw = os.getenv("OPENAI_REASONING_EFFORT", DEFAULT_REASONING_EFFORT)
    if raw is None:
        return None
    raw = raw.strip().lower()
    if raw in ("", "none", "off", "false", "0"):
        return None
    return raw


@dataclass
class UsageAccumulator:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    price_input_per_1m: Optional[float] = None
    price_output_per_1m: Optional[float] = None


class LLMUsageTracker:
    def __init__(self) -> None:
        self.by_model: dict[str, UsageAccumulator] = {}
        self.pricing_missing_models: set[str] = set()

    def _normalize_usage(self, usage) -> Optional[dict]:
        if usage is None:
            return None

        input_tokens = getattr(usage, "input_tokens", None)
        if input_tokens is None:
            input_tokens = getattr(usage, "prompt_tokens", None)

        output_tokens = getattr(usage, "output_tokens", None)
        if output_tokens is None:
            output_tokens = getattr(usage, "completion_tokens", None)
        if output_tokens is None:
            output_tokens = 0

        total_tokens = getattr(usage, "total_tokens", None)
        if total_tokens is None and input_tokens is not None:
            total_tokens = input_tokens + output_tokens

        if input_tokens is None or total_tokens is None:
            return None

        reasoning_tokens = 0
        output_details = getattr(usage, "output_tokens_details", None)
        if output_details is not None:
            reasoning_tokens = getattr(output_details, "reasoning_tokens", 0) or 0

        cached_tokens = 0
        input_details = getattr(usage, "input_tokens_details", None)
        if input_details is not None:
            cached_tokens = getattr(input_details, "cached_tokens", 0) or 0

        return {
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "total_tokens": int(total_tokens),
            "reasoning_tokens": int(reasoning_tokens),
            "cached_tokens": int(cached_tokens),
        }

    def record_usage(self, model: str, usage) -> Optional[dict]:
        usage_info = self._normalize_usage(usage)
        if usage_info is None:
            return None

        input_tokens = usage_info["input_tokens"]
        output_tokens = usage_info["output_tokens"]
        total_tokens = usage_info["total_tokens"]
        reasoning_tokens = usage_info["reasoning_tokens"]
        cached_tokens = usage_info["cached_tokens"]

        input_price, output_price = _get_pricing_for_model(model)
        cost_usd = 0.0
        if input_price is not None:
            cost_usd += (input_tokens / 1_000_000) * input_price
        if output_price is not None:
            cost_usd += (output_tokens / 1_000_000) * output_price

        missing_pricing = False
        if input_tokens > 0 and input_price is None:
            missing_pricing = True
        if output_tokens > 0 and output_price is None:
            missing_pricing = True
        if missing_pricing:
            self.pricing_missing_models.add(model)

        acc = self.by_model.setdefault(model, UsageAccumulator())
        acc.requests += 1
        acc.input_tokens += input_tokens
        acc.output_tokens += output_tokens
        acc.total_tokens += total_tokens
        acc.reasoning_tokens += reasoning_tokens
        acc.cached_tokens += cached_tokens
        acc.cost_usd += cost_usd
        if input_price is not None:
            acc.price_input_per_1m = input_price
        if output_price is not None:
            acc.price_output_per_1m = output_price

        return {
            **usage_info,
            "cost_usd": cost_usd,
            "pricing_missing": missing_pricing,
            "price_input_per_1m": input_price,
            "price_output_per_1m": output_price,
        }

    def summary(self) -> dict:
        totals = UsageAccumulator()
        by_model_summary: dict[str, dict] = {}

        for model, acc in self.by_model.items():
            totals.requests += acc.requests
            totals.input_tokens += acc.input_tokens
            totals.output_tokens += acc.output_tokens
            totals.total_tokens += acc.total_tokens
            totals.reasoning_tokens += acc.reasoning_tokens
            totals.cached_tokens += acc.cached_tokens
            totals.cost_usd += acc.cost_usd

            by_model_summary[model] = {
                "requests": acc.requests,
                "input_tokens": acc.input_tokens,
                "output_tokens": acc.output_tokens,
                "total_tokens": acc.total_tokens,
                "reasoning_tokens": acc.reasoning_tokens,
                "cached_tokens": acc.cached_tokens,
                "cost_usd": acc.cost_usd,
                "price_input_per_1m": acc.price_input_per_1m,
                "price_output_per_1m": acc.price_output_per_1m,
            }

        return {
            "total_requests": totals.requests,
            "total_input_tokens": totals.input_tokens,
            "total_output_tokens": totals.output_tokens,
            "total_tokens": totals.total_tokens,
            "total_reasoning_tokens": totals.reasoning_tokens,
            "total_cached_tokens": totals.cached_tokens,
            "total_cost_usd": totals.cost_usd,
            "by_model": by_model_summary,
            "pricing_missing_models": sorted(self.pricing_missing_models),
        }


USAGE_TRACKER = LLMUsageTracker()


def _format_usage_for_log(usage_info: Optional[dict]) -> str:
    if not usage_info:
        return ""

    parts = [
        f" input={usage_info['input_tokens']}",
        f" output={usage_info['output_tokens']}",
        f" total={usage_info['total_tokens']}",
    ]

    if usage_info.get("reasoning_tokens"):
        parts.append(f" reasoning={usage_info['reasoning_tokens']}")
    if usage_info.get("cached_tokens"):
        parts.append(f" cached={usage_info['cached_tokens']}")

    cost_usd = usage_info.get("cost_usd")
    if cost_usd is not None:
        parts.append(f" cost=${cost_usd:.6f}")
        if usage_info.get("pricing_missing"):
            parts.append(" pricing_missing=True")

    return "".join(parts)


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
        reasoning_effort: Optional[str] = None,
    ):
        """
        Initialize the LLM client.

        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            model: Model to use for structured outputs (default: gpt-4o-2024-08-06).
                  Must support Structured Outputs (gpt-4o-mini, gpt-4o-2024-08-06+).
            embedding_model: Model to use for embeddings.
            reasoning_effort: Reasoning effort for gpt-5/o-series models.
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
        self.reasoning_effort = reasoning_effort if reasoning_effort is not None else _default_reasoning_effort()
        self.usage_tracker = USAGE_TRACKER

    def _supports_reasoning_effort(self, model: Optional[str] = None) -> bool:
        model_name = (model or self.model or "").lower()
        if model_name.startswith("gpt-5"):
            return True
        return re.match(r"^o\d", model_name) is not None

    def _build_response_kwargs(
        self,
        messages: list[dict],
        response_model: Type[T],
        temperature: Optional[float],
    ) -> dict:
        kwargs = {
            "model": self.model,
            "input": messages,
            "text_format": response_model,
            "temperature": temperature,
        }
        if self.reasoning_effort and self._supports_reasoning_effort():
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        return kwargs

    def _record_usage_from_response(self, response) -> Optional[dict]:
        usage = getattr(response, "usage", None)
        model = getattr(response, "model", self.model)
        return self.usage_tracker.record_usage(model, usage)

    def _record_usage_from_embedding(self, response) -> Optional[dict]:
        usage = getattr(response, "usage", None)
        model = getattr(response, "model", self.embedding_model)
        return self.usage_tracker.record_usage(model, usage)

    def _extract_refusal(self, response) -> Optional[str]:
        output_items = getattr(response, "output", None) or []
        for item in output_items:
            if getattr(item, "type", None) == "message":
                refusal = getattr(item, "refusal", None)
                if refusal:
                    return refusal
                content_items = getattr(item, "content", None) or []
                for content in content_items:
                    refusal = getattr(content, "refusal", None)
                    if refusal:
                        return refusal
        return None

    def _log_response_structure_error(self, response, usage_info: Optional[dict]) -> None:
        output_items = getattr(response, "output", None) or []
        output_types = [getattr(item, "type", None) for item in output_items]
        message_content_types = []
        for item in output_items:
            if getattr(item, "type", None) == "message":
                content_items = getattr(item, "content", None) or []
                message_content_types.append(
                    [getattr(content, "type", None) for content in content_items]
                )

        output_text = getattr(response, "output_text", "") or ""

        logger.error(
            "Response structure error: status=%s error=%s incomplete=%s output_types=%s "
            "message_content_types=%s output_text_len=%s%s",
            getattr(response, "status", None),
            getattr(response, "error", None),
            getattr(response, "incomplete_details", None),
            output_types,
            message_content_types,
            len(output_text),
            _format_usage_for_log(usage_info),
        )

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
                    **self._build_response_kwargs(messages, response_model, temperature)
                )
                duration = time.perf_counter() - start
                usage_info = self._record_usage_from_response(response)

                model_name = getattr(response, "model", self.model)
                parsed = getattr(response, "output_parsed", None)
                if parsed is not None:
                    logger.info(
                        "LLM structured success model=%s duration=%.3fs%s",
                        model_name,
                        duration,
                        _format_usage_for_log(usage_info),
                    )
                    return parsed

                refusal = self._extract_refusal(response)
                if refusal:
                    raise ValueError(f"Model refused: {refusal}")

                if getattr(response, "error", None) is not None:
                    self._log_response_structure_error(response, usage_info)
                    raise ValueError(f"Responses API error: {response.error}")

                if getattr(response, "incomplete_details", None) is not None:
                    self._log_response_structure_error(response, usage_info)
                    raise ValueError(f"Responses API incomplete: {response.incomplete_details}")

                self._log_response_structure_error(response, usage_info)
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
                    **self._build_response_kwargs(messages, response_model, temperature)
                )
                duration = time.perf_counter() - start
                usage_info = self._record_usage_from_response(response)

                model_name = getattr(response, "model", self.model)
                parsed = getattr(response, "output_parsed", None)
                if parsed is not None:
                    logger.info(
                        "LLM structured async success model=%s duration=%.3fs%s",
                        model_name,
                        duration,
                        _format_usage_for_log(usage_info),
                    )
                    return parsed

                refusal = self._extract_refusal(response)
                if refusal:
                    raise ValueError(f"Model refused: {refusal}")

                if getattr(response, "error", None) is not None:
                    self._log_response_structure_error(response, usage_info)
                    raise ValueError(f"Responses API error: {response.error}")

                if getattr(response, "incomplete_details", None) is not None:
                    self._log_response_structure_error(response, usage_info)
                    raise ValueError(f"Responses API incomplete: {response.incomplete_details}")

                self._log_response_structure_error(response, usage_info)
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
                usage_info = self._record_usage_from_embedding(response)

                logger.info(
                    "Embedding success model=%s duration=%.3fs%s",
                    getattr(response, "model", self.embedding_model),
                    duration,
                    _format_usage_for_log(usage_info),
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
                usage_info = self._record_usage_from_embedding(response)

                logger.info(
                    "Embedding async success model=%s duration=%.3fs%s",
                    getattr(response, "model", self.embedding_model),
                    duration,
                    _format_usage_for_log(usage_info),
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


def get_usage_summary() -> dict:
    """Return aggregated LLM usage summary for the current process."""
    return USAGE_TRACKER.summary()


def format_usage_summary(summary: dict) -> list[str]:
    """Format usage summary into CLI-friendly lines."""
    total_requests = summary.get("total_requests", 0)
    if total_requests == 0:
        return []

    lines = [f"total_requests={total_requests}"]

    tokens_line = (
        "tokens"
        f" input={summary.get('total_input_tokens', 0)}"
        f" output={summary.get('total_output_tokens', 0)}"
        f" total={summary.get('total_tokens', 0)}"
    )
    if summary.get("total_reasoning_tokens", 0):
        tokens_line += f" reasoning={summary.get('total_reasoning_tokens', 0)}"
    if summary.get("total_cached_tokens", 0):
        tokens_line += f" cached={summary.get('total_cached_tokens', 0)}"
    lines.append(tokens_line)

    total_cost = summary.get("total_cost_usd", 0.0)
    pricing_missing_models = summary.get("pricing_missing_models", [])
    if total_cost or pricing_missing_models:
        lines.append(f"estimated_cost_usd=${total_cost:.6f}")
        if pricing_missing_models:
            missing_keys = []
            for model in pricing_missing_models:
                missing_keys.extend(_pricing_env_keys(model))
            lines.append(f"pricing_missing_env={', '.join(missing_keys)}")
    else:
        lines.append("estimated_cost_usd=unavailable (set OPENAI_PRICE_<MODEL>_INPUT_PER_1M/OUTPUT_PER_1M)")

    by_model = summary.get("by_model", {})
    for model in sorted(by_model):
        stats = by_model[model]
        model_line = (
            f"model={model} requests={stats.get('requests', 0)}"
            f" input={stats.get('input_tokens', 0)}"
            f" output={stats.get('output_tokens', 0)}"
            f" total={stats.get('total_tokens', 0)}"
        )
        if stats.get("reasoning_tokens", 0):
            model_line += f" reasoning={stats.get('reasoning_tokens', 0)}"
        if stats.get("cached_tokens", 0):
            model_line += f" cached={stats.get('cached_tokens', 0)}"
        if stats.get("cost_usd", 0.0):
            model_line += f" cost=${stats.get('cost_usd', 0.0):.6f}"
        lines.append(model_line)

    return lines


def get_client(api_key: Optional[str] = None) -> LLMClient:
    """
    Get a configured LLM client instance.

    Args:
        api_key: Optional API key override.

    Returns:
        Configured LLMClient instance.
    """
    return LLMClient(api_key=api_key)
