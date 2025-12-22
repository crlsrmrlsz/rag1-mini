"""Unified OpenRouter API client.

## RAG Theory: Centralized LLM Communication

All LLM calls in a RAG pipeline should go through a single client for:
- Consistent retry logic and error handling
- Centralized rate limit management
- Request/response logging for debugging
- Easy model switching for A/B testing

## Library Usage

Uses `requests` for HTTP calls. The retry pattern implements
exponential backoff, which is essential for handling:
- Rate limits (429 errors) from OpenRouter
- Temporary server errors (5xx)
- Network transients

## Data Flow

1. Module (preprocessing/generation/evaluation) needs LLM
2. Imports call_chat_completion() from here
3. Constructs messages in OpenAI format
4. Receives response string or raises exception
"""

import os
import time
from typing import Dict, List, Optional, Any, Type, TypeVar

from pydantic import BaseModel, ValidationError as PydanticValidationError

T = TypeVar("T", bound=BaseModel)

import requests
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


class OpenRouterError(Exception):
    """Base exception for OpenRouter API errors."""
    pass


class RateLimitError(OpenRouterError):
    """Raised when rate limit is exceeded after all retries."""
    pass


class APIError(OpenRouterError):
    """Raised when API returns an error response."""
    pass


def call_chat_completion(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float = 0.3,
    max_tokens: int = 1024,
    json_mode: bool = False,
    timeout: int = 60,
    max_retries: int = 3,
    backoff_base: float = 1.5,
) -> str:
    """Call OpenRouter chat completion API with retry logic.

    This is the unified LLM call function used by all modules:
    - preprocessing: Query classification, step-back prompting
    - generation: Answer synthesis
    - evaluation: RAGAS answer generation

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
            Example: [{"role": "user", "content": "Hello"}]
        model: OpenRouter model ID (e.g., "openai/gpt-4o-mini").
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative).
        max_tokens: Maximum tokens in response.
        json_mode: If True, request JSON response format.
        timeout: Request timeout in seconds.
        max_retries: Number of retries on failure.
        backoff_base: Backoff multiplier for retries.

    Returns:
        The assistant's response content as a string.

    Raises:
        OpenRouterError: On API errors after all retries.
        RateLimitError: If rate limited after all retries.

    Example:
        >>> response = call_chat_completion(
        ...     messages=[{"role": "user", "content": "What is 2+2?"}],
        ...     model="openai/gpt-4o-mini",
        ... )
        >>> print(response)
        "4"
    """
    from .files import setup_logging
    logger = setup_logging(__name__)

    if not OPENROUTER_API_KEY:
        raise OpenRouterError("OPENROUTER_API_KEY not set in environment")

    url = f"{OPENROUTER_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                url, json=payload, headers=headers, timeout=timeout
            )

            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]

                # Log successful LLM call
                chars_in = sum(len(m.get("content", "")) for m in messages)
                chars_out = len(content)
                logger.info(f"[LLM] model={model} chars_in={chars_in} chars_out={chars_out}")

                return content

            # Retryable errors: rate limit or server errors
            if response.status_code >= 500 or response.status_code == 429:
                if attempt < max_retries:
                    delay = backoff_base ** (attempt + 1)
                    error_type = "Rate limit" if response.status_code == 429 else "Server error"
                    logger.warning(
                        f"{error_type} ({response.status_code}), "
                        f"retry {attempt + 1}/{max_retries} after {delay:.1f}s"
                    )
                    time.sleep(delay)
                    continue
                else:
                    if response.status_code == 429:
                        raise RateLimitError(f"Rate limited after {max_retries} retries")
                    raise APIError(f"Server error {response.status_code} after {max_retries} retries")

            # Non-retryable client errors
            try:
                error_detail = response.json().get("error", {}).get("message", response.text)
            except Exception:
                error_detail = response.text
            raise APIError(f"API error {response.status_code}: {error_detail}")

        except requests.RequestException as exc:
            if attempt < max_retries:
                delay = backoff_base ** (attempt + 1)
                logger.warning(
                    f"Request failed ({exc}), retry {attempt + 1}/{max_retries} in {delay:.1f}s"
                )
                time.sleep(delay)
                continue
            raise OpenRouterError(f"Request failed after {max_retries} retries: {exc}")

    raise OpenRouterError("Max retries exceeded")


def call_simple_prompt(
    prompt: str,
    model: str,
    temperature: float = 0.3,
    max_tokens: int = 1024,
    **kwargs,
) -> str:
    """Convenience wrapper for single-prompt calls.

    Converts a simple prompt string to the messages format.
    Useful for evaluation and simple LLM tasks.

    Args:
        prompt: The prompt text.
        model: OpenRouter model ID.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens.
        **kwargs: Additional arguments passed to call_chat_completion.

    Returns:
        The model's response text.

    Example:
        >>> response = call_simple_prompt(
        ...     "Summarize: The quick brown fox...",
        ...     model="openai/gpt-4o-mini",
        ... )
    """
    messages = [{"role": "user", "content": prompt}]
    return call_chat_completion(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )


def call_structured_completion(
    messages: List[Dict[str, str]],
    model: str,
    response_model: Type[T],
    temperature: float = 0.0,
    max_tokens: int = 1024,
    timeout: int = 60,
    max_retries: int = 3,
    backoff_base: float = 1.5,
) -> T:
    """Call OpenRouter with JSON Schema enforcement and Pydantic validation.

    This function provides structured outputs by:
    1. Converting Pydantic model to JSON Schema
    2. Sending schema to OpenRouter for enforcement (strict mode)
    3. Parsing and validating response with Pydantic

    Uses JSON Schema mode for guaranteed valid responses. Falls back to
    json_object mode with warning if schema mode is unsupported by the model.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
        model: OpenRouter model ID (e.g., "openai/gpt-4o-mini").
        response_model: Pydantic BaseModel class defining expected response.
        temperature: Sampling temperature (0.0 recommended for structured).
        max_tokens: Maximum tokens in response.
        timeout: Request timeout in seconds.
        max_retries: Number of retries on failure.
        backoff_base: Backoff multiplier for retries.

    Returns:
        Validated Pydantic model instance of type response_model.

    Raises:
        OpenRouterError: On API errors after all retries.
        RateLimitError: If rate limited after all retries.
        PydanticValidationError: If response fails Pydantic validation.

    Example:
        >>> from pydantic import BaseModel
        >>> class Result(BaseModel):
        ...     answer: str
        >>> result = call_structured_completion(
        ...     messages=[{"role": "user", "content": "Say hello"}],
        ...     model="openai/gpt-4o-mini",
        ...     response_model=Result,
        ... )
        >>> result.answer
        "Hello!"
    """
    from .files import setup_logging
    from .schemas import get_openrouter_schema

    logger = setup_logging(__name__)

    if not OPENROUTER_API_KEY:
        raise OpenRouterError("OPENROUTER_API_KEY not set in environment")

    url = f"{OPENROUTER_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    # Build payload with JSON Schema enforcement
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": get_openrouter_schema(response_model),
    }

    use_fallback = False

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                url, json=payload, headers=headers, timeout=timeout
            )

            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]

                # Log successful LLM call
                chars_in = sum(len(m.get("content", "")) for m in messages)
                chars_out = len(content)
                logger.info(f"[LLM] model={model} chars_in={chars_in} chars_out={chars_out} (structured)")

                # Parse and validate with Pydantic
                try:
                    return response_model.model_validate_json(content)
                except PydanticValidationError as e:
                    if not use_fallback:
                        # Schema mode may have failed, try json_object fallback
                        logger.warning(
                            f"Pydantic validation failed, trying json_object fallback: {e}"
                        )
                        payload["response_format"] = {"type": "json_object"}
                        use_fallback = True
                        continue
                    # Already in fallback mode, re-raise
                    raise

            # Check for schema mode unsupported error (400 with specific message)
            if response.status_code == 400 and not use_fallback:
                try:
                    error_msg = response.json().get("error", {}).get("message", "")
                    if "response_format" in error_msg.lower() or "schema" in error_msg.lower():
                        logger.warning(
                            f"JSON Schema mode not supported by {model}, falling back to json_object"
                        )
                        payload["response_format"] = {"type": "json_object"}
                        use_fallback = True
                        continue
                except Exception:
                    pass

            # Retryable errors: rate limit or server errors
            if response.status_code >= 500 or response.status_code == 429:
                if attempt < max_retries:
                    delay = backoff_base ** (attempt + 1)
                    error_type = "Rate limit" if response.status_code == 429 else "Server error"
                    logger.warning(
                        f"{error_type} ({response.status_code}), "
                        f"retry {attempt + 1}/{max_retries} after {delay:.1f}s"
                    )
                    time.sleep(delay)
                    continue
                else:
                    if response.status_code == 429:
                        raise RateLimitError(f"Rate limited after {max_retries} retries")
                    raise APIError(f"Server error {response.status_code} after {max_retries} retries")

            # Non-retryable client errors
            try:
                error_detail = response.json().get("error", {}).get("message", response.text)
            except Exception:
                error_detail = response.text
            raise APIError(f"API error {response.status_code}: {error_detail}")

        except requests.RequestException as exc:
            if attempt < max_retries:
                delay = backoff_base ** (attempt + 1)
                logger.warning(
                    f"Request failed ({exc}), retry {attempt + 1}/{max_retries} in {delay:.1f}s"
                )
                time.sleep(delay)
                continue
            raise OpenRouterError(f"Request failed after {max_retries} retries: {exc}")

    raise OpenRouterError("Max retries exceeded")
