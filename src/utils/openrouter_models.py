"""Fetch available models from OpenRouter API.

Provides dynamic model loading with fallback to stored defaults.
Models are filtered and categorized for preprocessing (cheap/fast)
and generation (quality) tasks.
"""

import os
from typing import List, Tuple, Optional, Dict, Any

import requests
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


# =============================================================================
# FALLBACK MODELS (used if API fetch fails)
# =============================================================================

FALLBACK_PREPROCESSING_MODELS: List[Tuple[str, str]] = [
    ("deepseek/deepseek-chat", "DeepSeek V3 - Cheapest"),
    ("openai/gpt-4o-mini", "GPT-4o Mini - Fast"),
    ("anthropic/claude-3-5-haiku-latest", "Claude 3.5 Haiku - Quality"),
]

FALLBACK_GENERATION_MODELS: List[Tuple[str, str]] = [
    ("deepseek/deepseek-chat", "DeepSeek V3 - Value"),
    ("openai/gpt-4o-mini", "GPT-4o Mini - Balanced"),
    ("google/gemini-2.0-flash-001", "Gemini 2.0 Flash - Fast"),
    ("anthropic/claude-3-5-haiku-latest", "Claude 3.5 Haiku - Premium"),
]

# Models we're interested in (by provider/prefix)
PREFERRED_PROVIDERS = [
    "deepseek/",
    "openai/",
    "anthropic/",
    "google/",
    "mistralai/",
]

# Models suitable for cheap/fast classification tasks
PREPROCESSING_MODEL_PATTERNS = [
    "deepseek/deepseek-chat",
    "deepseek/deepseek-v3",
    "openai/gpt-4o-mini",
    "openai/gpt-4.1-mini",
    "openai/gpt-4.1-nano",
    "anthropic/claude-3-5-haiku",
    "anthropic/claude-3-haiku",
    "google/gemini-2.0-flash",
    "google/gemini-flash",
    "mistralai/mistral-small",
    "mistralai/ministral",
]

# Models suitable for answer generation (higher quality)
GENERATION_MODEL_PATTERNS = [
    "deepseek/deepseek-chat",
    "deepseek/deepseek-v3",
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "openai/gpt-4.1",
    "anthropic/claude-3-5-haiku",
    "anthropic/claude-3-5-sonnet",
    "google/gemini-2.0-flash",
    "google/gemini-2.5-flash",
    "google/gemini-pro",
]


# =============================================================================
# API FETCHING
# =============================================================================


def fetch_available_models() -> Optional[List[Dict[str, Any]]]:
    """Fetch all available models from OpenRouter API.

    Returns:
        List of model dictionaries with id, name, pricing, etc.
        None if API call fails.
    """
    if not OPENROUTER_API_KEY:
        return None

    try:
        response = requests.get(
            f"{OPENROUTER_BASE_URL}/models",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            timeout=10,
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("data", [])
    except requests.RequestException:
        pass
    return None


def _format_price(price_per_token: float) -> str:
    """Format price per token as price per 1M tokens."""
    if price_per_token == 0:
        return "free"
    price_per_million = price_per_token * 1_000_000
    if price_per_million < 0.01:
        return f"${price_per_million:.4f}"
    return f"${price_per_million:.2f}"


def _model_matches_patterns(model_id: str, patterns: List[str]) -> bool:
    """Check if model ID matches any of the patterns."""
    return any(model_id.startswith(pattern) for pattern in patterns)


def _format_model_label(model: Dict[str, Any]) -> str:
    """Create a human-readable label for a model."""
    name = model.get("name", model["id"])
    pricing = model.get("pricing", {})

    prompt_price = float(pricing.get("prompt", 0))
    completion_price = float(pricing.get("completion", 0))

    prompt_str = _format_price(prompt_price)
    completion_str = _format_price(completion_price)

    return f"{name} ({prompt_str}/{completion_str} per 1M)"


def _filter_models(
    models: List[Dict[str, Any]],
    patterns: List[str],
    max_prompt_price: float = 0.000005,  # $5 per 1M tokens
) -> List[Tuple[str, str]]:
    """Filter models by patterns and price, return as (id, label) tuples."""
    result = []

    for model in models:
        model_id = model.get("id", "")

        # Check if model matches our patterns
        if not _model_matches_patterns(model_id, patterns):
            continue

        # Check price limit
        pricing = model.get("pricing", {})
        prompt_price = float(pricing.get("prompt", 0))
        if prompt_price > max_prompt_price:
            continue

        label = _format_model_label(model)
        result.append((model_id, label))

    # Sort by prompt price (cheapest first)
    result.sort(key=lambda x: float(
        next((m.get("pricing", {}).get("prompt", 999)
              for m in models if m.get("id") == x[0]), 999)
    ))

    return result


# =============================================================================
# PUBLIC API
# =============================================================================


def get_preprocessing_models(
    cached_models: Optional[List[Dict[str, Any]]] = None
) -> List[Tuple[str, str]]:
    """Get models suitable for preprocessing (classification, step-back).

    These should be cheap and fast since they're used for simple tasks.

    Args:
        cached_models: Optional pre-fetched models to avoid repeat API calls.

    Returns:
        List of (model_id, label) tuples.
    """
    models = cached_models or fetch_available_models()

    if not models:
        return FALLBACK_PREPROCESSING_MODELS

    filtered = _filter_models(
        models,
        PREPROCESSING_MODEL_PATTERNS,
        max_prompt_price=0.000002,  # Max $2 per 1M for preprocessing
    )

    return filtered if filtered else FALLBACK_PREPROCESSING_MODELS


def get_generation_models(
    cached_models: Optional[List[Dict[str, Any]]] = None
) -> List[Tuple[str, str]]:
    """Get models suitable for answer generation.

    These can be slightly more expensive for better quality.

    Args:
        cached_models: Optional pre-fetched models to avoid repeat API calls.

    Returns:
        List of (model_id, label) tuples.
    """
    models = cached_models or fetch_available_models()

    if not models:
        return FALLBACK_GENERATION_MODELS

    filtered = _filter_models(
        models,
        GENERATION_MODEL_PATTERNS,
        max_prompt_price=0.000010,  # Max $10 per 1M for generation
    )

    return filtered if filtered else FALLBACK_GENERATION_MODELS
