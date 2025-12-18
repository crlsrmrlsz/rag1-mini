"""OpenRouter embedding client for RAG1-Mini.

Provides API integration for generating text embeddings via OpenRouter.
"""

import time
import requests
from typing import List

from src.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    EMBEDDING_MODEL_ID,
)
from src.utils.file_utils import setup_logging

logger = setup_logging(__name__)

# --------------------------------------------------------------------------------
# OPENROUTER EMBEDDING CLIENT
# --------------------------------------------------------------------------------

def call_openrouter_embeddings_api(
    inputs: List[str],
    max_retries: int = 3,
    backoff_base: float = 1.5
) -> List[List[float]]:
    """
    Calls the OpenRouter embeddings API.

    Args:
        inputs: List of text strings.
        max_retries: How many retries on failure (HTTP/network).
        backoff_base: Backoff multiplier for retry delays.

    Returns:
        A list of embedding vectors (one list per input).
    """
    url = f"{OPENROUTER_BASE_URL}/embeddings"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": EMBEDDING_MODEL_ID,
        "input": inputs,
    }

    attempt = 0
    while True:
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)

            if response.status_code == 200:
                result = response.json()
                # The API returns "data" as a list of embeddings
                embeddings = [item["embedding"] for item in result.get("data", [])]
                return embeddings

            # Rate limit or server errors
            if response.status_code >= 500 or response.status_code == 429:
                attempt += 1
                if attempt > max_retries:
                    response.raise_for_status()
                delay = backoff_base**attempt
                logger.warning(f"Server error {response.status_code}, retry {attempt} after {delay:.1f}s")
                time.sleep(delay)
                continue

            # Hard failure
            response.raise_for_status()

        except requests.RequestException as exc:
            attempt += 1
            if attempt > max_retries:
                raise
            delay = backoff_base**attempt
            logger.warning(f"Request failed ({exc}), retry {attempt} in {delay:.1f}s")
            time.sleep(delay)
            continue


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Wrapper over call_openrouter_embeddings_api.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of vectors (float lists).
    """
    if not texts:
        return []
    return call_openrouter_embeddings_api(texts)
