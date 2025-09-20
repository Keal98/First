"""Helpers for working with the Google Generative AI client."""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List

import google.generativeai as genai

from .config import EMBED_MODEL, LLM_MODEL

_CONFIGURED = False


def ensure_genai_configured() -> None:
    """Validate and configure the Google Generative AI client."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set")
    genai.configure(api_key=api_key)
    _CONFIGURED = True


def _extract_embeddings(payload: Dict[str, Any]) -> List[List[float]]:
    """Normalise embedding payloads from the Gemini API."""
    embeddings = payload.get("embedding")
    if isinstance(embeddings, dict) and "values" in embeddings:
        return [embeddings["values"]]
    if isinstance(embeddings, list):
        if not embeddings:
            return []
        first = embeddings[0]
        if isinstance(first, dict) and "values" in first:
            return [item["values"] for item in embeddings]
        if isinstance(first, list):
            return embeddings
    return []


def embed_batch(texts: Iterable[str]) -> List[List[float]]:
    """Return embeddings for a batch of texts, falling back to per-item calls."""
    ensure_genai_configured()
    items = list(texts)
    if not items:
        return []
    try:
        response = genai.embed_content(
            model=EMBED_MODEL,
            content=items,
            task_type="retrieval_document",
        )
        embeddings = _extract_embeddings(response)
        if embeddings:
            return embeddings
    except Exception:
        # Fall back to per-item requests
        pass

    results: List[List[float]] = []
    for text in items:
        response = genai.embed_content(
            model=EMBED_MODEL,
            content=text,
            task_type="retrieval_document",
        )
        extracted = _extract_embeddings(response)
        if not extracted:
            raise RuntimeError("Failed to extract embedding values from response")
        results.append(extracted[0])
    return results


def make_generative_model():
    """Return a configured GenerativeModel instance."""
    ensure_genai_configured()
    return genai.GenerativeModel(LLM_MODEL)
