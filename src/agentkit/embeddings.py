"""Embedding providers and cosine similarity (block 10).

Providers
---------
VoyageEmbeddings  — Voyage AI API (requires VOYAGE_API_KEY)
LocalEmbeddings   — fastembed / BAAI/bge-small-en-v1.5 (lazy, no GPU needed)

get_embedding_provider(name) selects via EMBEDDING_PROVIDER env var.
Default is "voyage".  Set EMBEDDING_PROVIDER=local to use fastembed.
"""
from __future__ import annotations

import logging
import os
import time
from typing import ClassVar, Protocol, runtime_checkable

import numpy as np

from agentkit.config import require_env

logger = logging.getLogger(__name__)

_VOYAGE_BATCH_SIZE = 128
_VOYAGE_MAX_RETRIES = 3


# ── Protocol ──────────────────────────────────────────────────────────────────


@runtime_checkable
class EmbeddingProvider(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed(self, texts: list[str]) -> np.ndarray: ...


# ── Cosine similarity ─────────────────────────────────────────────────────────


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between rows of a (n, d) and b (m, d) → (n, m).

    Zero vectors are handled safely via a small epsilon floor — they produce
    a similarity of 0 with every other vector rather than NaN.
    """
    a_n = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-10)
    b_n = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-10)
    return a_n @ b_n.T


# ── Voyage AI ─────────────────────────────────────────────────────────────────


class VoyageEmbeddings:
    """Voyage AI embedding provider.

    Requires VOYAGE_API_KEY in the environment (or .env file).
    Batches requests to stay within the 128-text-per-call API limit.
    Retries up to _VOYAGE_MAX_RETRIES times with exponential backoff.
    """

    def __init__(self, model: str | None = None) -> None:
        self._model = model or os.getenv("VOYAGE_EMBEDDING_MODEL", "voyage-4-lite")
        self._api_key = require_env("VOYAGE_API_KEY")
        self._client: object | None = None
        self._dimension: int | None = None

    def _get_client(self) -> object:
        if self._client is None:
            try:
                import voyageai
            except ImportError as exc:
                raise ImportError("voyageai is not installed. Run: uv add voyageai") from exc
            self._client = voyageai.Client(api_key=self._api_key)
        return self._client

    @property
    def name(self) -> str:
        return f"voyage:{self._model}"

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = self.embed(["probe"]).shape[1]
        return self._dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        client = self._get_client()
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), _VOYAGE_BATCH_SIZE):
            batch = texts[i : i + _VOYAGE_BATCH_SIZE]
            for attempt in range(_VOYAGE_MAX_RETRIES):
                try:
                    result = client.embed(batch, model=self._model)  # type: ignore[attr-defined]
                    all_embeddings.extend(result.embeddings)
                    break
                except Exception as exc:
                    if attempt == _VOYAGE_MAX_RETRIES - 1:
                        raise
                    wait = 2**attempt
                    logger.warning(
                        "Voyage API error (attempt %d/%d): %s — retrying in %ds",
                        attempt + 1, _VOYAGE_MAX_RETRIES, exc, wait,
                    )
                    time.sleep(wait)

        arr = np.array(all_embeddings, dtype=np.float32)
        if self._dimension is None:
            self._dimension = arr.shape[1]
        return arr


# ── Local (fastembed) ─────────────────────────────────────────────────────────


class LocalEmbeddings:
    """Local embedding provider using fastembed.

    Model: BAAI/bge-small-en-v1.5 (384-dim, ~130 MB download, ~33 MB RAM).
    The model is loaded lazily on the first embed() call — importing this
    module does NOT trigger the download.

    Requires the ``local`` dependency group:
        uv sync --group local
    """

    _KNOWN_DIMS: ClassVar[dict[str, int]] = {
        "BAAI/bge-small-en-v1.5": 384,
        "BAAI/bge-base-en-v1.5": 768,
    }

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self._model_name = model_name
        self._model: object | None = None

    def _get_model(self) -> object:
        if self._model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:
                raise ImportError(
                    "fastembed is required for LocalEmbeddings. "
                    "Install with: uv sync --group local"
                ) from exc
            self._model = TextEmbedding(model_name=self._model_name)
        return self._model

    @property
    def name(self) -> str:
        return f"local:{self._model_name}"

    @property
    def dimension(self) -> int:
        return self._KNOWN_DIMS.get(self._model_name, 384)

    def embed(self, texts: list[str]) -> np.ndarray:
        model = self._get_model()
        return np.array(list(model.embed(texts)), dtype=np.float32)  # type: ignore[attr-defined]


# ── Factory ───────────────────────────────────────────────────────────────────


def get_embedding_provider(name: str | None = None) -> EmbeddingProvider:
    """Return an EmbeddingProvider selected by *name* or EMBEDDING_PROVIDER env var.

    Supported values: ``"voyage"`` (default), ``"local"``.
    """
    provider = name or os.getenv("EMBEDDING_PROVIDER", "voyage")
    if provider == "voyage":
        return VoyageEmbeddings()
    if provider == "local":
        return LocalEmbeddings()
    raise ValueError(
        f"Unknown embedding provider: {provider!r}. Valid values: 'voyage', 'local'."
    )
