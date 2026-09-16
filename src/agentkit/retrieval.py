"""In-memory vector index (block 10).

VectorIndex stores texts, metadata, and their embeddings as a numpy matrix.
Search is O(n) brute-force cosine similarity — practical up to ~50k chunks
on a modern CPU (single-threaded, ~100 ms for 10k × 1024-dim vectors).
Beyond that, switch to an ANN library: faiss, usearch, or hnswlib all drop
latency to O(log n) at the cost of some recall.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from agentkit.embeddings import EmbeddingProvider, cosine_similarity

logger = logging.getLogger(__name__)


class VectorIndex:
    """In-memory vector index: embed, store, search, persist.

    Parameters
    ----------
    provider:
        Any EmbeddingProvider (VoyageEmbeddings, LocalEmbeddings, or a mock).
    """

    def __init__(self, provider: EmbeddingProvider) -> None:
        self._provider = provider
        self._texts: list[str] = []
        self._metadata: list[dict[str, Any]] = []
        self._embeddings: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self._texts)

    # ── Mutation ──────────────────────────────────────────────────────────────

    def add(
        self,
        texts: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """Embed *texts* and append them to the index.

        Parameters
        ----------
        texts:
            Non-empty list of strings to index.
        metadata:
            Optional per-text metadata dicts.  Defaults to empty dicts.
        """
        if not texts:
            return
        meta = metadata if metadata is not None else [{} for _ in texts]
        if len(meta) != len(texts):
            raise ValueError(
                f"len(metadata)={len(meta)} must equal len(texts)={len(texts)}"
            )

        new_embs = self._provider.embed(texts)
        self._embeddings = (
            new_embs if self._embeddings is None else np.vstack([self._embeddings, new_embs])
        )
        self._texts.extend(texts)
        self._metadata.extend(meta)
        logger.debug("VectorIndex: added %d texts (total %d)", len(texts), len(self._texts))

    # ── Query ─────────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Return the *top_k* most similar chunks to *query*.

        Returns a list of dicts, each with keys:
            ``text``     — the stored text
            ``score``    — cosine similarity in [-1, 1] (typically [0, 1])
            ``metadata`` — the dict passed at add() time
        """
        if self._embeddings is None or not self._texts:
            return []

        q_emb = self._provider.embed([query])          # (1, d)
        scores = cosine_similarity(q_emb, self._embeddings)[0]  # (n,)

        k = min(top_k, len(scores))
        top_idx = np.argsort(scores)[::-1][:k]

        return [
            {
                "text": self._texts[i],
                "score": float(scores[i]),
                "metadata": self._metadata[i],
            }
            for i in top_idx
        ]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save index to a ``.npz`` file.

        Persisting avoids re-embedding the same corpus on every run.
        The provider is NOT saved — reconstruct it with the same model when
        loading.
        """
        if self._embeddings is None:
            raise ValueError("Cannot save an empty index")
        path = Path(path)
        np.savez(
            path,
            embeddings=self._embeddings,
            texts=np.array(self._texts, dtype=object),
            metadata=np.array([json.dumps(m) for m in self._metadata], dtype=object),
        )
        logger.info("VectorIndex: saved %d vectors → %s", len(self._texts), path)

    def load(self, path: str | Path) -> None:
        """Load index from a ``.npz`` file produced by save()."""
        path = Path(path)
        data = np.load(path, allow_pickle=True)
        self._embeddings = data["embeddings"]
        self._texts = list(data["texts"])
        self._metadata = [json.loads(str(m)) for m in data["metadata"]]
        logger.info("VectorIndex: loaded %d vectors ← %s", len(self._texts), path)
