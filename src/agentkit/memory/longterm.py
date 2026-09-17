"""Long-term memory backed by ChromaDB (block 14).

Design
------
Memory items are stored as ChromaDB documents with the project's own
EmbeddingProvider (fastembed local or Voyage API). ChromaDB never loads
a second embedding model — the project's provider is always used via
_ChromaEmbeddingFn wrapper.

Deduplication
-------------
Before every add(), the collection is queried for the nearest existing
record belonging to the same user.  If cosine_similarity >= dedup_threshold
(default 0.9) the *existing* record is updated (text, updated_at, source)
rather than a new document created.  This prevents near-identical facts
("I use Python" / "My language is Python") from accumulating as separate
entries.

Contradiction handling
----------------------
When two facts concern the same topic but differ (e.g. city of residence),
their cosine similarity is typically 0.7–0.8 — below the dedup threshold,
so both are stored.  search() sorts results by score DESC, then updated_at
DESC, so the *newest* fact always appears first.  Callers that care about
contradictions can look at the full ranked list.

Score convention
----------------
ChromaDB returns cosine *distance* (0 = identical, 2 = opposite for
non-unit vectors; in practice 0–1 for normalised embeddings).
We convert to similarity score: score = 1.0 - distance.
min_score=0.5 means "at least moderately relevant".

RSS overhead
------------
Importing chromadb adds +52 MB RSS; PersistentClient adds +10 MB more.
Total delta from baseline: ~62 MB.  On the reference 180 MB service this
brings the process to ~242 MB.  Acceptable for a persistent daemon;
consider lazy import (only at first LongTermMemory instantiation) if the
CLI subcommand must stay lightweight.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_MEMORY_PATH = os.path.join(os.path.expanduser("~"), ".agentkit", "memory")
_COLLECTION_NAME = "memories"


# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class Memory:
    """One stored fact about a user."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = "default"
    text: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    source_session_id: str | None = None
    access_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    # Populated by search() — transient, not persisted
    score: float = 0.0


# ── ChromaDB embedding function wrapper ───────────────────────────────────────


def _make_chroma_embedding_fn(provider: Any) -> Any:
    """Build a ChromaDB-compatible EmbeddingFunction wrapping *provider*.

    ChromaDB 1.x requires proper subclassing of EmbeddingFunction[Documents]
    to get the default embed_query() implementation and pass internal
    validation.  We build the class at call time to avoid importing chromadb
    at module level (keeps CLI startup fast when memory is not used).
    """
    from chromadb import EmbeddingFunction
    from chromadb.api.types import Documents, Embeddings

    provider_name = f"agentkit:{getattr(provider, 'name', 'unknown')}"

    class _Fn(EmbeddingFunction[Documents]):  # type: ignore[type-arg]
        def __init__(self) -> None:  # noqa: D107
            # Do NOT call super().__init__() — it emits a DeprecationWarning
            # when no-args __init__ is not defined on the base class.
            self._p = provider

        def name(self) -> str:  # type: ignore[override]
            return provider_name

        def __call__(self, input: Documents) -> Embeddings:  # noqa: A002
            arr = self._p.embed(list(input))
            return arr.tolist()

        @classmethod
        def build_from_config(cls, config: dict[str, Any]) -> "_Fn":  # type: ignore[override]
            raise NotImplementedError("Cannot restore _ChromaEmbeddingFn from config")

        def get_config(self) -> dict[str, Any]:
            return {}

    return _Fn()


# ── LongTermMemory ────────────────────────────────────────────────────────────


class LongTermMemory:
    """Persistent vector memory store for long-running agents.

    Parameters
    ----------
    path:
        Directory for the ChromaDB persistent store.
        Defaults to MEMORY_DB_PATH env var or ~/.agentkit/memory.
    embedding_provider:
        EmbeddingProvider to use.  Defaults to get_embedding_provider()
        (respects EMBEDDING_PROVIDER env var).  Always pass explicitly
        in tests to avoid network calls.
    dedup_threshold:
        Cosine similarity threshold for deduplication.  If the nearest
        existing record for the same user has score >= threshold, the
        existing record is updated instead of a new one created.
    """

    def __init__(
        self,
        path: str | None = None,
        embedding_provider: Any | None = None,
        dedup_threshold: float = 0.9,
    ) -> None:
        import chromadb

        self.path = path or os.getenv("MEMORY_DB_PATH") or _DEFAULT_MEMORY_PATH
        self.dedup_threshold = dedup_threshold

        if embedding_provider is None:
            from agentkit.embeddings import get_embedding_provider
            embedding_provider = get_embedding_provider()
        self._provider = embedding_provider
        self._emb_fn = _make_chroma_embedding_fn(self._provider)

        os.makedirs(self.path, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self.path)
        self._col = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            embedding_function=self._emb_fn,  # type: ignore[arg-type]
            metadata={"hnsw:space": "cosine"},
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _meta_to_memory(self, id_: str, doc: str, meta: dict[str, Any], score: float = 0.0) -> Memory:
        extra = {k: v for k, v in meta.items() if k not in {
            "user_id", "created_at", "updated_at", "source_session_id",
            "access_count", "_metadata_json",
        }}
        extra.update(json.loads(meta.get("_metadata_json", "{}")))
        return Memory(
            id=id_,
            user_id=meta.get("user_id", "default"),
            text=doc,
            created_at=float(meta.get("created_at", 0.0)),
            updated_at=float(meta.get("updated_at", 0.0)),
            source_session_id=meta.get("source_session_id") or None,
            access_count=int(meta.get("access_count", 0)),
            metadata=extra,
            score=score,
        )

    def _memory_to_meta(self, m: Memory) -> dict[str, Any]:
        return {
            "user_id": m.user_id,
            "created_at": m.created_at,
            "updated_at": m.updated_at,
            "source_session_id": m.source_session_id or "",
            "access_count": m.access_count,
            # Extra metadata serialised as JSON to survive ChromaDB's
            # flat-dict requirement (only str/int/float/bool values allowed)
            "_metadata_json": json.dumps(m.metadata),
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def add(
        self,
        user_id: str,
        text: str,
        source_session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Memory:
        """Store *text* as a new memory for *user_id*.

        If a very similar memory (score >= dedup_threshold) already exists
        for the same user, the existing record is updated instead.
        """
        text = text.strip()
        if not text:
            raise ValueError("Memory text cannot be empty")

        # Try to find a near-duplicate
        existing = self._nearest_for_user(user_id, text)
        if existing is not None:
            existing_mem, sim = existing
            if sim >= self.dedup_threshold:
                # Update the existing record
                existing_mem.text = text
                existing_mem.updated_at = time.time()
                if source_session_id:
                    existing_mem.source_session_id = source_session_id
                if metadata:
                    existing_mem.metadata.update(metadata)
                self._col.update(
                    ids=[existing_mem.id],
                    documents=[existing_mem.text],
                    metadatas=[self._memory_to_meta(existing_mem)],
                )
                logger.debug(
                    "Memory dedup: updated %s (sim=%.3f >= %.2f)",
                    existing_mem.id, sim, self.dedup_threshold,
                )
                return existing_mem

        # Create new record
        now = time.time()
        m = Memory(
            user_id=user_id,
            text=text,
            created_at=now,
            updated_at=now,
            source_session_id=source_session_id,
            metadata=dict(metadata or {}),
        )
        self._col.add(
            ids=[m.id],
            documents=[m.text],
            metadatas=[self._memory_to_meta(m)],
        )
        logger.debug("Memory added: %s for user %s", m.id, user_id)
        return m

    def search(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        min_score: float = 0.5,
    ) -> list[Memory]:
        """Return up to *top_k* memories relevant to *query* for *user_id*.

        Results are sorted by score DESC, then updated_at DESC — newer
        facts appear first when similarity is tied (handles contradictions).
        Only results with score >= min_score are returned.
        """
        try:
            results = self._col.query(
                query_texts=[query],
                n_results=top_k,
                where={"user_id": user_id},
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:  # noqa: BLE001
            # Collection empty or no docs for this user
            logger.debug("Memory search returned nothing: %s", exc)
            return []

        ids = (results.get("ids") or [[]])[0]
        docs = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        dists = (results.get("distances") or [[]])[0]

        memories: list[Memory] = []
        for id_, doc, meta, dist in zip(ids, docs, metas, dists):
            score = max(0.0, 1.0 - dist)
            if score < min_score:
                continue
            memories.append(self._meta_to_memory(id_, doc, meta, score=score))

        # Primary sort: score DESC; secondary: updated_at DESC
        memories.sort(key=lambda m: (m.score, m.updated_at), reverse=True)
        return memories

    def list_all(self, user_id: str) -> list[Memory]:
        """Return all memories for *user_id*, newest first."""
        try:
            results = self._col.get(
                where={"user_id": user_id},
                include=["documents", "metadatas"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Memory list_all returned nothing: %s", exc)
            return []

        ids = results.get("ids") or []
        docs = results.get("documents") or []
        metas = results.get("metadatas") or []

        memories = [
            self._meta_to_memory(id_, doc, meta)
            for id_, doc, meta in zip(ids, docs, metas)
        ]
        memories.sort(key=lambda m: m.updated_at, reverse=True)
        return memories

    def delete(self, memory_id: str) -> bool:
        """Delete memory by id. Returns True if it existed."""
        try:
            existing = self._col.get(ids=[memory_id])
            if not existing.get("ids"):
                return False
            self._col.delete(ids=[memory_id])
            return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("Memory delete failed: %s", exc)
            return False

    def delete_user(self, user_id: str) -> int:
        """Delete all memories for *user_id*. Returns count deleted."""
        try:
            existing = self._col.get(
                where={"user_id": user_id},
                include=[],
            )
            ids = existing.get("ids") or []
            if ids:
                self._col.delete(ids=ids)
            return len(ids)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Memory delete_user failed: %s", exc)
            return 0

    # ── Internal ──────────────────────────────────────────────────────────────

    def _nearest_for_user(
        self, user_id: str, text: str
    ) -> tuple[Memory, float] | None:
        """Return the nearest existing memory for *user_id* and its similarity."""
        try:
            results = self._col.query(
                query_texts=[text],
                n_results=1,
                where={"user_id": user_id},
                include=["documents", "metadatas", "distances"],
            )
        except Exception:  # noqa: BLE001
            return None

        ids = (results.get("ids") or [[]])[0]
        docs = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        dists = (results.get("distances") or [[]])[0]

        if not ids:
            return None

        sim = max(0.0, 1.0 - dists[0])
        mem = self._meta_to_memory(ids[0], docs[0], metas[0], score=sim)
        return mem, sim
