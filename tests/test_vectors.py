"""Block 10 unit tests: chunking, cosine_similarity, VectorIndex."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from agentkit.chunking import fixed_length_chunking, sentence_aware_chunking
from agentkit.embeddings import cosine_similarity
from agentkit.retrieval import VectorIndex

# ── Fake deterministic provider ───────────────────────────────────────────────


class _FakeProvider:
    """Deterministic embeddings for tests: hash(text) → reproducible unit vector."""

    name = "fake"
    dimension = 4

    def embed(self, texts: list[str]) -> np.ndarray:
        result = np.zeros((len(texts), 4), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = int(hashlib.md5(t.encode()).hexdigest()[:8], 16) % (2**31)
            rng = np.random.default_rng(seed)
            v = rng.random(4).astype(np.float32)
            norm = np.linalg.norm(v)
            result[i] = v / max(norm, 1e-10)
        return result


# ── fixed_length_chunking ─────────────────────────────────────────────────────


def test_fixed_empty_text() -> None:
    assert fixed_length_chunking("") == []


def test_fixed_short_text_single_chunk() -> None:
    chunks = fixed_length_chunking("hello", chunk_size=100, overlap=10)
    assert chunks == ["hello"]


def test_fixed_chunk_count_and_max_length() -> None:
    text = "a" * 1000
    chunks = fixed_length_chunking(text, chunk_size=100, overlap=10)
    assert all(len(c) <= 100 for c in chunks)
    assert len(chunks) > 1


def test_fixed_overlap_present() -> None:
    text = "abcdefghij" * 20  # 200 chars
    chunks = fixed_length_chunking(text, chunk_size=50, overlap=10)
    assert len(chunks) >= 2
    # Overlap: end of chunk[0] equals start of chunk[1]
    assert chunks[0][-10:] == chunks[1][:10]


def test_fixed_covers_full_text() -> None:
    text = "hello world " * 50
    chunks = fixed_length_chunking(text, chunk_size=100, overlap=20)
    # First chunk starts with beginning; last chunk ends with end of text
    assert text.startswith(chunks[0])
    assert text.rstrip().endswith(chunks[-1].rstrip())


def test_fixed_overlap_equal_chunk_size_raises() -> None:
    with pytest.raises(ValueError, match="overlap"):
        fixed_length_chunking("text", chunk_size=100, overlap=100)


def test_fixed_overlap_greater_than_chunk_size_raises() -> None:
    with pytest.raises(ValueError, match="overlap"):
        fixed_length_chunking("text", chunk_size=50, overlap=200)


# ── sentence_aware_chunking ───────────────────────────────────────────────────

_SENTENCES = [
    "Alice went to the market.",
    "Bob stayed at home.",
    "Charlie visited the park.",
    "David cooked dinner.",
    "Eve read a book.",
    "Frank played guitar.",
    "Grace studied math.",
    "Henry walked the dog.",
]
_SENTENCE_TEXT = " ".join(_SENTENCES)


def test_sentence_aware_empty() -> None:
    assert sentence_aware_chunking("") == []


def test_sentence_aware_overlap_gte_chunk_size_raises() -> None:
    with pytest.raises(ValueError, match="overlap"):
        sentence_aware_chunking("text", chunk_size=50, overlap=50)


def test_sentence_aware_no_mid_sentence_cut() -> None:
    """No chunk (except possibly the last) should cut in the middle of a sentence."""
    chunks = sentence_aware_chunking(_SENTENCE_TEXT, chunk_size=80, overlap=20)
    assert len(chunks) > 1
    for chunk in chunks[:-1]:
        # Must end at a sentence boundary
        assert chunk.rstrip()[-1] in ".!?", (
            f"Chunk ends mid-sentence: {chunk!r}"
        )


def test_sentence_aware_covers_all_content() -> None:
    chunks = sentence_aware_chunking(_SENTENCE_TEXT, chunk_size=80, overlap=20)
    combined = " ".join(chunks)
    # Every sentence should appear somewhere in the chunks
    for sent in _SENTENCES:
        assert sent in combined, f"Missing sentence: {sent!r}"


def test_sentence_aware_single_long_sentence_falls_back() -> None:
    """A sentence longer than chunk_size is split by character fallback."""
    long_sent = "word " * 200  # ~1000 chars, no sentence-ending punctuation
    chunks = sentence_aware_chunking(long_sent, chunk_size=100, overlap=10)
    assert all(len(c) <= 100 for c in chunks)
    assert len(chunks) > 1


# ── cosine_similarity ─────────────────────────────────────────────────────────


def test_cosine_identical_vectors() -> None:
    v = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    result = cosine_similarity(v, v)
    assert pytest.approx(result[0, 0], abs=1e-5) == 1.0


def test_cosine_orthogonal_vectors() -> None:
    a = np.array([[1.0, 0.0]], dtype=np.float32)
    b = np.array([[0.0, 1.0]], dtype=np.float32)
    result = cosine_similarity(a, b)
    assert pytest.approx(result[0, 0], abs=1e-5) == 0.0


def test_cosine_zero_vector_no_nan() -> None:
    zero = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    other = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    result = cosine_similarity(zero, other)
    assert not np.isnan(result).any()
    assert pytest.approx(result[0, 0], abs=1e-5) == 0.0


def test_cosine_batch_shape() -> None:
    a = np.random.rand(3, 5).astype(np.float32)
    b = np.random.rand(4, 5).astype(np.float32)
    result = cosine_similarity(a, b)
    assert result.shape == (3, 4)


def test_cosine_symmetry() -> None:
    a = np.random.rand(2, 4).astype(np.float32)
    b = np.random.rand(2, 4).astype(np.float32)
    assert np.allclose(cosine_similarity(a, b), cosine_similarity(b, a).T, atol=1e-5)


# ── VectorIndex ───────────────────────────────────────────────────────────────


@pytest.fixture()
def index() -> VectorIndex:
    return VectorIndex(_FakeProvider())


def test_index_empty_search(index: VectorIndex) -> None:
    results = index.search("query", top_k=3)
    assert results == []


def test_index_add_and_search_returns_top_k(index: VectorIndex) -> None:
    texts = ["apple", "orange", "car", "truck", "bicycle"]
    index.add(texts)
    results = index.search("apple", top_k=2)
    assert len(results) == 2


def test_index_search_result_fields(index: VectorIndex) -> None:
    index.add(["apple", "orange"], metadata=[{"id": 1}, {"id": 2}])
    results = index.search("apple", top_k=1)
    assert len(results) == 1
    r = results[0]
    assert "text" in r
    assert "score" in r
    assert "metadata" in r
    assert isinstance(r["score"], float)
    assert r["metadata"]["id"] in (1, 2)


def test_index_search_order_is_descending(index: VectorIndex) -> None:
    index.add(["alpha", "beta", "gamma", "delta"])
    results = index.search("alpha", top_k=4)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_index_top_k_capped_at_index_size(index: VectorIndex) -> None:
    index.add(["x", "y"])
    results = index.search("x", top_k=10)
    assert len(results) == 2


def test_index_metadata_preserved(index: VectorIndex) -> None:
    meta = [{"source": "wiki", "page": i} for i in range(3)]
    index.add(["a", "b", "c"], metadata=meta)
    results = index.search("a", top_k=3)
    returned_sources = {r["metadata"]["source"] for r in results}
    assert returned_sources == {"wiki"}


def test_index_add_metadata_length_mismatch_raises(index: VectorIndex) -> None:
    with pytest.raises(ValueError, match="len"):
        index.add(["a", "b"], metadata=[{"x": 1}])


def test_index_save_load_roundtrip(index: VectorIndex, tmp_path: Path) -> None:
    texts = ["dog", "cat", "fish"]
    meta = [{"i": i} for i in range(3)]
    index.add(texts, metadata=meta)

    save_path = tmp_path / "index"
    index.save(save_path)

    index2 = VectorIndex(_FakeProvider())
    index2.load(str(save_path) + ".npz")

    assert index2._texts == texts
    assert index2._metadata == meta
    assert index2._embeddings is not None
    assert index2._embeddings.shape == index._embeddings.shape
    # Search should work after load
    results = index2.search("dog", top_k=1)
    assert len(results) == 1


def test_index_save_empty_raises(index: VectorIndex, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty"):
        index.save(tmp_path / "empty")


# ── live tests ────────────────────────────────────────────────────────────────


@pytest.mark.live
def test_voyage_embed_dimension() -> None:
    from agentkit.embeddings import VoyageEmbeddings

    provider = VoyageEmbeddings()
    embs = provider.embed(["hello world"])
    assert embs.shape[0] == 1
    assert embs.shape[1] > 0
    assert embs.shape[1] == provider.dimension


@pytest.mark.live
def test_voyage_cat_kitten_dog_ordering() -> None:
    """sim(cat, kitten) > sim(cat, dog) — semantic ordering preserved."""
    from agentkit.embeddings import VoyageEmbeddings, cosine_similarity

    provider = VoyageEmbeddings()
    texts = ["cat", "kitten", "dog"]
    embs = provider.embed(texts)
    sim = cosine_similarity(embs, embs)
    assert sim[0, 1] > sim[0, 2], (
        f"Expected sim(cat,kitten)={sim[0,1]:.3f} > sim(cat,dog)={sim[0,2]:.3f}"
    )


@pytest.mark.live
def test_count_tokens_reasonable() -> None:
    from agentkit.tokens import count_tokens

    text = "The quick brown fox jumps over the lazy dog."
    n = count_tokens(text)
    # Known ~10 tokens; allow generous range for different tokenisers
    assert 5 <= n <= 30, f"Unexpected token count: {n}"
