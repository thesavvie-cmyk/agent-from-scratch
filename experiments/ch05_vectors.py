"""Chapter 5 vector search experiments (block 10).

Sections
--------
a) Cosine similarity: cat / kitten / dog on both embedding providers
b) Chunking comparison: fixed vs sentence-aware on a real article
c) Document ranking (listing 5.6 equivalent): 4 docs, query "Artificial Intelligence"
d) Nobel Prize RAG pipeline: Tavily → tokens → chunk → embed → search "quantum computing"
e) Tavily advanced (search_depth=advanced, chunks_per_source=3) comparison
f) Provider benchmark: embed 200 chunks, measure wall time and peak memory

Usage
-----
    uv run python experiments/ch05_vectors.py --section a
    uv run python experiments/ch05_vectors.py --section all
    uv run python experiments/ch05_vectors.py --section f --provider local

All sections except a/f/c require VOYAGE_API_KEY.
Section d/e also require TAVILY_API_KEY.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import tracemalloc
from typing import Any

# ── helpers ────────────────────────────────────────────────────────────────────


def _hr(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def _fmt_table(headers: list[str], rows: list[list[Any]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    fmt = "  " + "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print("  " + "  ".join("─" * w for w in widths))
    for row in rows:
        print(fmt.format(*[str(c) for c in row]))


# ── section a: cosine similarity table ────────────────────────────────────────


def section_a(provider_name: str | None = None) -> None:
    _hr("Section A — Cosine similarity: cat / kitten / dog")
    from agentkit.embeddings import cosine_similarity, get_embedding_provider

    texts = ["cat", "kitten", "dog"]
    providers_to_run: list[str] = (
        [provider_name] if provider_name else ["voyage", "local"]
    )

    for pname in providers_to_run:
        print(f"\nProvider: {pname}")
        try:
            provider = get_embedding_provider(pname)
            embs = provider.embed(texts)
            sim = cosine_similarity(embs, embs)
            headers = [""] + texts
            rows = []
            for i, t in enumerate(texts):
                row: list[Any] = [t]
                for j in range(len(texts)):
                    row.append(f"{sim[i, j]:.4f}")
                rows.append(row)
            _fmt_table(headers, rows)
            cat_kit = sim[0, 1]
            cat_dog = sim[0, 2]
            verdict = "✓ sim(cat,kitten) > sim(cat,dog)" if cat_kit > cat_dog else "✗ unexpected ordering"
            print(f"  {verdict}  ({cat_kit:.4f} vs {cat_dog:.4f})")
        except Exception as exc:  # noqa: BLE001
            print(f"  [SKIP] {exc}")


# ── section b: chunking comparison ────────────────────────────────────────────


def section_b() -> None:
    _hr("Section B — Chunking: fixed vs sentence-aware")
    from agentkit.chunking import fixed_length_chunking, sentence_aware_chunking

    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        print("  [SKIP] TAVILY_API_KEY not set")
        return

    try:
        from tavily import TavilyClient  # type: ignore[import]
    except ImportError:
        print("  [SKIP] tavily-python not installed")
        return

    client = TavilyClient(api_key=tavily_key)
    print("  Fetching article about 'vector embeddings in machine learning'...")
    resp = client.search(
        "vector embeddings in machine learning",
        search_depth="basic",
        include_raw_content=True,
        max_results=1,
    )
    raw_content: str = ""
    for r in resp.get("results", []):
        raw_content = r.get("raw_content") or r.get("content") or ""
        if len(raw_content) > 500:
            break

    if not raw_content:
        print("  [SKIP] Could not retrieve raw article content")
        return

    print(f"  Article length: {len(raw_content)} chars")

    for chunk_size, overlap in [(500, 50), (1000, 100)]:
        fixed = fixed_length_chunking(raw_content, chunk_size=chunk_size, overlap=overlap)
        sentence = sentence_aware_chunking(raw_content, chunk_size=chunk_size, overlap=overlap)

        print(f"\n  chunk_size={chunk_size}, overlap={overlap}")
        headers = ["Strategy", "Chunks", "Min len", "Max len", "Avg len"]
        rows = []
        for name, chunks in [("fixed", fixed), ("sentence", sentence)]:
            lengths = [len(c) for c in chunks]
            rows.append([
                name,
                len(chunks),
                min(lengths),
                max(lengths),
                f"{sum(lengths)/len(lengths):.0f}",
            ])
        _fmt_table(headers, rows)

    # Show first 2 chunks from sentence-aware at 500/50
    sa_chunks = sentence_aware_chunking(raw_content, chunk_size=500, overlap=50)
    print("\n  First sentence-aware chunk (500/50):")
    print(f"  {sa_chunks[0][:200]!r}...")
    if len(sa_chunks) > 1:
        print("\n  Second chunk starts with:")
        print(f"  {sa_chunks[1][:100]!r}...")


# ── section c: document ranking ───────────────────────────────────────────────


def section_c(provider_name: str | None = None) -> None:
    _hr("Section C — Document ranking (Listing 5.6)")
    from agentkit.embeddings import get_embedding_provider
    from agentkit.retrieval import VectorIndex

    documents = [
        "Artificial Intelligence is transforming technology.",
        "The weather today is sunny and warm.",
        "Machine learning models require large datasets.",
        "Python is a popular programming language for AI.",
    ]
    query = "Artificial Intelligence"

    pname = provider_name or "voyage"
    try:
        provider = get_embedding_provider(pname)
    except Exception as exc:  # noqa: BLE001
        print(f"  [SKIP] {exc}")
        return

    index = VectorIndex(provider)
    index.add(documents, metadata=[{"doc_id": i} for i in range(len(documents))])
    results = index.search(query, top_k=len(documents))

    print(f"\n  Query: {query!r}   Provider: {pname}")
    headers = ["Rank", "Score", "Document"]
    rows = []
    for rank, r in enumerate(results, 1):
        rows.append([rank, f"{r['score']:.4f}", r["text"][:60]])
    _fmt_table(headers, rows)


# ── section d: Nobel Prize RAG pipeline ───────────────────────────────────────


def section_d(provider_name: str | None = None) -> None:
    _hr("Section D — Nobel Prize RAG pipeline")
    from agentkit.chunking import sentence_aware_chunking
    from agentkit.embeddings import get_embedding_provider
    from agentkit.retrieval import VectorIndex
    from agentkit.tokens import count_tokens

    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        print("  [SKIP] TAVILY_API_KEY not set")
        return

    try:
        from tavily import TavilyClient  # type: ignore[import]
    except ImportError:
        print("  [SKIP] tavily-python not installed")
        return

    print("  Fetching Nobel Prize in Physics 2024 content...")
    client = TavilyClient(api_key=tavily_key)
    resp = client.search(
        "Nobel Prize in Physics 2024 machine learning neural networks",
        search_depth="basic",
        include_raw_content=True,
        max_results=3,
    )

    raw_texts: list[str] = []
    for r in resp.get("results", []):
        content = r.get("raw_content") or r.get("content") or ""
        if content:
            raw_texts.append(content)

    if not raw_texts:
        print("  [SKIP] No content retrieved")
        return

    full_text = "\n\n".join(raw_texts)
    total_tokens = count_tokens(full_text)
    print(f"  Raw content: {len(full_text)} chars, ~{total_tokens} tokens")

    # Chunk
    chunks = sentence_aware_chunking(full_text, chunk_size=500, overlap=50)
    print(f"  After chunking (500/50): {len(chunks)} chunks")

    # Embed and index
    pname = provider_name or "voyage"
    try:
        provider = get_embedding_provider(pname)
    except Exception as exc:  # noqa: BLE001
        print(f"  [SKIP] Embedding provider: {exc}")
        return

    index = VectorIndex(provider)
    index.add(chunks)
    print(f"  Indexed {len(index)} chunks with {provider.name}")

    # Search
    query = "quantum computing"
    results = index.search(query, top_k=3)
    print(f"\n  Query: {query!r}")
    for i, r in enumerate(results, 1):
        print(f"\n  [{i}] score={r['score']:.4f}")
        print(f"      {r['text'][:200]!r}")

    # Token savings
    top3_text = " ".join(r["text"] for r in results)
    retrieved_tokens = count_tokens(top3_text)
    saving_pct = (1 - retrieved_tokens / total_tokens) * 100
    print(f"\n  Token savings: {total_tokens} → {retrieved_tokens} "
          f"({saving_pct:.1f}% reduction with top-3 chunks)")


# ── section e: Tavily advanced search comparison ──────────────────────────────


def section_e(provider_name: str | None = None) -> None:
    _hr("Section E — Tavily advanced search comparison")
    from agentkit.chunking import sentence_aware_chunking
    from agentkit.embeddings import get_embedding_provider
    from agentkit.retrieval import VectorIndex
    from agentkit.tokens import count_tokens

    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        print("  [SKIP] TAVILY_API_KEY not set")
        return

    try:
        from tavily import TavilyClient  # type: ignore[import]
    except ImportError:
        print("  [SKIP] tavily-python not installed")
        return

    pname = provider_name or "voyage"
    try:
        provider = get_embedding_provider(pname)
    except Exception as exc:  # noqa: BLE001
        print(f"  [SKIP] Embedding provider: {exc}")
        return

    client = TavilyClient(api_key=tavily_key)
    query = "Nobel Prize in Physics 2024 machine learning"
    search_query = query
    retrieval_query = "quantum computing machine learning"

    configs = [
        {"search_depth": "basic", "label": "basic (no chunks)"},
        {"search_depth": "advanced", "label": "advanced"},
    ]

    results_table: list[list[Any]] = []

    for cfg in configs:
        try:
            kwargs: dict[str, Any] = {
                "search_depth": cfg["search_depth"],
                "include_raw_content": True,
                "max_results": 3,
            }
            if cfg["search_depth"] == "advanced":
                kwargs["chunks_per_source"] = 3

            resp = client.search(search_query, **kwargs)

            raw_texts = []
            for r in resp.get("results", []):
                content = r.get("raw_content") or r.get("content") or ""
                if content:
                    raw_texts.append(content)

            full_text = "\n\n".join(raw_texts)
            total_tokens = count_tokens(full_text)

            chunks = sentence_aware_chunking(full_text, chunk_size=500, overlap=50)
            index = VectorIndex(provider)
            index.add(chunks)
            top_results = index.search(retrieval_query, top_k=3)
            top3_text = " ".join(r["text"] for r in top_results)
            retrieved_tokens = count_tokens(top3_text)

            best_score = top_results[0]["score"] if top_results else 0.0
            results_table.append([
                cfg["label"],
                len(raw_texts),
                len(chunks),
                total_tokens,
                retrieved_tokens,
                f"{best_score:.4f}",
            ])
        except Exception as exc:  # noqa: BLE001
            results_table.append([cfg["label"], "ERROR", str(exc), "", "", ""])

    headers = ["Config", "Docs", "Chunks", "Total tok", "Top-3 tok", "Best score"]
    _fmt_table(headers, results_table)


# ── section f: provider benchmark ─────────────────────────────────────────────


def section_f(provider_name: str | None = None) -> None:
    _hr("Section F — Provider benchmark: 200 chunks")

    # Generate synthetic chunks
    base = (
        "Vector embeddings map tokens to high-dimensional spaces where semantic "
        "similarity corresponds to geometric proximity. This enables nearest-neighbour "
        "retrieval over large corpora without keyword matching. "
    )
    chunks = [(base * 3)[: 400 + (i % 5) * 20] for i in range(200)]
    print(f"  Corpus: {len(chunks)} synthetic chunks, avg {sum(len(c) for c in chunks)//len(chunks)} chars")

    providers_to_bench: list[str] = (
        [provider_name] if provider_name else ["voyage", "local"]
    )

    rows: list[list[Any]] = []
    for pname in providers_to_bench:
        try:
            from agentkit.embeddings import get_embedding_provider

            provider = get_embedding_provider(pname)

            tracemalloc.start()
            t0 = time.perf_counter()
            embs = provider.embed(chunks)
            elapsed = time.perf_counter() - t0
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            peak_mb = peak / 1024 / 1024
            rows.append([
                provider.name,
                embs.shape[1],
                f"{elapsed:.2f}s",
                f"{elapsed / len(chunks) * 1000:.1f}ms",
                f"{peak_mb:.1f}MB",
            ])
        except Exception as exc:  # noqa: BLE001
            rows.append([pname, "ERROR", str(exc), "", ""])

    headers = ["Provider", "Dims", "Total time", "Per chunk", "Peak RAM"]
    _fmt_table(headers, rows)


# ── main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Block 10 vector experiments")
    parser.add_argument(
        "--section",
        default="all",
        choices=["a", "b", "c", "d", "e", "f", "all"],
        help="Which section to run (default: all)",
    )
    parser.add_argument(
        "--provider",
        default=None,
        choices=["voyage", "local"],
        help="Override embedding provider (default: per-section default)",
    )
    args = parser.parse_args()

    sections = {
        "a": lambda: section_a(args.provider),
        "b": section_b,
        "c": lambda: section_c(args.provider),
        "d": lambda: section_d(args.provider),
        "e": lambda: section_e(args.provider),
        "f": lambda: section_f(args.provider),
    }

    to_run = list(sections.keys()) if args.section == "all" else [args.section]
    for key in to_run:
        try:
            sections[key]()
        except KeyboardInterrupt:
            print("\n  [interrupted]")
            sys.exit(1)
        except Exception:  # noqa: BLE001
            print(f"\n  [ERROR in section {key}]")
            traceback_str = __import__("traceback").format_exc()
            print(traceback_str)

    print()


if __name__ == "__main__":
    main()
