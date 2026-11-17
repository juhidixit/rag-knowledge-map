"""Render results/qasper_results.json as Markdown tables (results/RESULTS.md)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

MODES = {"bm25": "BM25", "dense": "Dense (MiniLM)", "hybrid": "Hybrid (RRF)", "hybrid_rerank": "Hybrid + cross-encoder"}


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def main(path: str = "results/qasper_results.json") -> None:
    r = json.loads(Path(path).read_text())
    m = r["meta"]
    hub = {
        "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
        "ms-marco-MiniLM-L-6-v2": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    }
    for key in ("embed_model", "rerank_model"):  # local copies of the same Hugging Face weights
        m[key] = hub.get(Path(m[key]).name, m[key])
    out = [
        "# QASPER retrieval benchmark",
        "",
        f"{m['n_questions']} answerable questions with annotated evidence paragraphs over {m['n_papers']} full-text "
        f"arXiv NLP papers (QASPER dev v0.3). Embedder `{m['embed_model']}`, reranker `{m['rerank_model']}`. "
        f"Hardware: {m['machine']}, CPU only. All numbers in %, higher is better.",
        "",
    ]
    for scope, title in (
        ("in-doc", "Search inside the paper (chat-with-a-PDF)"),
        ("corpus", "Search across all 281 papers"),
    ):
        out += [
            f"## {title}",
            "",
            "| Chunking | Retriever | Hit@1 | Hit@5 | MRR@10 | nDCG@10 | Evidence recall@5 |",
            "|---|---|---|---|---|---|---|",
        ]
        for cname, run in r["runs"].items():
            for mode, label in MODES.items():
                a = run.get(f"{scope}/{mode}")
                if a:
                    out.append(
                        f"| {cname} | {label} | {pct(a['hit@1'])} | {pct(a['hit@5'])} | {pct(a['mrr@10'])} | "
                        f"{pct(a['ndcg@10'])} | {pct(a['ev_recall@5'])} |"
                    )
        out.append("")
    if r.get("rerank_sweep"):
        out += [
            "## Rerank depth: quality vs latency (section+header chunks)",
            "",
            "| Scope | Candidates reranked | Hit@5 | MRR@10 | nDCG@10 | p50 ms | p95 ms |",
            "|---|---|---|---|---|---|---|",
        ]
        for key, a in r["rerank_sweep"].items():
            scope, depth = key.split("/top")
            out.append(
                f"| {scope} | {depth} | {pct(a['hit@5'])} | {pct(a['mrr@10'])} | {pct(a['ndcg@10'])} | {a['p50_ms']} | {a['p95_ms']} |"
            )
        out.append("")
    if r.get("latency"):
        lat = r["latency"]
        out += [
            f"## End-to-end query latency ({lat['n_queries']} queries, {lat['n_chunks']} chunks indexed, one query at a time)",
            "",
            "| Retriever | Scope | p50 ms | p95 ms |",
            "|---|---|---|---|",
        ]
        for mode, label in MODES.items():
            for scope in ("in-doc", "corpus"):
                a = lat.get(f"{scope}/{mode}")
                if a:
                    out.append(f"| {label} | {scope} | {a['p50_ms']} | {a['p95_ms']} |")
        out.append("")
    dest = Path(path).with_name("RESULTS.md")
    dest.write_text("\n".join(out))
    print(dest.read_text())


if __name__ == "__main__":
    main(*sys.argv[1:])
