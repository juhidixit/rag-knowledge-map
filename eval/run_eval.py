"""Benchmark chunking strategies x retrieval modes on QASPER (arXiv NLP papers).

    python eval/run_eval.py --qasper data/qasper-dev-v0.3.json --out results/

Relevance: a retrieved chunk counts as relevant if it covers >= 50% of the words of at
least one annotator-marked evidence paragraph (or >= 100 words of it, for very long
paragraphs that every chunker has to split). This keeps the comparison fair between
structure-aware chunkers (whole paragraphs) and fixed windows (which cut paragraphs).

Two scopes are reported:
  * corpus     - search all papers at once (multi-document library)
  * in-doc     - search only inside the paper the question is about (chat-with-a-PDF)
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qasper import load_qasper  # noqa: E402

from ragmap.chunking import chunk_document  # noqa: E402
from ragmap.embeddings import get_embedder  # noqa: E402
from ragmap.engine import MODES, Engine, EngineConfig  # noqa: E402
from ragmap.retrieval import CrossEncoderReranker  # noqa: E402

CHUNKERS = {
    "fixed-200w": dict(strategy="fixed", max_words=200, overlap=40),
    "paragraph": dict(strategy="paragraph", max_words=200),
    "section": dict(strategy="section", max_words=200, header=False),
    "section+header": dict(strategy="section", max_words=200, header=True),
}


def paragraph_spans(doc) -> list[tuple[int, int]]:
    spans, n = [], 0
    for p in doc.paragraphs:
        w = len(p.text.split())
        spans.append((n, n + w))
        n += w
    return spans


def relevant_evidence(chunk, spans, evidence: set[int]) -> set[int]:
    """Evidence paragraphs that this chunk sufficiently covers."""
    out = set()
    for e in evidence:
        a, b = spans[e]
        ov = max(0, min(b, chunk.word_end) - max(a, chunk.word_start))
        if ov and (ov >= 0.5 * (b - a) or ov >= 100):
            out.add(e)
    return out


def score_question(hits, spans, evidence, ks=(1, 3, 5, 10)) -> dict:
    rel = [relevant_evidence(h.chunk, spans, evidence) if h.chunk.doc_id == hits_doc else set() for h, hits_doc in hits]
    flags = [1 if r else 0 for r in rel]
    out = {}
    for k in ks:
        out[f"hit@{k}"] = float(any(flags[:k]))
        covered = set().union(*rel[:k]) if rel[:k] else set()
        out[f"ev_recall@{k}"] = len(covered) / len(evidence)
    first = next((i for i, f in enumerate(flags[:10]) if f), None)
    out["mrr@10"] = 0.0 if first is None else 1.0 / (first + 1)
    dcg = sum(f / math.log2(i + 2) for i, f in enumerate(flags[:10]))
    n_rel = min(10, max(1, len(evidence)))
    idcg = sum(1 / math.log2(i + 2) for i in range(n_rel))
    out["ndcg@10"] = dcg / idcg
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qasper", required=True)
    ap.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--rerank-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--chunkers", default=",".join(CHUNKERS))
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--limit", type=int, default=0, help="max questions (0 = all)")
    ap.add_argument("--latency-queries", type=int, default=200)
    ap.add_argument("--rerank-sweep", default="5,10,20,30", help="rerank depths for the quality/latency trade-off")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "cache").mkdir(parents=True, exist_ok=True)
    out_file = out / "qasper_results.json"

    docs, questions = load_qasper(args.qasper)
    if args.limit:
        questions = questions[: args.limit]
    doc_by_id = {d.doc_id: d for d in docs}
    spans = {d.doc_id: paragraph_spans(d) for d in docs}
    print(f"{len(docs)} papers, {len(questions)} answerable questions with paragraph evidence", flush=True)

    embedder = get_embedder(args.embed_model)
    reranker = CrossEncoderReranker(args.rerank_model)
    modes = args.modes.split(",")
    results: dict = {
        "meta": {
            "dataset": Path(args.qasper).name,
            "n_papers": len(docs),
            "n_questions": len(questions),
            "embed_model": args.embed_model,
            "rerank_model": args.rerank_model,
            "machine": f"{platform.machine()} / {platform.processor() or 'cpu'} / {__import__('os').cpu_count()} vCPU",
            "torch_threads": __import__("torch").get_num_threads(),
        },
        "runs": {},
    }
    if out_file.exists():  # resume an interrupted run
        prev = json.loads(out_file.read_text())
        if prev.get("meta", {}).get("n_questions") == len(questions):
            results["runs"] = prev.get("runs", {})
            results["rerank_sweep"] = prev.get("rerank_sweep", {})

    def save() -> None:
        out_file.write_text(json.dumps(results, indent=2))

    def build(cname: str) -> tuple[Engine, list]:
        spec = dict(CHUNKERS[cname])
        strategy = spec.pop("strategy")
        eng = Engine(EngineConfig(chunking=strategy), embedder=embedder, reranker=reranker)
        flat = [c for d in docs for c in chunk_document(d, strategy, **spec)]
        cache = out / "cache" / f"{cname}.npy"
        if cache.exists() and np.load(cache, mmap_mode="r").shape[0] == len(flat):
            vecs = np.load(cache)
        else:
            vecs = embedder.encode([c.index_text for c in flat], batch_size=128)
            np.save(cache, vecs)
        eng.chunks, eng._doc_of, eng.docs = flat, [c.doc_id for c in flat], doc_by_id
        eng.bm25.add([c.index_text for c in flat])
        eng.store.add(vecs)
        return eng, flat

    for cname in args.chunkers.split(","):
        t = time.perf_counter()
        eng, flat = build(cname)
        build_s = time.perf_counter() - t
        avg_words = statistics.mean(c.word_end - c.word_start for c in flat)
        print(f"[{cname}] {len(flat)} chunks (avg {avg_words:.0f} words), indexed in {build_s:.1f}s", flush=True)
        run = results["runs"].setdefault(cname, {})
        run.update(n_chunks=len(flat), avg_words=round(avg_words, 1))
        for scope in ("in-doc", "corpus"):
            for mode in modes:
                key = f"{scope}/{mode}"
                if key in run:
                    continue
                rows = []
                t = time.perf_counter()
                for q in questions:
                    res = eng.search(q.text, k=10, mode=mode, doc_ids=[q.doc_id] if scope == "in-doc" else None)
                    rows.append(score_question([(h, q.doc_id) for h in res["hits"]], spans[q.doc_id], q.evidence))
                agg = {m: round(float(np.mean([r[m] for r in rows])), 4) for m in rows[0]}
                agg["seconds"] = round(time.perf_counter() - t, 1)
                run[key] = agg
                save()
                print(
                    f"  {scope:7s} {mode:14s} " + " ".join(f"{m}={agg[m]:.3f}" for m in ("hit@5", "mrr@10", "ndcg@10")),
                    flush=True,
                )

    # Rerank depth sweep (quality vs latency) on the default chunker.
    lat_chunker = "section+header" if "section+header" in results["runs"] else args.chunkers.split(",")[-1]
    eng, flat = build(lat_chunker)
    sweep = results.setdefault("rerank_sweep", {})
    for depth in [int(x) for x in args.rerank_sweep.split(",") if x]:
        for scope in ("in-doc", "corpus"):
            key = f"{scope}/top{depth}"
            if key in sweep:
                continue
            eng.config.rerank_top = depth
            rows, ts = [], []
            for q in questions:
                t = time.perf_counter()
                res = eng.search(q.text, k=10, mode="hybrid_rerank", doc_ids=[q.doc_id] if scope == "in-doc" else None)
                ts.append((time.perf_counter() - t) * 1000)
                rows.append(score_question([(h, q.doc_id) for h in res["hits"]], spans[q.doc_id], q.evidence))
            agg = {m: round(float(np.mean([r[m] for r in rows])), 4) for m in ("hit@5", "mrr@10", "ndcg@10")}
            agg.update(p50_ms=round(float(np.percentile(ts, 50)), 1), p95_ms=round(float(np.percentile(ts, 95)), 1))
            sweep[key] = agg
            save()
            print(f"sweep {key} {agg}", flush=True)
    eng.config.rerank_top = EngineConfig().rerank_top

    # Latency of each mode, one query at a time, default config.
    lq = [q.text for q in questions[: args.latency_queries]]
    for q in lq[:10]:  # warm-up
        eng.search(q, mode="hybrid_rerank")
    latency = {}
    for mode in modes:
        for scope in ("corpus", "in-doc"):
            ts = []
            for q, qq in zip(lq, questions, strict=False):
                t = time.perf_counter()
                eng.search(q, k=5, mode=mode, doc_ids=[qq.doc_id] if scope == "in-doc" else None)
                ts.append((time.perf_counter() - t) * 1000)
            latency[f"{scope}/{mode}"] = {
                "p50_ms": round(float(np.percentile(ts, 50)), 1),
                "p95_ms": round(float(np.percentile(ts, 95)), 1),
                "mean_ms": round(float(np.mean(ts)), 1),
            }
            print(f"latency {scope:7s} {mode:14s} {latency[f'{scope}/{mode}']}", flush=True)
    results["latency"] = {
        "chunker": lat_chunker,
        "rerank_top": eng.config.rerank_top,
        "n_chunks": len(flat),
        "n_queries": len(lq),
        **latency,
    }

    save()
    print(f"wrote {out_file}")


if __name__ == "__main__":
    main()
