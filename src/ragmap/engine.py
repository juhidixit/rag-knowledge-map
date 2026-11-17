"""The retrieval engine: owns documents, chunks, the BM25 index, the vector store and the reranker."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

import numpy as np

from ragmap.chunking import chunk_document
from ragmap.embeddings import DEFAULT_RERANK_MODEL, Embedder, get_embedder
from ragmap.retrieval import BM25Index, CrossEncoderReranker, FaissStore, PgVectorStore, rrf_fuse
from ragmap.types import Chunk, Document, Hit

MODES = ("bm25", "dense", "hybrid", "hybrid_rerank")


@dataclass
class EngineConfig:
    chunking: str = "section"
    chunk_words: int = 200
    embed_model: str | None = None
    rerank_model: str | None = DEFAULT_RERANK_MODEL
    candidates: int = 50  # per retriever, before fusion
    rerank_top: int = 10  # fused candidates passed to the cross-encoder (see rerank-depth sweep)
    rrf_k: int = 60
    vector_store: str = os.environ.get("RAGMAP_VECTOR_STORE", "faiss")
    pg_dsn: str | None = os.environ.get("RAGMAP_PG_DSN")


class Engine:
    def __init__(self, config: EngineConfig | None = None, embedder: Embedder | None = None, reranker=None):
        self.config = config or EngineConfig()
        self.embedder = embedder or get_embedder(self.config.embed_model)
        self._reranker = reranker
        self.docs: dict[str, Document] = {}
        self.chunks: list[Chunk] = []
        self._doc_of: list[str] = []
        self.bm25 = BM25Index()
        if self.config.vector_store == "pgvector":
            if not self.config.pg_dsn:
                raise ValueError("RAGMAP_PG_DSN must be set for the pgvector store")
            self.store = PgVectorStore(self.embedder.dim, self.config.pg_dsn)
        else:
            self.store = FaissStore(self.embedder.dim)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ ingestion
    @property
    def reranker(self):
        if self._reranker is None and self.config.rerank_model:
            self._reranker = CrossEncoderReranker(self.config.rerank_model)
        return self._reranker

    def add_document(self, doc: Document, chunks: list[Chunk] | None = None) -> list[Chunk]:
        if doc.doc_id in self.docs:
            return self.doc_chunks(doc.doc_id)
        chunks = (
            chunks
            if chunks is not None
            else chunk_document(doc, self.config.chunking, max_words=self.config.chunk_words)
        )
        vectors = self.embedder.encode([c.index_text for c in chunks]) if chunks else None
        with self._lock:
            self.docs[doc.doc_id] = doc
            self.chunks.extend(chunks)
            self._doc_of.extend(c.doc_id for c in chunks)
            self.bm25.add([c.index_text for c in chunks])
            if vectors is not None:
                self.store.add(vectors)
        return chunks

    def add_documents(self, docs: list[Document]) -> None:
        """Bulk ingestion (one BM25 rebuild, batched embedding)."""
        new = [d for d in docs if d.doc_id not in self.docs]
        all_chunks = []
        for d in new:
            all_chunks.extend(chunk_document(d, self.config.chunking, max_words=self.config.chunk_words))
        vectors = self.embedder.encode([c.index_text for c in all_chunks], batch_size=128)
        with self._lock:
            for d in new:
                self.docs[d.doc_id] = d
            self.chunks.extend(all_chunks)
            self._doc_of.extend(c.doc_id for c in all_chunks)
            self.bm25.add([c.index_text for c in all_chunks])
            self.store.add(vectors)

    def doc_chunks(self, doc_id: str) -> list[Chunk]:
        return [c for c in self.chunks if c.doc_id == doc_id]

    def _allowed(self, doc_ids: list[str] | None) -> np.ndarray | None:
        if not doc_ids:
            return None
        wanted = set(doc_ids)
        return np.array([i for i, d in enumerate(self._doc_of) if d in wanted], dtype=np.int64)

    # ------------------------------------------------------------------ search
    def search(self, query: str, k: int = 5, mode: str = "hybrid_rerank", doc_ids: list[str] | None = None) -> dict:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        t0 = time.perf_counter()
        timings: dict[str, float] = {}
        allowed = self._allowed(doc_ids)
        n_cand = max(k, self.config.candidates)
        per: dict[int, dict[str, float]] = {}

        sparse_rank: list[int] = []
        if mode in ("bm25", "hybrid", "hybrid_rerank"):
            t = time.perf_counter()
            s = self.bm25.scores(query)
            if allowed is not None:
                mask = np.full(len(s), -np.inf, dtype=np.float32)
                mask[allowed] = s[allowed]
                s = mask
            top = np.argsort(-s)[:n_cand]
            sparse_rank = [int(i) for i in top if np.isfinite(s[i]) and s[i] > 0]
            for i in sparse_rank:
                per.setdefault(i, {})["bm25"] = round(float(s[i]), 4)
            timings["bm25_ms"] = (time.perf_counter() - t) * 1000

        dense_rank: list[int] = []
        if mode in ("dense", "hybrid", "hybrid_rerank"):
            t = time.perf_counter()
            q = self.embedder.encode([query])[0]
            timings["embed_ms"] = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            sims, ids = self.store.search(q, n_cand, allowed)
            dense_rank = [int(i) for i in ids]
            for i, sim in zip(dense_rank, sims, strict=True):
                per.setdefault(i, {})["dense"] = round(float(sim), 4)
            timings["dense_ms"] = (time.perf_counter() - t) * 1000

        if mode == "bm25":
            ranked = [(i, per[i]["bm25"]) for i in sparse_rank]
        elif mode == "dense":
            ranked = [(i, per[i]["dense"]) for i in dense_rank]
        else:
            fused = rrf_fuse([sparse_rank, dense_rank], k=self.config.rrf_k)
            ranked = sorted(fused.items(), key=lambda x: -x[1])
            for i, s in ranked:
                per[i]["rrf"] = round(s, 5)
            if mode == "hybrid_rerank" and self.reranker is not None and ranked:
                t = time.perf_counter()
                cand = [i for i, _ in ranked[: self.config.rerank_top]]
                ce = self.reranker.score(query, [self.chunks[i].text for i in cand])
                for i, s in zip(cand, ce, strict=True):
                    per[i]["rerank"] = round(float(s), 4)
                reranked = sorted(zip(cand, ce.tolist(), strict=True), key=lambda x: -x[1])
                # Candidates beyond the rerank depth keep their fused order below the reranked block.
                floor = min(ce.tolist()) - 1.0
                rest = [(i, floor - r) for r, (i, _) in enumerate(ranked[self.config.rerank_top :])]
                ranked = reranked + rest
                timings["rerank_ms"] = (time.perf_counter() - t) * 1000

        hits = [
            Hit(chunk=self.chunks[i], score=s, rank=r, scores=per.get(i, {})) for r, (i, s) in enumerate(ranked[:k], 1)
        ]
        timings["total_ms"] = (time.perf_counter() - t0) * 1000
        return {"query": query, "mode": mode, "hits": hits, "timings": {k_: round(v, 2) for k_, v in timings.items()}}
