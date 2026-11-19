"""Sparse (BM25), dense (FAISS / pgvector) and hybrid retrieval, plus cross-encoder reranking.

Everything works on integer positions into a shared chunk list, so the sparse and dense
sides can be fused cheaply.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
from scipy import sparse

from ragmap.embeddings import DEFAULT_DEVICE, DEFAULT_RERANK_MODEL

_TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as _STOP
except Exception:  # pragma: no cover
    _STOP = frozenset()


def _stem(tok: str) -> str:
    # Tiny suffix stripper: enough to match "models"/"model", "trained"/"train".
    for suf in ("ings", "ing", "edly", "ed", "es", "s"):
        if len(tok) > len(suf) + 3 and tok.endswith(suf):
            return tok[: -len(suf)]
    return tok


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


# --------------------------------------------------------------------------------------
# Sparse
# --------------------------------------------------------------------------------------
class BM25Index:
    """Okapi BM25 over a sparse matrix: scoring a query is one sparse mat-vec."""

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.vocab: dict[str, int] = {}
        self._docs: list[list[str]] = []
        self.W: sparse.csr_matrix | None = None

    def __len__(self) -> int:
        return len(self._docs)

    def add(self, texts: Sequence[str]) -> None:
        self._docs.extend(tokenize(t) for t in texts)
        self._build()

    def _build(self) -> None:
        rows, cols, vals = [], [], []
        for i, toks in enumerate(self._docs):
            counts: dict[int, int] = {}
            for t in toks:
                j = self.vocab.setdefault(t, len(self.vocab))
                counts[j] = counts.get(j, 0) + 1
            rows.extend([i] * len(counts))
            cols.extend(counts.keys())
            vals.extend(counts.values())
        n, v = len(self._docs), max(1, len(self.vocab))
        tf = sparse.csr_matrix((np.array(vals, dtype=np.float32), (rows, cols)), shape=(n, v))
        dl = np.asarray(tf.sum(axis=1)).ravel()
        avgdl = dl.mean() if n else 1.0
        df = np.bincount(tf.indices, minlength=v)
        idf = np.log1p((n - df + 0.5) / (df + 0.5)).astype(np.float32)
        tf = tf.tocoo()
        denom = tf.data + self.k1 * (1 - self.b + self.b * dl[tf.row] / max(avgdl, 1e-9))
        w = idf[tf.col] * tf.data * (self.k1 + 1) / denom
        self.W = sparse.csr_matrix((w.astype(np.float32), (tf.row, tf.col)), shape=(n, v))

    def scores(self, query: str) -> np.ndarray:
        if self.W is None or not len(self):
            return np.zeros(0, dtype=np.float32)
        ids = [self.vocab[t] for t in set(tokenize(query)) if t in self.vocab]
        if not ids:
            return np.zeros(len(self), dtype=np.float32)
        return np.asarray(self.W[:, ids].sum(axis=1)).ravel()


# --------------------------------------------------------------------------------------
# Dense
# --------------------------------------------------------------------------------------
class FaissStore:
    """Exact inner-product search (vectors are L2-normalised, so IP == cosine)."""

    def __init__(self, dim: int):
        import faiss

        self._faiss = faiss
        self.index = faiss.IndexFlatIP(dim)

    def __len__(self) -> int:
        return self.index.ntotal

    def add(self, vectors: np.ndarray) -> None:
        self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))

    def search(self, qvec: np.ndarray, k: int, allowed: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        k = min(k, len(self) if allowed is None else len(allowed))
        if k <= 0:
            return np.zeros(0, np.float32), np.zeros(0, np.int64)
        params = None
        if allowed is not None:
            params = self._faiss.SearchParameters(sel=self._faiss.IDSelectorBatch(allowed.astype(np.int64)))
        D, ids = self.index.search(qvec.reshape(1, -1).astype(np.float32), k, params=params)
        keep = ids[0] >= 0
        return D[0][keep], ids[0][keep]


class PgVectorStore:
    """Same interface backed by Postgres + pgvector (HNSW, cosine). Optional dependency.

    Use it when the corpus outgrows memory or several API replicas must share one index:
    ``RAGMAP_VECTOR_STORE=pgvector RAGMAP_PG_DSN=postgresql://...``.
    """

    def __init__(self, dim: int, dsn: str, table: str = "ragmap_chunks"):
        import psycopg
        from pgvector.psycopg import register_vector

        self.dim, self.table = dim, table
        self.conn = psycopg.connect(dsn, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        self.conn.execute(f"DROP TABLE IF EXISTS {table}")
        self.conn.execute(f"CREATE TABLE {table} (pos bigint PRIMARY KEY, embedding vector({dim}))")
        self.conn.execute(f"CREATE INDEX ON {table} USING hnsw (embedding vector_cosine_ops)")
        self._n = 0

    def __len__(self) -> int:
        return self._n

    def add(self, vectors: np.ndarray) -> None:
        with self.conn.cursor() as cur:
            with cur.copy(f"COPY {self.table} (pos, embedding) FROM STDIN WITH (FORMAT BINARY)") as copy:
                copy.set_types(["int8", "vector"])
                for i, v in enumerate(vectors):
                    copy.write_row((self._n + i, v.astype(np.float32)))
        self._n += len(vectors)

    def search(self, qvec: np.ndarray, k: int, allowed: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        q = qvec.astype(np.float32)
        if allowed is None:
            rows = self.conn.execute(
                f"SELECT pos, 1 - (embedding <=> %s) FROM {self.table} ORDER BY embedding <=> %s LIMIT %s", (q, q, k)
            ).fetchall()
        else:
            rows = self.conn.execute(
                f"SELECT pos, 1 - (embedding <=> %s) FROM {self.table} WHERE pos = ANY(%s) "
                f"ORDER BY embedding <=> %s LIMIT %s",
                (q, list(map(int, allowed)), q, k),
            ).fetchall()
        if not rows:
            return np.zeros(0, np.float32), np.zeros(0, np.int64)
        ids, sims = zip(*rows, strict=True)
        return np.array(sims, dtype=np.float32), np.array(ids, dtype=np.int64)


# --------------------------------------------------------------------------------------
# Fusion + reranking
# --------------------------------------------------------------------------------------
def rrf_fuse(
    rankings: Sequence[Sequence[int]], k: int = 60, weights: Sequence[float] | None = None
) -> dict[int, float]:
    """Reciprocal Rank Fusion: score(d) = sum_i w_i / (k + rank_i(d))."""
    weights = weights or [1.0] * len(rankings)
    fused: dict[int, float] = {}
    for w, ranking in zip(weights, rankings, strict=True):
        for r, idx in enumerate(ranking, start=1):
            fused[idx] = fused.get(idx, 0.0) + w / (k + r)
    return fused


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL, device: str | None = None, max_length: int = 256):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, device=device or DEFAULT_DEVICE, max_length=max_length)
        self.name = model_name

    def score(self, query: str, passages: Sequence[str], batch_size: int = 32) -> np.ndarray:
        if not passages:
            return np.zeros(0, dtype=np.float32)
        return np.asarray(
            self.model.predict([(query, p) for p in passages], batch_size=batch_size, show_progress_bar=False),
            dtype=np.float32,
        )
