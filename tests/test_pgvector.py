"""Integration test for the Postgres/pgvector store. Runs only when RAGMAP_TEST_PG_DSN is set, e.g.

    docker run -d -e POSTGRES_USER=ragmap -e POSTGRES_PASSWORD=ragmap -e POSTGRES_DB=ragmap -p 5432:5432 pgvector/pgvector:pg16
    RAGMAP_TEST_PG_DSN=postgresql://ragmap:ragmap@localhost:5432/ragmap pytest tests/test_pgvector.py
"""

import os

import numpy as np
import pytest

from ragmap.retrieval import FaissStore

DSN = os.environ.get("RAGMAP_TEST_PG_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="RAGMAP_TEST_PG_DSN not set")


def test_pgvector_matches_faiss():
    from ragmap.retrieval import PgVectorStore

    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(300, 32)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    pg, fa = PgVectorStore(32, DSN, table="ragmap_test"), FaissStore(32)
    pg.add(vecs[:200])
    pg.add(vecs[200:])  # second batch keeps global positions
    fa.add(vecs)
    q = vecs[250] + 0.01
    q /= np.linalg.norm(q)
    _, pg_ids = pg.search(q, 5)
    _, fa_ids = fa.search(q, 5)
    assert pg_ids[0] == fa_ids[0] == 250
    assert len(set(pg_ids) & set(fa_ids)) >= 4  # HNSW is approximate
    allowed = np.arange(0, 100)
    _, ids = pg.search(q, 5, allowed)
    assert set(ids.tolist()) <= set(allowed.tolist())
