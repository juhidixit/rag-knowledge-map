import numpy as np
import pytest

from ragmap.engine import MODES
from ragmap.retrieval import BM25Index, FaissStore, rrf_fuse, tokenize


def test_tokenize_drops_stopwords_and_folds_plurals():
    assert tokenize("The models are trained on batteries") == ["model", "train", "batteri"]


def test_bm25_prefers_matching_document():
    idx = BM25Index()
    idx.add(["liquid cooling plates", "air cooling fans", "model predictive control"])
    s = idx.scores("predictive control")
    assert int(np.argmax(s)) == 2
    assert s[0] == 0


def test_bm25_idf_rewards_rare_terms():
    idx = BM25Index()
    idx.add(["cooling cooling battery", "cooling fan", "cooling chiller"])
    assert np.argmax(idx.scores("cooling chiller")) == 2


def test_rrf_fuse_rewards_agreement():
    fused = rrf_fuse([[1, 2, 3], [3, 1, 4]])
    ranked = sorted(fused, key=lambda i: -fused[i])
    assert ranked[0] == 1 and set(ranked) == {1, 2, 3, 4}


def test_faiss_filter():
    store = FaissStore(4)
    v = np.eye(4, dtype=np.float32)
    store.add(v)
    sims, ids = store.search(v[0], k=2, allowed=np.array([1, 2]))
    assert set(ids.tolist()) <= {1, 2}


@pytest.mark.parametrize("mode", MODES)
def test_engine_modes_return_relevant_passage(engine, mode):
    res = engine.search("model predictive control pre-cools the battery", k=3, mode=mode)
    assert res["hits"], mode
    assert res["hits"][0].chunk.section == "Model Predictive Control"
    assert res["timings"]["total_ms"] >= 0


def test_engine_doc_filter(engine, doc):
    assert engine.search("cooling", doc_ids=["missing"])["hits"] == []
    assert engine.search("cooling", doc_ids=[doc.doc_id])["hits"]


def test_engine_rejects_unknown_mode(engine):
    with pytest.raises(ValueError):
        engine.search("x", mode="magic")


def test_duplicate_document_is_not_reindexed(engine, doc):
    n = len(engine.chunks)
    engine.add_document(doc)
    assert len(engine.chunks) == n
