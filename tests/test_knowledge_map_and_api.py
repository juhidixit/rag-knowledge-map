import pymupdf
import pytest
from fastapi.testclient import TestClient

from ragmap import knowledge_map as km_mod
from ragmap.api import create_app
from ragmap.chunking import chunk_document
from ragmap.ingest import parse_pdf
from ragmap.knowledge_map import build_knowledge_map

from .conftest import SAMPLE


@pytest.fixture(params=["spacy", "regex"])
def extractor(request, monkeypatch):
    if request.param == "spacy" and not km_mod._spacy():
        pytest.skip("en_core_web_sm not installed")
    if request.param == "regex":
        monkeypatch.setattr(km_mod, "_NLP", False)
    return request.param


def test_knowledge_map_structure(doc, extractor):
    chunks = chunk_document(doc, "section", max_words=60)
    km = build_knowledge_map(doc, chunks)
    labels = {n["label"].lower() for n in km["nodes"]}
    assert "lithium-ion battery packs" in labels
    if extractor == "spacy":  # noun-phrase extraction gives cleaner concepts
        assert {"liquid cooling", "battery thermal management system", "glycol coolant"} <= labels
    ids = {c.chunk_id for c in chunks}
    for n in km["nodes"]:
        assert n["chunks"] and set(n["chunks"]) <= ids  # every concept links to its sources
    n_nodes = len(km["nodes"])
    for link in km["links"]:
        assert 0 <= link["source"] < n_nodes and 0 <= link["target"] < n_nodes
    assert km["topics"] and sum(t["size"] for t in km["topics"]) == n_nodes


def _make_pdf() -> bytes:
    pdf = pymupdf.open()
    page = pdf.new_page()
    y = 72
    for block in SAMPLE.split("\n\n"):
        block = block.strip()
        size = 18 if block.startswith("# ") else 14 if block.startswith("## ") else 10
        text = block.lstrip("# ").replace("\n", " ")
        rect = pymupdf.Rect(72, y, 540, y + 200)
        used = page.insert_textbox(rect, text, fontsize=size)
        y += 200 - used + 12
        if y > 700:
            page, y = pdf.new_page(), 72
    return pdf.tobytes()


def test_pdf_ingestion_recovers_headings():
    doc = parse_pdf(_make_pdf())
    sections = {p.section for p in doc.paragraphs}
    assert "Liquid Cooling" in sections and "Model Predictive Control" in sections
    assert all(p.page for p in doc.paragraphs)


@pytest.fixture
def client(engine):
    return TestClient(create_app(engine=engine))


def test_api_end_to_end(client):
    r = client.post("/api/documents", files={"file": ("ev.pdf", _make_pdf(), "application/pdf")})
    assert r.status_code == 200, r.text
    km = r.json()
    doc_id = km["doc_id"]
    assert km["nodes"]

    assert any(d["doc_id"] == doc_id for d in client.get("/api/documents").json())
    assert client.get(f"/api/documents/{doc_id}/map").json()["doc_id"] == doc_id

    r = client.post(
        "/api/search",
        json={"query": "How much faster is liquid cooling than air cooling?", "doc_ids": [doc_id], "answer": True},
    )
    body = r.json()
    assert r.status_code == 200
    assert "Liquid Cooling" in body["hits"][0]["section"]
    assert body["answer"]["generator"] == "extractive"
    assert body["concepts"][doc_id]  # retrieved passages light up concepts on the map

    assert client.post("/api/search", json={"query": "x", "doc_ids": ["nope"]}).status_code == 404
    assert client.post("/api/search", json={"query": ""}).status_code == 422
    assert client.get("/api/documents/nope/map").status_code == 404


def test_upload_rejects_empty(client):
    r = client.post("/api/documents", files={"file": ("empty.txt", b"   ", "text/plain")})
    assert r.status_code == 422


def test_index_page(client):
    assert "ragmap" in client.get("/").text
