"""FastAPI service: upload documents, explore their knowledge maps, ask questions."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ragmap.engine import Engine, EngineConfig
from ragmap.generate import generate_answer
from ragmap.ingest import load_path, parse_pdf, parse_text
from ragmap.knowledge_map import build_knowledge_map

log = logging.getLogger("ragmap")
STATIC = Path(__file__).parent / "static"
MAX_UPLOAD_MB = int(os.environ.get("RAGMAP_MAX_UPLOAD_MB", "40"))


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    k: int = Field(default=5, ge=1, le=50)
    mode: Literal["bm25", "dense", "hybrid", "hybrid_rerank"] = "hybrid_rerank"
    doc_ids: list[str] | None = None
    answer: bool = False


def supporting_concepts(kmap: dict, hits, limit: int = 12) -> list[int]:
    """Concepts whose source passages were retrieved, weighted by retrieval rank (1/rank)."""
    rank_of = {h.chunk.chunk_id: h.rank for h in hits}
    scored = []
    for n in kmap["nodes"]:
        w = sum(1.0 / rank_of[c] for c in n["chunks"] if c in rank_of)
        if w >= 0.5:  # appears in the top-2 passages, or in several lower ones
            scored.append((w * (1 + n["score"]) ** 0.5, n["id"]))
    return [i for _, i in sorted(scored, reverse=True)[:limit]]


def create_app(engine: Engine | None = None, preload: str | None = None) -> FastAPI:
    maps: dict[str, dict] = {}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        _preload()
        yield

    app = FastAPI(
        title="ragmap", version="0.1.0", description="Hybrid RAG + interactive knowledge maps", lifespan=lifespan
    )

    @lru_cache(maxsize=1)
    def get_engine() -> Engine:
        return engine or Engine(EngineConfig())

    def index_document(doc) -> dict:
        eng = get_engine()
        chunks = eng.add_document(doc)
        if doc.doc_id not in maps:
            maps[doc.doc_id] = build_knowledge_map(doc, chunks, embedder=eng.embedder)
        return maps[doc.doc_id]

    def _preload() -> None:
        folder = preload or os.environ.get("RAGMAP_PRELOAD")
        if folder and Path(folder).is_dir():
            for p in sorted(Path(folder).iterdir()):
                if p.suffix.lower() in {".pdf", ".txt", ".md"}:
                    try:
                        index_document(load_path(p))
                        log.info("preloaded %s", p.name)
                    except Exception as exc:  # keep serving even if one file is bad
                        log.warning("could not preload %s: %s", p.name, exc)

    @app.get("/api/health")
    def health() -> dict:
        eng = get_engine()
        return {"status": "ok", "documents": len(eng.docs), "chunks": len(eng.chunks)}

    @app.post("/api/documents")
    async def upload(file: UploadFile = File(...)) -> dict:
        data = await file.read()
        if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(413, f"File larger than {MAX_UPLOAD_MB} MB")
        name = file.filename or "upload"
        try:
            if name.lower().endswith(".pdf") or data[:4] == b"%PDF":
                doc = parse_pdf(data)
            else:
                doc = parse_text(data.decode("utf-8", errors="replace"), title=Path(name).stem)
        except Exception as exc:
            raise HTTPException(422, f"Could not parse {name}: {exc}") from exc
        if not doc.paragraphs:
            raise HTTPException(422, "No text found in document")
        doc.meta["filename"] = name
        return index_document(doc)

    @app.get("/api/documents")
    def list_documents() -> list[dict]:
        return [
            {"doc_id": d.doc_id, "title": d.title, **maps.get(d.doc_id, {}).get("stats", {})}
            for d in get_engine().docs.values()
        ]

    @app.get("/api/documents/{doc_id}/map")
    def get_map(doc_id: str) -> dict:
        if doc_id not in maps:
            raise HTTPException(404, "Unknown document")
        return maps[doc_id]

    @app.get("/api/documents/{doc_id}/chunks")
    def get_chunks(doc_id: str, ids: str = "") -> list[dict]:
        eng = get_engine()
        if doc_id not in eng.docs:
            raise HTTPException(404, "Unknown document")
        wanted = set(filter(None, ids.split(",")))
        return [c.to_dict() for c in eng.doc_chunks(doc_id) if not wanted or c.chunk_id in wanted]

    @app.post("/api/search")
    def search(req: SearchRequest) -> dict:
        eng = get_engine()
        if req.doc_ids and not set(req.doc_ids) <= set(eng.docs):
            raise HTTPException(404, "Unknown document id")
        res = eng.search(req.query, k=req.k, mode=req.mode, doc_ids=req.doc_ids)
        lit = {
            doc_id: supporting_concepts(m, res["hits"])
            for doc_id, m in maps.items()
            if not req.doc_ids or doc_id in req.doc_ids
        }
        out = {
            "query": req.query,
            "mode": req.mode,
            "timings": res["timings"],
            "hits": [h.to_dict() for h in res["hits"]],
            "concepts": lit,
        }
        if req.answer:
            out["answer"] = generate_answer(req.query, res["hits"])
        return out

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    return app


app = create_app()
