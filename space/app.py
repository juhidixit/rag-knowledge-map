"""Hugging Face Space entry point: the ragmap web app with a few open-access papers preloaded.

The Space runs on the free ZeroGPU tier, but nothing here uses a GPU: embedding, reranking and the
knowledge map all run on CPU. Answers are extractive (no LLM, no API key), so the demo cannot spend money.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# `spaces` must be imported before torch on ZeroGPU hardware. It is absent when running locally.
try:
    import spaces  # noqa: F401
except ImportError:
    spaces = None

HERE = Path(__file__).resolve().parent
# Deployed layout: ./src/ragmap next to this file. Repository layout: ../src/ragmap.
for src in (HERE / "src", HERE.parent / "src"):
    if (src / "ragmap").is_dir():
        sys.path.insert(0, str(src))
        break

os.environ.setdefault("RAGMAP_DEVICE", "cpu")
os.environ.setdefault("RAGMAP_MAX_UPLOAD_MB", "10")
os.environ.setdefault("RAGMAP_MAX_USER_DOCS", "10")
os.environ.setdefault(
    "RAGMAP_DEMO_NOTICE", "Demo instance: uploads are deleted on restart. Answers are extractive (no LLM)."
)
os.environ.setdefault("RAGMAP_ABOUT_URL", "/about")
os.environ.pop("ANTHROPIC_API_KEY", None)  # never call a paid API from the public demo
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import gradio as gr  # noqa: E402
import uvicorn  # noqa: E402

from ragmap.api import create_app  # noqa: E402
from ragmap.engine import Engine, EngineConfig  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
LIBRARY = HERE / "library"
REPO = "https://github.com/juhidixit/rag-knowledge-map"


def about_markdown() -> str:
    lines = [
        "# ragmap demo",
        f"Source code, benchmark and write-up: [{REPO}]({REPO}).",
        "",
        "The main app is at [/](/). This page lists the preloaded papers and their licenses.",
        "",
        "## Preloaded papers",
    ]
    meta = HERE / "papers.json"
    papers = json.loads(meta.read_text()) if meta.exists() else []
    for p in papers:
        lines.append(f"- {p['attribution']}")
    if not papers:
        lines.append("- None (upload your own PDF on the main page).")
    lines += [
        "",
        "## Limits",
        "- Uploads up to 10 MB, at most 10 uploaded documents until the Space restarts.",
        "- Uploads are kept in memory only and are deleted on restart.",
        "- Answers are extractive: the best-matching sentences of the top passage. No LLM is called.",
    ]
    return "\n".join(lines)


def build_app():
    # Download (first boot) and load both models before serving, so the first question is not slow.
    engine = Engine(EngineConfig())
    _ = engine.reranker
    app = create_app(engine=engine, preload=str(LIBRARY))
    with gr.Blocks(title="About ragmap") as about:
        gr.Markdown(about_markdown())
    # On ZeroGPU hardware `spaces` patches mount_gradio_app to register the app with the scheduler.
    return gr.mount_gradio_app(app, about, path="/about")


if __name__ == "__main__":
    uvicorn.run(build_app(), host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))
