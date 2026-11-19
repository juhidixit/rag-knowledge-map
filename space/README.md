---
title: ragmap – PDF knowledge maps
emoji: 🗺️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: app.py
pinned: true
license: mit
short_description: Turn a PDF into a knowledge map you can question
---

# ragmap: PDF knowledge maps

Open a preloaded paper, click a concept to see the passages and pages it came from, or ask a question and
see the cited passages and the concepts that support them. You can also upload your own PDF (up to 10 MB).

* Hybrid retrieval: BM25 + MiniLM dense embeddings (FAISS), Reciprocal Rank Fusion, cross-encoder reranking.
* Knowledge map: spaCy noun phrases, NPMI co-occurrence graph, Louvain topics, every node linked to its passages.
* Answers are extractive (no LLM, no API key). Uploads are kept in memory and deleted on restart.

Source code, the QASPER benchmark and the write-up: https://github.com/juhidixit/rag-knowledge-map

This Space is deployed from that repository by `.github/workflows/deploy-space.yml`; do not edit it here.

## Preloaded papers

<!-- PAPERS -->
