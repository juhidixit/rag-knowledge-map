"""Command line: `ragmap serve`, `ragmap map file.pdf`, `ragmap ask file.pdf "question"`."""

from __future__ import annotations

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser(prog="ragmap")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="run the web app + API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--preload", help="folder of PDFs/markdown to index at startup")
    m = sub.add_parser("map", help="print a document's knowledge map as JSON")
    m.add_argument("path")
    m.add_argument("--max-concepts", type=int, default=60)
    a = sub.add_parser("ask", help="retrieve passages from a document")
    a.add_argument("path")
    a.add_argument("question")
    a.add_argument("--mode", default="hybrid_rerank")
    a.add_argument("-k", type=int, default=3)
    args = ap.parse_args()

    if args.cmd == "serve":
        import uvicorn

        from ragmap.api import create_app

        uvicorn.run(create_app(preload=args.preload), host=args.host, port=args.port)
        return

    from ragmap.engine import Engine
    from ragmap.ingest import load_path

    doc = load_path(args.path)
    engine = Engine()
    chunks = engine.add_document(doc)
    if args.cmd == "map":
        from ragmap.knowledge_map import build_knowledge_map

        print(json.dumps(build_knowledge_map(doc, chunks, engine.embedder, args.max_concepts), indent=2))
    else:
        res = engine.search(args.question, k=args.k, mode=args.mode)
        for h in res["hits"]:
            print(f"[{h.rank}] {h.chunk.section} (p. {h.chunk.page})  score={h.score:.3f}\n{h.chunk.text}\n")
        print(f"timings: {res['timings']}")


if __name__ == "__main__":
    main()
