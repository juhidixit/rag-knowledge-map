"""Download the demo library for the Hugging Face Space, keeping only CC BY 4.0 papers.

Reads arXiv IDs from space/papers.txt, looks up each paper's license in arXiv's OAI-PMH metadata,
downloads the PDFs of the first N that are CC BY 4.0 into space/library/, and writes
space/papers.json with an attribution line per paper. Runs in the deploy workflow (needs arxiv.org).
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

OAI = "https://export.arxiv.org/oai2?verb=GetRecord&identifier=oai:arXiv.org:{id}&metadataPrefix=arXiv"
PDF = "https://arxiv.org/pdf/{id}"
NS = {"oai": "http://www.openarchives.org/OAI/2.0/", "ax": "http://arxiv.org/OAI/arXiv/"}
CC_BY = {"http://creativecommons.org/licenses/by/4.0/", "https://creativecommons.org/licenses/by/4.0/"}
UA = {"User-Agent": "ragmap-demo-deploy/0.1 (+https://github.com/juhidixit/rag-knowledge-map)"}


def read_ids(path: Path) -> list[str]:
    ids = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.append(line)
    return ids


def parse_record(xml: bytes) -> dict:
    root = ET.fromstring(xml)
    meta = root.find(".//ax:arXiv", NS)
    if meta is None:
        raise ValueError("no arXiv metadata in response")

    def text(tag: str) -> str:
        el = meta.find(f"ax:{tag}", NS)
        return " ".join((el.text or "").split()) if el is not None else ""

    authors = []
    for a in meta.findall("ax:authors/ax:author", NS):
        parts = [a.findtext("ax:forenames", "", NS), a.findtext("ax:keyname", "", NS)]
        authors.append(" ".join(p.strip() for p in parts if p and p.strip()))
    return {"id": text("id"), "title": text("title"), "authors": authors, "license": text("license")}


def attribution(rec: dict) -> str:
    names = rec["authors"]
    who = ", ".join(names[:3]) + (" et al." if len(names) > 3 else "")
    return (
        f'"{rec["title"]}" by {who}, arXiv:{rec["id"]} (https://arxiv.org/abs/{rec["id"]}), '
        f"licensed CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). PDF redistributed unmodified."
    )


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="space/papers.txt")
    ap.add_argument("--out", default="space/library")
    ap.add_argument("--meta", default="space/papers.json")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kept: list[dict] = []
    for arxiv_id in read_ids(Path(args.ids)):
        if len(kept) >= args.n:
            break
        time.sleep(3)  # arXiv asks automated clients for one request every 3 seconds
        try:
            rec = parse_record(fetch(OAI.format(id=arxiv_id)))
        except Exception as exc:
            print(f"skip {arxiv_id}: metadata lookup failed ({exc})")
            continue
        if rec["license"] not in CC_BY:
            print(f"skip {arxiv_id}: license is {rec['license'] or 'arXiv default (not CC BY)'}")
            continue
        time.sleep(3)
        pdf = fetch(PDF.format(id=arxiv_id))
        if not pdf.startswith(b"%PDF"):
            print(f"skip {arxiv_id}: download is not a PDF")
            continue
        (out / f"{arxiv_id}.pdf").write_bytes(pdf)
        rec["attribution"] = attribution(rec)
        kept.append(rec)
        print(f"keep {arxiv_id}: {rec['title']} ({len(pdf) // 1024} KB)")

    if not kept:
        raise SystemExit("No CC BY 4.0 papers found; add candidates to space/papers.txt")
    Path(args.meta).write_text(json.dumps(kept, indent=2) + "\n")


if __name__ == "__main__":
    main()
