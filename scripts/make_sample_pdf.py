"""Render one QASPER paper's full text into a real, multi-page PDF.

Used for the demo and for the end-to-end PDF ingestion test: it exercises heading detection
(font-size based), page tracking and paragraph reconstruction on an actual PDF file.

    python scripts/make_sample_pdf.py data/qasper-dev-v0.3.json 1611.01576 samples/qrnn.pdf
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import pymupdf as fitz

CSS = """
body { font-family: serif; font-size: 10.5pt; line-height: 1.35; }
h1 { font-size: 17pt; text-align: center; margin-bottom: 4pt; }
h2 { font-size: 13pt; margin-top: 10pt; }
h3 { font-size: 11.5pt; margin-top: 8pt; }
p { text-align: justify; margin: 0 0 6pt 0; }
.abstract { font-style: italic; }
"""


def paper_html(paper: dict) -> str:
    parts = [
        f"<h1>{html.escape(paper['title'])}</h1>",
        "<h2>Abstract</h2>",
        f"<p class='abstract'>{html.escape(paper['abstract'])}</p>",
    ]
    for sec in paper["full_text"]:
        names = (sec.get("section_name") or "").split(" ::: ")
        tag = "h2" if len(names) == 1 else "h3"
        parts.append(f"<{tag}>{html.escape(names[-1])}</{tag}>")
        parts.extend(f"<p>{html.escape(p)}</p>" for p in sec["paragraphs"] if p.strip())
    return "\n".join(parts)


def render(paper: dict, out: Path) -> None:
    story = fitz.Story(html=paper_html(paper), user_css=CSS)
    writer = fitz.DocumentWriter(str(out))
    page_rect = fitz.paper_rect("letter")
    content = page_rect + (60, 60, -60, -60)
    more = True
    while more:
        dev = writer.begin_page(page_rect)
        more, _ = story.place(content)
        story.draw(dev)
        writer.end_page()
    writer.close()
    # Add page numbers + a running header so the parser's header/footer filter is exercised.
    doc = fitz.open(str(out))
    for i, page in enumerate(doc, start=1):
        page.insert_text((page.rect.width / 2 - 6, page.rect.height - 30), str(i), fontsize=9)
        page.insert_text((60, 36), paper["title"][:80], fontsize=8, color=(0.4, 0.4, 0.4))
    doc.metadata["title"] = paper["title"]
    doc.set_metadata({"title": paper["title"]})
    doc.saveIncr()


if __name__ == "__main__":
    src, pid, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    out.parent.mkdir(parents=True, exist_ok=True)
    render(json.loads(Path(src).read_text())[pid], out)
    print(f"wrote {out}")
