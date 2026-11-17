"""Turn PDFs (and plain text / markdown) into structured `Document`s.

PDF parsing uses PyMuPDF's span-level layout info to recover structure that plain
text extraction throws away:

* body font size is estimated as the most common span size (weighted by characters);
* lines that are noticeably larger or bold-and-short are treated as section headings;
* text that repeats at the same position on most pages (running headers/footers,
  page numbers) is dropped;
* hyphenated line breaks are re-joined.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from pathlib import Path

from ragmap.types import Document, Paragraph

_NUMBERED_HEADING = re.compile(r"^(\d+(\.\d+)*|[IVX]+\.|[A-Z]\.)\s+[A-Z][^.!?]{1,80}$")
_PAGE_NUM = re.compile(r"^(page\s*)?\d{1,4}(\s*/\s*\d+)?$", re.I)
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def doc_id_for(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()[:12]


def _clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)  # ligatures (ﬁ -> fi), full-width chars
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # de-hyphenate across line breaks
    text = text.replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def parse_pdf(data: bytes, title: str | None = None, doc_id: str | None = None) -> Document:
    import pymupdf as fitz

    pdf = fitz.open(stream=data, filetype="pdf")
    blocks: list[dict] = []  # {page, text, size, bold, y0, y1}
    size_counter: Counter[float] = Counter()

    for page_no, page in enumerate(pdf, start=1):
        height = page.rect.height or 1.0
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            lines_text, sizes, bold_chars, n_chars = [], [], 0, 0
            for line in block["lines"]:
                spans = [s for s in line["spans"] if s["text"].strip()]
                if not spans:
                    continue
                lines_text.append("".join(s["text"] for s in spans))
                for s in spans:
                    n = len(s["text"].strip())
                    size = round(s["size"], 1)
                    sizes.append((size, n))
                    size_counter[size] += n
                    n_chars += n
                    if s["flags"] & 16 or "bold" in s["font"].lower():
                        bold_chars += n
            if not lines_text:
                continue
            avg_size = sum(sz * n for sz, n in sizes) / max(1, sum(n for _, n in sizes))
            blocks.append(
                {
                    "page": page_no,
                    "text": "\n".join(lines_text),
                    "size": avg_size,
                    "bold": bold_chars / max(1, n_chars) > 0.6,
                    "y0": block["bbox"][1] / height,
                    "y1": block["bbox"][3] / height,
                }
            )

    if not blocks:
        raise ValueError("No extractable text found in PDF (is it scanned? OCR is not supported yet).")

    body_size = size_counter.most_common(1)[0][0]
    n_pages = len(pdf)

    # Running headers/footers: identical short text near the top/bottom on many pages.
    edge_texts = Counter(
        _clean(b["text"]).lower() for b in blocks if (b["y0"] < 0.08 or b["y1"] > 0.92) and len(b["text"]) < 120
    )
    repeated = {t for t, c in edge_texts.items() if n_pages >= 3 and c >= max(3, 0.5 * n_pages)}

    meta = pdf.metadata or {}
    doc_title = title or (meta.get("title") or "").strip()
    if not doc_title:
        first_page = [b for b in blocks if b["page"] == 1]
        if first_page:
            doc_title = _clean(max(first_page, key=lambda b: b["size"])["text"])[:200]

    paragraphs: list[Paragraph] = []
    section = ""
    for b in blocks:
        text = _clean(b["text"])
        if not text or text.lower() in repeated or _PAGE_NUM.match(text):
            continue
        if text == doc_title and b["page"] == 1:
            continue
        words = len(text.split())
        is_heading = words <= 14 and (
            b["size"] >= body_size * 1.12
            or (b["bold"] and not text.endswith("."))
            or bool(_NUMBERED_HEADING.match(text))
        )
        if is_heading:
            section = text
            continue
        paragraphs.append(Paragraph(text=text, section=section, page=b["page"]))

    paragraphs = _merge_fragments(paragraphs)
    return Document(
        doc_id=doc_id or doc_id_for(data),
        title=doc_title or "Untitled document",
        paragraphs=paragraphs,
        meta={"pages": n_pages, "source": "pdf"},
    )


def _merge_fragments(paragraphs: list[Paragraph], min_words: int = 8) -> list[Paragraph]:
    """PDF blocks often split one paragraph across columns/pages; glue tiny or unterminated
    fragments onto the following block when they share a section."""
    out: list[Paragraph] = []
    for p in paragraphs:
        if out and out[-1].section == p.section:
            prev = out[-1]
            unterminated = not prev.text.rstrip().endswith((".", "!", "?", ":", '"', ")"))
            if len(prev.text.split()) < min_words or (unterminated and p.text[:1].islower()):
                prev.text = f"{prev.text} {p.text}"
                continue
        out.append(Paragraph(text=p.text, section=p.section, page=p.page))
    return out


def parse_text(text: str, title: str | None = None, doc_id: str | None = None) -> Document:
    """Plain text / Markdown: blank lines separate paragraphs, `#` lines are headings."""
    paragraphs: list[Paragraph] = []
    section = ""
    for raw in re.split(r"\n\s*\n", text):
        raw = raw.strip()
        if not raw:
            continue
        lines = raw.splitlines()
        m = _MD_HEADING.match(lines[0])
        if m:
            section = m.group(2).strip()
            if not title and len(m.group(1)) == 1:
                title = section
            lines = lines[1:]
            if not lines:
                continue
        body = _clean("\n".join(lines))
        if body:
            paragraphs.append(Paragraph(text=body, section=section))
    data = text.encode()
    return Document(
        doc_id=doc_id or doc_id_for(data),
        title=title or (paragraphs[0].text[:80] if paragraphs else "Untitled document"),
        paragraphs=paragraphs,
        meta={"source": "text"},
    )


def load_path(path: str | Path) -> Document:
    path = Path(path)
    data = path.read_bytes()
    if path.suffix.lower() == ".pdf":
        return parse_pdf(data)
    return parse_text(data.decode("utf-8", errors="replace"), title=None)
