"""Chunking strategies.

All strategies target a similar chunk size (``max_words``) so retrieval comparisons in
the eval harness are about *where the boundaries fall*, not about chunk length.

* ``fixed``     sliding window over the concatenated text, ignores document structure.
* ``paragraph`` one chunk per paragraph (long paragraphs are split, tiny ones merged).
* ``section``   structure-aware: packs whole paragraphs that belong to the same section
                up to ``max_words`` and prepends a contextual header
                ("<doc title> > <section>") to the *indexed* text only.
"""

from __future__ import annotations

from collections.abc import Callable

from ragmap.types import Chunk, Document

STRATEGIES = ("fixed", "paragraph", "section")


def _offsets(doc: Document) -> list[int]:
    offs, n = [], 0
    for p in doc.paragraphs:
        offs.append(n)
        n += len(p.text.split())
    offs.append(n)
    return offs


def _mk(doc: Document, idx: int, text: str, section: str, page, ps: int, pe: int, header: bool, ws: int = 0) -> Chunk:
    index_text = f"{doc.title} > {section}\n{text}" if header and section else (f"{doc.title}\n{text}" if header else text)
    return Chunk(
        chunk_id=f"{doc.doc_id}:{idx}",
        doc_id=doc.doc_id,
        text=text,
        section=section,
        page=page,
        par_start=ps,
        par_end=pe,
        word_start=ws,
        word_end=ws + len(text.split()),
        index_text=index_text,
    )


def chunk_fixed(doc: Document, max_words: int = 200, overlap: int = 40) -> list[Chunk]:
    words: list[str] = []
    owner: list[int] = []  # paragraph index for each word
    for i, p in enumerate(doc.paragraphs):
        w = p.text.split()
        words.extend(w)
        owner.extend([i] * len(w))
    step = max(1, max_words - overlap)
    chunks = []
    for n, start in enumerate(range(0, max(1, len(words)), step)):
        end = min(len(words), start + max_words)
        if start >= end:
            break
        ps, pe = owner[start], owner[end - 1] + 1
        p = doc.paragraphs[ps]
        chunks.append(_mk(doc, n, " ".join(words[start:end]), p.section, p.page, ps, pe, False, start))
        if end == len(words):
            break
    return chunks


def chunk_paragraph(doc: Document, max_words: int = 200, min_words: int = 40) -> list[Chunk]:
    offs = _offsets(doc)
    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_start = 0
    for i, p in enumerate(doc.paragraphs):
        w = p.text.split()
        if len(w) > max_words:  # split very long paragraphs
            if buf:
                q = doc.paragraphs[buf_start]
                chunks.append(_mk(doc, len(chunks), " ".join(buf), q.section, q.page, buf_start, i, False, offs[buf_start]))
                buf = []
            for s in range(0, len(w), max_words):
                chunks.append(_mk(doc, len(chunks), " ".join(w[s : s + max_words]), p.section, p.page, i, i + 1, False, offs[i] + s))
            buf_start = i + 1
            continue
        if not buf:
            buf_start = i
        buf.extend(w)
        if len(buf) >= min_words:
            q = doc.paragraphs[buf_start]
            chunks.append(_mk(doc, len(chunks), " ".join(buf), q.section, q.page, buf_start, i + 1, False, offs[buf_start]))
            buf = []
    if buf:
        q = doc.paragraphs[buf_start]
        chunks.append(_mk(doc, len(chunks), " ".join(buf), q.section, q.page, buf_start, len(doc.paragraphs), False, offs[buf_start]))
    return chunks


def chunk_section(doc: Document, max_words: int = 200, header: bool = True) -> list[Chunk]:
    offs = _offsets(doc)
    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_start = 0
    cur_section = None

    def flush(end: int) -> None:
        nonlocal buf
        if buf:
            q = doc.paragraphs[buf_start]
            chunks.append(_mk(doc, len(chunks), " ".join(buf), q.section, q.page, buf_start, end, header, offs[buf_start]))
        buf = []

    for i, p in enumerate(doc.paragraphs):
        w = p.text.split()
        if p.section != cur_section or (buf and len(buf) + len(w) > max_words):
            flush(i)
            cur_section = p.section
            buf_start = i
        if len(w) > max_words:
            flush(i)
            for s in range(0, len(w), max_words):
                chunks.append(_mk(doc, len(chunks), " ".join(w[s : s + max_words]), p.section, p.page, i, i + 1, header, offs[i] + s))
            buf_start = i + 1
            continue
        if not buf:
            buf_start = i
        buf.extend(w)
    flush(len(doc.paragraphs))
    return chunks


CHUNKERS: dict[str, Callable[..., list[Chunk]]] = {
    "fixed": chunk_fixed,
    "paragraph": chunk_paragraph,
    "section": chunk_section,
}


def chunk_document(doc: Document, strategy: str = "section", **kwargs) -> list[Chunk]:
    if strategy not in CHUNKERS:
        raise ValueError(f"Unknown chunking strategy {strategy!r}; choose from {STRATEGIES}")
    return [c for c in CHUNKERS[strategy](doc, **kwargs) if c.text.strip()]
