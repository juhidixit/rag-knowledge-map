"""Core data structures shared across ingestion, chunking, retrieval and the knowledge map."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Paragraph:
    text: str
    section: str = ""
    page: int | None = None


@dataclass
class Document:
    doc_id: str
    title: str
    paragraphs: list[Paragraph]
    meta: dict = field(default_factory=dict)

    @property
    def n_words(self) -> int:
        return sum(len(p.text.split()) for p in self.paragraphs)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    section: str = ""
    page: int | None = None
    # Index range [par_start, par_end) of the source paragraphs this chunk covers.
    par_start: int = 0
    par_end: int = 0
    # Global word offsets [word_start, word_end) of the chunk within the document.
    word_start: int = 0
    word_end: int = 0
    # Text that is indexed. May include a contextual header (doc title + section path)
    # while `text` stays the clean passage shown to users.
    index_text: str = ""

    def __post_init__(self) -> None:
        if not self.index_text:
            self.index_text = self.text

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("index_text")
        return d


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int
    scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"rank": self.rank, "score": round(float(self.score), 5), "scores": self.scores, **self.chunk.to_dict()}
