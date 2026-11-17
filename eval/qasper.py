"""Load QASPER (QA over full-text NLP papers from arXiv) as a retrieval benchmark.

QASPER: Dasigi et al., 2021, "A Dataset of Information-Seeking Questions and Answers
Anchored in Research Papers". Each question was written by someone who had only read the
title + abstract, and annotators marked the *evidence paragraphs* in the full text that
answer it. That gives us graded passage-level relevance labels on real arXiv papers.

Download:  https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ragmap.types import Document, Paragraph


@dataclass
class Question:
    qid: str
    doc_id: str
    text: str
    # Each evidence set is the list of paragraph indices one annotator marked.
    evidence_sets: list[set[int]] = field(default_factory=list)

    @property
    def evidence(self) -> set[int]:
        out: set[int] = set()
        for s in self.evidence_sets:
            out |= s
        return out


def _norm(s: str) -> str:
    return " ".join(s.split()).strip()


def load_qasper(path: str | Path) -> tuple[list[Document], list[Question]]:
    raw = json.loads(Path(path).read_text())
    docs: list[Document] = []
    questions: list[Question] = []
    for pid, paper in raw.items():
        paragraphs = [Paragraph(text=_norm(paper["abstract"]), section="Abstract")]
        for sec in paper["full_text"]:
            name = (sec.get("section_name") or "").replace(" ::: ", " > ")
            for p in sec["paragraphs"]:
                p = _norm(p)
                if p:
                    paragraphs.append(Paragraph(text=p, section=name))
        index = {}
        for i, p in enumerate(paragraphs):
            index.setdefault(p.text, i)
        docs.append(Document(doc_id=pid, title=_norm(paper["title"]), paragraphs=paragraphs))

        for qa in paper["qas"]:
            sets = []
            for ans in qa["answers"]:
                a = ans["answer"]
                if a.get("unanswerable"):
                    continue
                ev = {index[_norm(e)] for e in a.get("evidence", []) if _norm(e) in index}
                # skip evidence that only points at figures/tables ("FLOAT SELECTED: ...")
                if ev:
                    sets.append(ev)
            if sets:
                questions.append(Question(qid=qa["question_id"], doc_id=pid, text=qa["question"], evidence_sets=sets))
    return docs, questions
