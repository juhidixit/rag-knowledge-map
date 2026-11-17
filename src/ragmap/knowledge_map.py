"""Turn a document into a knowledge map: key concepts, how they co-occur, and topic clusters.

Every node keeps pointers to the chunks it was found in, so the UI can jump from a concept
to the exact source passages (and the retrieval engine can light up the concepts that
support an answer).

Pipeline
1. Candidate phrases: 1-4 word n-grams inside stop-word-free runs of each chunk
   (RAKE-style), plus ALL-CAPS acronyms. Stemmed keys merge "language models" and
   "language model".
2. Scoring: tf * idf over the document's own chunks (concepts concentrated in a few
   passages are more specific than ones spread everywhere), a bonus for multi-word
   phrases, and - when an embedder is available - KeyBERT-style similarity to the
   document centroid so that off-topic boilerplate sinks.
3. Redundancy pruning: a phrase is dropped when a longer selected phrase accounts for
   most of its occurrences ("machine translation" vs "neural machine translation").
4. Edges: chunk-level co-occurrence weighted by normalised PMI, capped per node.
5. Topics: Louvain community detection on the weighted graph.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import networkx as nx
import numpy as np

from ragmap.retrieval import _STOP
from ragmap.types import Chunk, Document

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\-]*[A-Za-z0-9]|[A-Za-z]")
_SPLIT = re.compile(r"['’]s\b|[.,;:!?()\[\]{}\"“”'’`/|<>=+*&%$#@~^]|\s-\s|\d+(?:\.\d+)?")

# Words that are frequent in any technical document but are rarely concepts on their own.
GENERIC = set(
    """
    paper work approach method methods result results table figure fig section et al use used using
    based show shows shown propose proposed present presented study studies example examples
    different number set sets case cases model models task tasks data performance new first second
    third previous following given also well however thus may can could would one two three four
    five several many various large small high low best better good much more less like due
    order way respectively include includes including total average value values provide provides
    obtain obtained significant significantly important particular general specific similar
    original final main overall section appendix experiment experiments evaluation evaluate
    evaluated baseline baselines approaches compare compared comparison achieve achieves achieved
    improvement improve improves improved state art time times effect effects term terms type
    types level levels list lists part parts problem problems process note see within across
    """.split()
)


_NOISE = re.compile(
    r"^(BIBREF|TABREF|FIGREF|SECREF|EQREF|INLINEFORM|DISPLAYFORM)\d*$|^(left|right|leftright)?arrow$|^(math|text)(bf|rm|it|cal)$",
    re.I,
)

try:  # word-frequency prior: very common English words are rarely concepts on their own
    from wordfreq import zipf_frequency
except ImportError:  # pragma: no cover - optional dependency
    zipf_frequency = None


def _specificity(word: str) -> float:
    if zipf_frequency is None or (word.isupper() and len(word) > 1):
        return 1.0
    return min(1.0, max(0.15, (6.0 - zipf_frequency(word.lower(), "en")) / 3.0))


def _stem(w: str) -> str:
    """Plural folding only - conservative so that labels stay readable."""
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith(("ses", "xes", "ches", "shes")):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def _norm_key(words: list[str]) -> str:
    return " ".join(_stem(w.lower()) for w in words)


def _candidates(text: str, max_n: int = 4) -> list[tuple[str, str]]:
    """Return (key, surface) candidate phrases for one passage."""
    out = []
    for run in _SPLIT.split(text):
        words = _WORD.findall(run)
        seg: list[str] = []
        segments = []
        for w in words:
            if w.lower() in _STOP or (w.lower() in GENERIC and not w.isupper()) or _NOISE.match(w):
                if seg:
                    segments.append(seg)
                seg = []
            else:
                seg.append(w)
        if seg:
            segments.append(seg)
        for seg in segments:
            for n in range(1, min(max_n, len(seg)) + 1):
                for i in range(len(seg) - n + 1):
                    gram = seg[i : i + n]
                    surface = " ".join(gram)
                    if n == 1:
                        w = gram[0]
                        is_acronym = w.isupper() and 2 <= len(w) <= 8
                        if not is_acronym and len(w) < 4:
                            continue
                    out.append((_norm_key(gram), surface))
    return out


_NLP = None


def _spacy():
    """spaCy noun-chunk extraction if `en_core_web_sm` is installed, else None (regex fallback)."""
    global _NLP
    if _NLP is None:
        try:
            import spacy

            _NLP = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
        except Exception:
            _NLP = False
    return _NLP or None


def _candidates_nlp(texts: list[str], max_n: int = 4) -> list[list[tuple[str, str]]]:
    nlp = _spacy()
    out = []
    for doc in nlp.pipe(texts, batch_size=32):
        cands = []
        for nc in doc.noun_chunks:
            toks = [t for t in nc if not (t.is_stop or t.is_punct or t.like_num or t.pos_ in {"DET", "PRON", "NUM"})]
            toks = [t for t in toks if not _NOISE.match(t.text) and _WORD.fullmatch(t.text)]
            while toks and toks[0].text.lower() in GENERIC:
                toks = toks[1:]
            if not toks or toks[-1].pos_ not in {"NOUN", "PROPN"} and not toks[-1].text.isupper():
                continue
            toks = toks[-max_n:]
            # every suffix ending at the head noun: "optimized cuDNN LSTM", "cuDNN LSTM", "LSTM"
            for i in range(len(toks)):
                gram = [t.text for t in toks[i:]]
                surface = doc[toks[i].i : toks[-1].i + 1].text  # keeps hyphens: "Lithium-ion"
                if len(surface.split()) > max_n + 1:
                    surface = " ".join(gram)
                if len(gram) == 1:
                    w = gram[0]
                    if w.lower() in GENERIC or (not (w.isupper() and len(w) >= 2) and len(w) < 4):
                        continue
                if gram[0].lower() in GENERIC:
                    continue
                cands.append((_norm_key(_WORD.findall(surface)), surface))
        out.append(cands)
    return out


def extract_concepts(
    doc: Document, chunks: list[Chunk], embedder=None, max_concepts: int = 60, min_count: int = 2
) -> list[dict]:
    n_chunks = max(1, len(chunks))
    tf: Counter[str] = Counter()
    df: Counter[str] = Counter()
    surfaces: dict[str, Counter[str]] = defaultdict(Counter)
    where: dict[str, list[int]] = defaultdict(list)
    per_chunk = _candidates_nlp([c.text for c in chunks]) if _spacy() else [_candidates(c.text) for c in chunks]
    for ci, cands in enumerate(per_chunk):
        seen = set()
        for key, surface in cands:
            tf[key] += 1
            surfaces[key][surface] += 1
            if key not in seen:
                df[key] += 1
                where[key].append(ci)
                seen.add(key)

    if sum(tf.values()) < 400:  # very short documents: accept singletons
        min_count = 1

    def label_of(k: str) -> str:
        # Prefer the most frequent surface form; keep acronyms upper-case.
        return surfaces[k].most_common(1)[0][0]

    # Phrases found in most passages are usually filler, unless they are specific terms
    # (the paper's own acronym or method name appears everywhere and must stay).
    keys = [
        k
        for k, v in tf.items()
        if v >= min_count
        and (df[k] < max(3, 0.6 * n_chunks) or max(_specificity(w) for w in label_of(k).split()) >= 0.8)
    ]
    if not keys:
        return []

    score = {}
    for k in keys:
        n_words = k.count(" ") + 1
        idf = math.log(1 + n_chunks / df[k])
        label = label_of(k)
        spec = max(_specificity(w) for w in label.split())
        score[k] = tf[k] * idf * (1 + 0.35 * (n_words - 1)) * (spec**1.5 if n_words == 1 else spec**0.5)
        if label.isupper():
            score[k] *= 1.3

    # Keep a generous shortlist, then refine with semantic similarity to the document.
    shortlist = sorted(keys, key=lambda k: -score[k])[: max_concepts * 4]
    if embedder is not None and shortlist:
        labels = [label_of(k) for k in shortlist]
        centroid = embedder.encode([c.text for c in chunks[:64]]).mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-9
        sims = embedder.encode(labels) @ centroid
        for k, s in zip(shortlist, sims, strict=True):
            score[k] *= 0.5 + max(0.0, float(s))

    ranked = sorted(shortlist, key=lambda k: -score[k])
    selected: list[str] = []
    for k in ranked:
        words = k.split()
        redundant = False
        for j, s in enumerate(selected):
            sw = s.split()
            if f" {k} " in f" {s} " and tf[s] >= 0.7 * tf[k]:
                redundant = True  # a longer selected phrase already covers this one
                break
            if f" {s} " in f" {k} " and len(words) > len(sw):
                if tf[k] >= 0.7 * tf[s]:
                    selected[j] = k  # upgrade to the more specific phrase
                redundant = True  # otherwise it is a minor variant of an existing concept
                break
        if not redundant:
            selected.append(k)
        if len(selected) >= max_concepts:
            break
    selected = list(dict.fromkeys(selected))

    out = []
    for k in selected:
        cids = where[k]
        pages = sorted({chunks[i].page for i in cids if chunks[i].page is not None})
        sections = [s for s, _ in Counter(chunks[i].section for i in cids if chunks[i].section).most_common(3)]
        out.append(
            {
                "key": k,
                "label": label_of(k),
                "score": round(score[k], 3),
                "count": tf[k],
                "chunk_idx": cids,
                "pages": pages,
                "sections": sections,
            }
        )
    return out


def build_knowledge_map(
    doc: Document, chunks: list[Chunk], embedder=None, max_concepts: int = 60, max_edges_per_node: int = 6
) -> dict:
    concepts = extract_concepts(doc, chunks, embedder=embedder, max_concepts=max_concepts)
    n = max(1, len(chunks))
    G = nx.Graph()
    G.add_nodes_from(range(len(concepts)))

    sets = [set(c["chunk_idx"]) for c in concepts]
    cand_edges: dict[int, list[tuple[float, int, int]]] = defaultdict(list)
    for i in range(len(concepts)):
        for j in range(i + 1, len(concepts)):
            co = len(sets[i] & sets[j])
            if co == 0:
                continue
            p_ij = co / n
            p_i, p_j = len(sets[i]) / n, len(sets[j]) / n
            pmi = math.log(p_ij / (p_i * p_j))
            npmi = pmi / -math.log(p_ij) if p_ij < 1 else 1.0
            if npmi <= 0:
                continue
            w = npmi * math.log1p(co)
            cand_edges[i].append((w, j, co))
            cand_edges[j].append((w, i, co))
    for i, lst in cand_edges.items():
        for w, j, co in sorted(lst, reverse=True)[:max_edges_per_node]:
            if G.has_edge(i, j):
                continue
            G.add_edge(i, j, weight=round(w, 4), co=co)

    communities: list[set[int]] = []
    if G.number_of_edges():
        communities = [set(c) for c in nx.community.louvain_communities(G, weight="weight", seed=7)]
    else:
        communities = [{i} for i in G.nodes]
    communities.sort(key=lambda c: -sum(concepts[i]["score"] for i in c))
    comm_of = {i: ci for ci, c in enumerate(communities) for i in c}

    chunk_ids = [c.chunk_id for c in chunks]
    nodes = []
    for i, c in enumerate(concepts):
        nodes.append(
            {
                "id": i,
                "label": c["label"],
                "score": c["score"],
                "count": c["count"],
                "community": comm_of.get(i, 0),
                "degree": G.degree(i),
                "chunks": [chunk_ids[k] for k in c["chunk_idx"]],
                "pages": c["pages"],
                "sections": c["sections"],
            }
        )
    links = [{"source": u, "target": v, "weight": d["weight"], "co": d["co"]} for u, v, d in G.edges(data=True)]
    topics = []
    for ci, members in enumerate(communities):
        top = sorted(members, key=lambda i: -concepts[i]["score"])
        chunk_cover = set().union(*(sets[i] for i in members)) if members else set()
        sec = Counter(chunks[k].section for k in chunk_cover if chunks[k].section).most_common(2)
        topics.append(
            {
                "id": ci,
                "label": " · ".join(concepts[i]["label"] for i in top[:2]),
                "size": len(members),
                "concepts": [concepts[i]["label"] for i in top[:8]],
                "sections": [s for s, _ in sec],
            }
        )
    return {
        "doc_id": doc.doc_id,
        "title": doc.title,
        "nodes": nodes,
        "links": links,
        "topics": topics,
        "outline": _outline(doc),
        "stats": {
            "paragraphs": len(doc.paragraphs),
            "chunks": len(chunks),
            "words": doc.n_words,
            "concepts": len(nodes),
            "relations": len(links),
            "pages": doc.meta.get("pages"),
        },
    }


def _outline(doc: Document) -> list[dict]:
    out: list[dict] = []
    for p in doc.paragraphs:
        if not out or out[-1]["section"] != p.section:
            out.append({"section": p.section or "(untitled)", "page": p.page, "paragraphs": 0})
        out[-1]["paragraphs"] += 1
    return out
