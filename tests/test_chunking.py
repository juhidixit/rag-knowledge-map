import pytest

from ragmap.chunking import chunk_document
from ragmap.types import Document, Paragraph


def test_markdown_sections_are_parsed(doc):
    assert doc.title == "Battery Thermal Management for Electric Vehicles"
    sections = {p.section for p in doc.paragraphs}
    assert {"Introduction", "Liquid Cooling", "Air Cooling", "Model Predictive Control"} <= sections


@pytest.mark.parametrize("strategy", ["fixed", "paragraph", "section"])
def test_chunks_respect_size_and_cover_document(doc, strategy):
    chunks = chunk_document(doc, strategy, max_words=50)
    assert chunks
    assert all(len(c.text.split()) <= 50 for c in chunks)
    # word offsets must agree with the chunk text
    assert all(c.word_end - c.word_start == len(c.text.split()) for c in chunks)
    covered = set()
    for c in chunks:
        covered.update(range(c.word_start, c.word_end))
    assert len(covered) == doc.n_words


def test_section_chunks_never_cross_sections(doc):
    for c in chunk_document(doc, "section", max_words=200):
        secs = {p.section for p in doc.paragraphs[c.par_start : c.par_end]}
        assert len(secs) == 1


def test_section_header_only_in_index_text(doc):
    c = chunk_document(doc, "section", max_words=200)[1]
    assert c.index_text.startswith(doc.title)
    assert not c.text.startswith(doc.title)


def test_long_paragraph_is_split():
    words = " ".join(f"w{i}" for i in range(450))
    d = Document("d", "t", [Paragraph(words, "S")])
    chunks = chunk_document(d, "section", max_words=200)
    assert [len(c.text.split()) for c in chunks] == [200, 200, 50]
    assert chunks[1].word_start == 200


def test_unknown_strategy(doc):
    with pytest.raises(ValueError):
        chunk_document(doc, "nope")
