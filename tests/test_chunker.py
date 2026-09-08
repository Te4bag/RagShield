"""chunk_documents: id format, metadata propagation, size and overlap.

`chunk_documents` reads `cfg['ingestion']` at call time, so each test sets the
size and overlap it needs rather than depending on config.yaml.
"""
import index.config_loader as cl
import pytest

from index.chunker import chunk_documents


@pytest.fixture
def sized(monkeypatch):
    """Set chunk_size / chunk_overlap for one test."""
    def _set(chunk_size, chunk_overlap):
        monkeypatch.setitem(cl.cfg, "ingestion",
                            {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap})
    return _set


def doc(doc_id="d.txt", text="body", source_type="txt"):
    return {"doc_id": doc_id, "text": text, "source_type": source_type}


def tokens(n, start=0):
    """Space-separated unique tokens, so chunk boundaries are legible."""
    return " ".join(f"w{i:03d}" for i in range(start, start + n))


# ------------------------------------------------------------------ chunk_id

def test_chunk_id_format_and_sequence(sized):
    sized(100, 20)

    chunks = chunk_documents([doc(text=tokens(80))])

    assert len(chunks) > 1
    assert [c["chunk_id"] for c in chunks] == [
        f"d.txt_ch{i}" for i in range(len(chunks))]


def test_the_counter_restarts_per_document(sized):
    """Ids are scoped to their document, not to the batch."""
    sized(100, 20)

    chunks = chunk_documents([doc("a.txt", tokens(40)), doc("b.txt", tokens(40))])

    a = [c["chunk_id"] for c in chunks if c["doc_id"] == "a.txt"]
    b = [c["chunk_id"] for c in chunks if c["doc_id"] == "b.txt"]
    assert a[0] == "a.txt_ch0"
    assert b[0] == "b.txt_ch0"


def test_chunk_ids_are_unique_across_documents(sized):
    sized(100, 20)

    chunks = chunk_documents([doc("a.txt", tokens(60)), doc("b.txt", tokens(60))])

    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids)), "chunk_id is the Chroma primary key"


# ----------------------------------------------------------------- metadata

def test_every_chunk_carries_its_document_metadata(sized):
    sized(100, 20)

    chunks = chunk_documents([doc("icd10.txt", tokens(60), "txt")])

    assert chunks
    for c in chunks:
        assert c["doc_id"] == "icd10.txt"
        assert c["source_type"] == "txt"
        assert set(c) == {"chunk_id", "doc_id", "text", "source_type"}


def test_source_type_is_preserved_per_document(sized):
    sized(100, 20)

    chunks = chunk_documents([doc("a.pdf", tokens(30), "pdf"),
                              doc("b.txt", tokens(30), "txt")])

    by_doc = {c["doc_id"]: c["source_type"] for c in chunks}
    assert by_doc == {"a.pdf": "pdf", "b.txt": "txt"}


# ------------------------------------------------------------ size / overlap

def test_chunks_respect_the_configured_size(sized):
    sized(100, 20)

    chunks = chunk_documents([doc(text=tokens(120))])

    assert max(len(c["text"]) for c in chunks) <= 100


def test_adjacent_chunks_overlap(sized):
    """The overlap is what stops a claim being split away from its context."""
    sized(100, 20)

    chunks = chunk_documents([doc(text=tokens(80))])

    assert len(chunks) >= 2
    first, second = set(chunks[0]["text"].split()), set(chunks[1]["text"].split())
    assert first & second, "no shared tokens between adjacent chunks"


def test_zero_overlap_produces_disjoint_chunks(sized):
    sized(100, 0)

    chunks = chunk_documents([doc(text=tokens(80))])

    assert len(chunks) >= 2
    first, second = set(chunks[0]["text"].split()), set(chunks[1]["text"].split())
    assert not (first & second)


def test_more_overlap_yields_more_chunks(sized, monkeypatch):
    """A direct check that the setting is actually plumbed through."""
    text = tokens(120)

    sized(100, 0)
    few = len(chunk_documents([doc(text=text)]))
    sized(100, 40)
    many = len(chunk_documents([doc(text=text)]))

    assert many > few


# ------------------------------------------------------------------ content

def test_a_short_document_is_a_single_chunk(sized):
    sized(600, 60)

    chunks = chunk_documents([doc(text="one short sentence")])

    assert len(chunks) == 1
    assert chunks[0]["text"] == "one short sentence"
    assert chunks[0]["chunk_id"] == "d.txt_ch0"


def test_no_content_is_dropped(sized):
    """Every token must survive somewhere, or retrieval cannot find it."""
    sized(100, 20)
    text = tokens(120)

    chunks = chunk_documents([doc(text=text)])

    seen = set()
    for c in chunks:
        seen.update(c["text"].split())
    assert seen == set(text.split())


def test_empty_document_list_returns_empty(sized):
    sized(600, 60)

    assert chunk_documents([]) == []
