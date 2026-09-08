"""Incremental indexing: what gets re-embedded, and what is left alone.

The decision lives in `plan_sync`, a pure function, so most of this is tested
with no database at all. `sync_documents` is exercised against a stub
collection — the point is which calls it makes, not what Chroma does with them,
and the suite stays offline with no embedding model loaded.
"""
import index.config_loader as cl
import pytest

from index.vector_store import RagShieldIndex, document_hash, plan_sync


def doc(doc_id, text, source_type="txt"):
    return {"doc_id": doc_id, "text": text, "source_type": source_type}


# ------------------------------------------------------------ document_hash

def test_hash_is_stable_for_identical_text():
    assert document_hash("the same text") == document_hash("the same text")


def test_hash_changes_when_text_changes():
    assert document_hash("original text") != document_hash("original text.")


def test_hash_handles_non_ascii():
    """Extracted PDF text is full of ligatures, dashes and quotes."""
    assert document_hash("café — “quoted”") == document_hash("café — “quoted”")


# ---------------------------------------------------------------- plan_sync

def test_new_document_is_added():
    plan = plan_sync({}, {"a.txt": "h1"})

    assert plan == {"added": ["a.txt"], "updated": [], "unchanged": [], "removed": []}


def test_unchanged_document_is_left_alone():
    """The whole point: an untouched corpus is not re-embedded."""
    plan = plan_sync({"a.txt": "h1"}, {"a.txt": "h1"})

    assert plan["unchanged"] == ["a.txt"]
    assert plan["added"] == [] and plan["updated"] == [] and plan["removed"] == []


def test_changed_document_is_updated_not_added():
    plan = plan_sync({"a.txt": "h1"}, {"a.txt": "h2"})

    assert plan["updated"] == ["a.txt"]
    assert plan["added"] == []


def test_deleted_document_is_removed():
    plan = plan_sync({"a.txt": "h1", "b.txt": "h2"}, {"a.txt": "h1"})

    assert plan["removed"] == ["b.txt"]
    assert plan["unchanged"] == ["a.txt"]


def test_a_realistic_mixed_batch():
    plan = plan_sync(
        {"keep.txt": "h1", "edit.txt": "h2", "gone.txt": "h3"},
        {"keep.txt": "h1", "edit.txt": "CHANGED", "new.txt": "h4"},
    )

    assert plan == {"added": ["new.txt"], "updated": ["edit.txt"],
                    "unchanged": ["keep.txt"], "removed": ["gone.txt"]}


def test_every_document_lands_in_exactly_one_bucket():
    existing = {"a": "1", "b": "2", "c": "3"}
    incoming = {"a": "1", "b": "CHANGED", "d": "4"}

    plan = plan_sync(existing, incoming)
    buckets = plan["added"] + plan["updated"] + plan["unchanged"] + plan["removed"]

    assert sorted(buckets) == sorted(set(existing) | set(incoming))
    assert len(buckets) == len(set(buckets))


def test_empty_incoming_removes_everything():
    plan = plan_sync({"a.txt": "h1", "b.txt": "h2"}, {})

    assert plan["removed"] == ["a.txt", "b.txt"]


def test_missing_hash_counts_as_changed():
    """A collection written before doc_hash existed re-indexes once.

    Older chunks report an empty hash, which never matches a real one, so they
    are replaced on the next sync and are correct from then on.
    """
    plan = plan_sync({"a.txt": ""}, {"a.txt": document_hash("body")})

    assert plan["updated"] == ["a.txt"]


# ----------------------------------------------------------- sync_documents

class StubCollection:
    """Records the calls `sync_documents` makes, without a database."""

    def __init__(self, metadatas=None):
        self.metadatas = list(metadatas or [])
        self.deleted = []
        self.added = []

    def get(self, include=None):
        return {"metadatas": self.metadatas}

    def delete(self, where=None):
        self.deleted.append(where["doc_id"])

    def add(self, ids, documents, metadatas):
        self.added.append({"ids": ids, "metadatas": metadatas})


@pytest.fixture
def index(monkeypatch):
    """A RagShieldIndex with a stub collection and no embedding model."""
    monkeypatch.setitem(cl.cfg, "ingestion", {"chunk_size": 600, "chunk_overlap": 60})

    def _make(metadatas=None):
        obj = object.__new__(RagShieldIndex)   # skip __init__: no model, no client
        obj.collection = StubCollection(metadatas)
        return obj
    return _make


def test_unchanged_document_triggers_no_embedding(index):
    body = "a body long enough to survive chunking without being split"
    h = document_hash(body)
    idx = index([{"doc_id": "a.txt", "chunk_id": "a.txt_ch0",
                  "source_type": "txt", "doc_hash": h}])

    plan = idx.sync_documents([doc("a.txt", body)])

    assert plan["unchanged"] == ["a.txt"]
    assert idx.collection.added == [], "an unchanged document was re-embedded"
    assert idx.collection.deleted == []


def test_only_the_new_document_is_embedded(index):
    """The scenario P5 exists for: adding one file to an indexed corpus."""
    old = "the first document, already indexed and unchanged since"
    idx = index([{"doc_id": "a.txt", "chunk_id": "a.txt_ch0",
                  "source_type": "txt", "doc_hash": document_hash(old)}])

    plan = idx.sync_documents([doc("a.txt", old), doc("b.txt", "a brand new second document")])

    assert plan["added"] == ["b.txt"] and plan["unchanged"] == ["a.txt"]
    (call,) = idx.collection.added
    assert {m["doc_id"] for m in call["metadatas"]} == {"b.txt"}


def test_changed_document_is_deleted_before_being_re_added(index):
    """Re-chunking can yield a different number of pieces.

    Updating in place would leave orphaned chunks from the longer old version,
    so the document is dropped wholesale first.
    """
    idx = index([{"doc_id": "a.txt", "chunk_id": "a.txt_ch0",
                  "source_type": "txt", "doc_hash": "stale"}])

    plan = idx.sync_documents([doc("a.txt", "the text has since been edited")])

    assert plan["updated"] == ["a.txt"]
    assert idx.collection.deleted == ["a.txt"]
    assert idx.collection.added


def test_removed_document_is_deleted_and_nothing_is_embedded(index):
    idx = index([{"doc_id": "gone.txt", "chunk_id": "gone.txt_ch0",
                  "source_type": "txt", "doc_hash": "h"}])

    plan = idx.sync_documents([])

    assert plan["removed"] == ["gone.txt"]
    assert idx.collection.deleted == ["gone.txt"]
    assert idx.collection.added == []


def test_the_stored_hash_matches_the_document(index):
    """Otherwise the next sync re-embeds everything, every time."""
    body = "a document whose hash must round-trip into the metadata"
    idx = index()

    idx.sync_documents([doc("a.txt", body)])

    (call,) = idx.collection.added
    assert {m["doc_hash"] for m in call["metadatas"]} == {document_hash(body)}


def test_sync_is_idempotent(index):
    """Running twice must embed once — the guard against silent rebuild loops."""
    body = "a document that does not change between the two runs"
    idx = index()

    idx.sync_documents([doc("a.txt", body)])
    embedded_metadatas = idx.collection.added[0]["metadatas"]
    idx.collection.metadatas = embedded_metadatas      # what Chroma would hold now

    second = idx.sync_documents([doc("a.txt", body)])

    assert second["unchanged"] == ["a.txt"]
    assert len(idx.collection.added) == 1, "second sync re-embedded an unchanged document"
