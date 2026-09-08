"""DocumentLoader: the length guard and the text cleaning.

Offline and fast — no models. PDFs are not exercised (that needs binary
fixtures); `.txt` covers the same guard and cleaning code paths, since both
loaders funnel through `_clean_text` and the same 50-character check.
"""
from ingest import DocumentLoader

GUARD = 50


def load(directory):
    return DocumentLoader(str(directory)).load()


def write(directory, name, text):
    (directory / name).write_text(text, encoding="utf-8")


# ------------------------------------------------------------- length guard

def test_exactly_at_the_guard_is_kept(tmp_path):
    """The check is `< 50`, so 50 characters is long enough."""
    write(tmp_path, "a.txt", "x" * GUARD)

    docs = load(tmp_path)

    assert len(docs) == 1
    assert len(docs[0]["text"]) == GUARD


def test_one_char_under_the_guard_is_dropped(tmp_path):
    write(tmp_path, "a.txt", "x" * (GUARD - 1))

    assert load(tmp_path) == []


def test_one_char_over_the_guard_is_kept(tmp_path):
    write(tmp_path, "a.txt", "x" * (GUARD + 1))

    assert len(load(tmp_path)) == 1


def test_the_guard_applies_after_cleaning_not_before(tmp_path):
    """Whitespace does not count toward the 50 characters.

    A file that is long only because it is padded with blank space is still a
    stub, and indexing it would put empty chunks into retrieval.
    """
    padded = "short text" + (" " * 200)
    assert len(padded) > GUARD, "raw text clears the guard"
    write(tmp_path, "a.txt", padded)

    assert load(tmp_path) == [], "but it is a stub once whitespace is collapsed"


def test_empty_file_is_dropped(tmp_path):
    write(tmp_path, "a.txt", "")

    assert load(tmp_path) == []


# ---------------------------------------------------------------- cleaning

def test_whitespace_is_collapsed_and_stripped(tmp_path):
    write(tmp_path, "a.txt", "  hello\n\n   world\t\tagain  " + "x" * GUARD)

    (doc,) = load(tmp_path)

    assert doc["text"].startswith("hello world again x")
    assert "\n" not in doc["text"] and "\t" not in doc["text"]
    assert "  " not in doc["text"]
    assert doc["text"] == doc["text"].strip()


# ---------------------------------------------------------------- metadata

def test_doc_id_is_the_filename_and_source_type_is_the_extension(tmp_path):
    write(tmp_path, "icd10.txt", "x" * GUARD)

    (doc,) = load(tmp_path)

    assert doc["doc_id"] == "icd10.txt"
    assert doc["source_type"] == "txt"
    assert set(doc) == {"doc_id", "text", "source_type"}


# ---------------------------------------------------------------- robustness

def test_unsupported_extensions_are_ignored(tmp_path):
    write(tmp_path, "notes.md", "x" * GUARD)
    write(tmp_path, "data.csv", "x" * GUARD)

    assert load(tmp_path) == []


def test_empty_directory_returns_empty_list(tmp_path):
    assert load(tmp_path) == []


def test_one_unreadable_file_does_not_lose_the_others(tmp_path):
    """A single bad file must not take down the whole corpus."""
    (tmp_path / "broken.txt").write_bytes(b"\xff\xfe\x00 invalid utf-8 " + b"x" * GUARD)
    write(tmp_path, "good.txt", "y" * GUARD)

    docs = load(tmp_path)

    assert [d["doc_id"] for d in docs] == ["good.txt"]


def test_reloading_the_same_loader_does_not_duplicate(tmp_path):
    """`load()` used to append to `self.documents` without clearing it.

    A reused loader therefore returned every document twice, and a third call
    three times — which would have silently tripled the corpus for any caller
    that did not build a fresh loader. `app.py` does build a fresh one, so this
    was latent rather than live.
    """
    write(tmp_path, "a.txt", "x" * GUARD)
    loader = DocumentLoader(str(tmp_path))

    assert len(loader.load()) == 1
    assert len(loader.load()) == 1
    assert len(loader.load()) == 1
