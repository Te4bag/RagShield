"""What the UI lists must be exactly what the loader ingests.

These two used to disagree: the loader read `.pdf` and `.txt`, while the
uploader accepted only `pdf` and both listings globbed `*.pdf`. A `.txt` was
therefore indexed but never shown, and if it was the only file present the app
reported "No documents found" over a corpus it had just indexed.

Offline: no models, no network. PDFs are not exercised here (that needs fitz
fixtures and belongs with the loader suite); `.txt` carries the reconciliation.
"""
import ingest
from ingest import SUPPORTED_EXTENSIONS, DocumentLoader, list_documents

BODY = ("Sequela codes are used for the late effects of an injury, and this "
        "sentence exists only to clear the fifty-character length guard.")


def write(directory, name, text=BODY):
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


def test_txt_is_listed(tmp_path):
    write(tmp_path, "notes.txt")
    assert [p.name for p in list_documents(tmp_path)] == ["notes.txt"]


def test_txt_is_actually_ingested(tmp_path):
    """The other half of the bug: listed *and* read."""
    write(tmp_path, "notes.txt")

    docs = DocumentLoader(str(tmp_path)).load()

    assert [d["doc_id"] for d in docs] == ["notes.txt"]
    assert docs[0]["source_type"] == "txt"


def test_listing_and_loader_agree(tmp_path):
    """The invariant that keeps the two from drifting again."""
    write(tmp_path, "notes.txt")
    write(tmp_path, "REPORT.TXT")
    write(tmp_path, "ignored.md")
    write(tmp_path, "archive.zip")
    (tmp_path / "subdir").mkdir()

    listed = {p.name for p in list_documents(tmp_path)}
    loaded = {d["doc_id"] for d in DocumentLoader(str(tmp_path)).load()}

    assert listed == loaded
    assert listed == {"notes.txt", "REPORT.TXT"}


def test_uppercase_extensions_are_handled(tmp_path):
    """pathlib's glob ignores case on Windows but not Linux.

    Globbing "*.pdf" made the sidebar platform-dependent: a REPORT.TXT would
    show on Windows and vanish on Linux, while the loader's `endswith` skipped
    it on both. Listing and loading now agree on either platform.
    """
    write(tmp_path, "REPORT.TXT")

    assert [p.name for p in list_documents(tmp_path)] == ["REPORT.TXT"]
    assert [d["doc_id"] for d in DocumentLoader(str(tmp_path)).load()] == ["REPORT.TXT"]


def test_unsupported_extensions_are_excluded(tmp_path):
    write(tmp_path, "readme.md")
    write(tmp_path, "data.csv")

    assert list_documents(tmp_path) == []


def test_directories_are_not_listed(tmp_path):
    (tmp_path / "nested.txt").mkdir()

    assert list_documents(tmp_path) == []


def test_missing_directory_is_empty_not_an_error(tmp_path):
    assert list_documents(tmp_path / "does-not-exist") == []


def test_listing_is_sorted_case_insensitively(tmp_path):
    for name in ("banana.txt", "Apple.txt", "cherry.txt"):
        write(tmp_path, name)

    assert [p.name for p in list_documents(tmp_path)] == [
        "Apple.txt", "banana.txt", "cherry.txt"]


def test_short_documents_are_dropped_by_the_length_guard(tmp_path):
    """The loader skips them, so the listing showing them is expected.

    A file can legitimately be listed and not indexed when it is under the
    50-char guard. That is a content decision, not an extension mismatch, and
    it is the one case where the two sets differ by design.
    """
    write(tmp_path, "stub.txt", "too short")

    assert [p.name for p in list_documents(tmp_path)] == ["stub.txt"]
    assert DocumentLoader(str(tmp_path)).load() == []


def test_supported_extensions_are_normalised(tmp_path):
    """The uploader strips the dot off these, so they must carry one."""
    assert all(ext.startswith(".") for ext in SUPPORTED_EXTENSIONS)
    assert all(ext == ext.lower() for ext in SUPPORTED_EXTENSIONS)
    assert ".pdf" in SUPPORTED_EXTENSIONS and ".txt" in SUPPORTED_EXTENSIONS


def test_public_names_are_exported(tmp_path):
    """app.py imports all three from the package, not the module."""
    for name in ("SUPPORTED_EXTENSIONS", "DocumentLoader", "list_documents"):
        assert hasattr(ingest, name), f"ingest.{name} is not exported"
