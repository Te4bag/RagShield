import fitz  # PyMuPDF
import os
import re
from pathlib import Path

# Every extension the loader can actually ingest. The UI reads this for its
# uploader filter and its document listings, so the two cannot drift apart --
# a .txt used to be indexed but never shown, and if it was the only file
# present the app reported "No documents found".
SUPPORTED_EXTENSIONS = (".pdf", ".txt")


def list_documents(directory):
    """Every file in `directory` the loader would ingest, sorted by name.

    Matches `DocumentLoader.load()` exactly, including case-insensitivity, so
    anything listed here is something the indexer will read. Globbing "*.pdf"
    instead would disagree across platforms: pathlib's glob ignores case on
    Windows but not on Linux, so a "REPORT.PDF" would be listed on one and
    silently skipped on the other.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []
    docs = [p for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    return sorted(docs, key=lambda p: p.name.lower())


class DocumentLoader:
    def __init__(self, directory_path):
        self.directory_path = directory_path
        self.documents = []

    def load(self):
        """Walks through the directory and loads supported files, skipping empty ones."""
        for filename in os.listdir(self.directory_path):
            file_path = os.path.join(self.directory_path, filename)
            doc_data = None
            
            # Case-insensitive so the loader agrees with list_documents().
            suffix = os.path.splitext(filename)[1].lower()
            if suffix == ".pdf":
                doc_data = self._load_pdf(file_path, filename)
            elif suffix == ".txt":
                doc_data = self._load_txt(file_path, filename)
            
            # Length Guard: Only append if document has substantial content
            if doc_data and doc_data["text"]:
                self.documents.append(doc_data)
        
        return self.documents

    def _load_pdf(self, path, name):
        text = ""
        try:
            with fitz.open(path) as doc:
                for page in doc:
                    text += page.get_text("text") + "\n"
            
            cleaned_text = self._clean_text(text)
            if len(cleaned_text) < 50:
                return None
                
            return {
                "doc_id": name, 
                "text": cleaned_text, 
                "source_type": "pdf"
            }
        except Exception as e:
            print(f"Error loading PDF {name}: {e}")
            return None

    def _load_txt(self, path, name):
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            
            cleaned_text = self._clean_text(text)
            if len(cleaned_text) < 50:
                return None

            return {
                "doc_id": name, 
                "text": cleaned_text, 
                "source_type": "txt"
            }
        except Exception as e:
            print(f"Error loading TXT {name}: {e}")
            return None

    def _clean_text(self, text):
        # Collapse whitespace and remove non-printable characters
        text = re.sub(r'\s+', ' ', text)
        return text.strip()