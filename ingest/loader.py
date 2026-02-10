import fitz  # PyMuPDF
import os
import re

class DocumentLoader:
    def __init__(self, directory_path):
        self.directory_path = directory_path
        self.documents = []

    def load(self):
        """Walks through the directory and loads supported files, skipping empty ones."""
        for filename in os.listdir(self.directory_path):
            file_path = os.path.join(self.directory_path, filename)
            doc_data = None
            
            if filename.endswith(".pdf"):
                doc_data = self._load_pdf(file_path, filename)
            elif filename.endswith(".txt"):
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