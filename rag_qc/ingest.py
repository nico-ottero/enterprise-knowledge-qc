from pypdf import PdfReader
from dataclasses import dataclass
from typing import Iterator
import os

@dataclass
class PageDoc:
    source: str
    page: int
    text: str

def iter_pdf_pages(pdf_path: str) -> Iterator[PageDoc]:
    reader = PdfReader(pdf_path)
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            yield PageDoc(source=os.path.basename(pdf_path), page=i+1, text=text)

def load_raw_documents(raw_dir: str) -> list[PageDoc]:
    docs: list[PageDoc] = []
    for fn in os.listdir(raw_dir):
        if fn.lower().endswith(".pdf"):
            docs.extend(list(iter_pdf_pages(os.path.join(raw_dir, fn))))
    return docs
