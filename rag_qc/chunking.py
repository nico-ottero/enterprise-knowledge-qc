from dataclasses import dataclass
from typing import Iterable

@dataclass
class Chunk:
    chunk_id: str
    source: str
    page: int
    text: str

def chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> list[str]:
    
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks

def make_chunks(pages: Iterable, id_fn) -> list[Chunk]:
    out: list[Chunk] = []
    for p in pages:
        parts = chunk_text(p.text)
        for idx, t in enumerate(parts):
            cid = id_fn(f"{p.source}:{p.page}:{idx}:{t[:80]}")
            out.append(Chunk(chunk_id=cid, source=p.source, page=p.page, text=t))
    return out
