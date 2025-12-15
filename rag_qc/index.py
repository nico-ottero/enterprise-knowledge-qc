import os, json
import numpy as np
import faiss
from .utils import ensure_dir, write_jsonl, read_jsonl
from .config import settings

META_PATH = "chunks.jsonl"
FAISS_PATH = "faiss.index"

def build_index(vectors: np.ndarray) -> faiss.Index:
   
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index

def save_index(index: faiss.Index, chunks_meta: list[dict], index_dir: str) -> None:
    ensure_dir(index_dir)
    faiss.write_index(index, os.path.join(index_dir, FAISS_PATH))
    write_jsonl(os.path.join(index_dir, META_PATH), chunks_meta)

def load_index(index_dir: str):
    index = faiss.read_index(os.path.join(index_dir, FAISS_PATH))
    meta = read_jsonl(os.path.join(index_dir, META_PATH))
    return index, meta

def exists(index_dir: str) -> bool:
    return os.path.exists(os.path.join(index_dir, FAISS_PATH)) and os.path.exists(os.path.join(index_dir, META_PATH))
