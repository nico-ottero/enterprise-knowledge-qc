import numpy as np
from dataclasses import dataclass

@dataclass
class Retrieved:
    score: float
    chunk: dict  # {chunk_id, source, page, text}

def search(index, meta: list[dict], query_vec: np.ndarray, k: int = 5) -> list[Retrieved]:
    # query_vec shape (dim,)
    q = query_vec.reshape(1, -1).astype("float32")
    scores, ids = index.search(q, k)
    out: list[Retrieved] = []
    for s, i in zip(scores[0].tolist(), ids[0].tolist()):
        if i == -1:
            continue
        out.append(Retrieved(score=float(s), chunk=meta[i]))
    return out
