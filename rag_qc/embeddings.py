import numpy as np
from dataclasses import dataclass
from typing import Protocol, List
from .config import settings

class EmbeddingProvider(Protocol):
    dim: int
    def embed(self, texts: List[str]) -> np.ndarray: ...

@dataclass
class LocalSentenceTransformersProvider:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dim: int = 384

    def __post_init__(self):
        from sentence_transformers import SentenceTransformer
        self._m = SentenceTransformer(self.model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        vecs = self._m.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype="float32")

def get_provider() -> EmbeddingProvider:
    if settings.embeddings_provider == "local":
        return LocalSentenceTransformersProvider()
    raise ValueError("EMBEDDINGS_PROVIDER no soportado aún en este scaffold (arrancamos local).")
