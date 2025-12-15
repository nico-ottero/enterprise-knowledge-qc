from dataclasses import dataclass
from typing import List
from .retrieve import Retrieved
from .judge import Judgment

@dataclass
class FinalAnswer:
    status: str
    confidence: float
    answer: str
    citations: list[dict]
    reasons: list[str]

def make_citations(retrieved: List[Retrieved]) -> list[dict]:
    cites = []
    for r in retrieved[:5]:
        cites.append({
            "source": r.chunk["source"],
            "page": r.chunk["page"],
            "chunk_id": r.chunk["chunk_id"],
            "score": r.score,
        })
    return cites

def extractive_answer(question: str, retrieved: List[Retrieved], judgment: Judgment) -> FinalAnswer:
    cites = make_citations(retrieved)

    if judgment.status in ("NO_EVIDENCE",):
        return FinalAnswer(
            status=judgment.status,
            confidence=judgment.confidence,
            answer="No puedo responder con la evidencia disponible. Requiere más contexto o documentos.",
            citations=cites,
            reasons=judgment.reasons,
        )

    # Extractivo: devolvemos los 1–2 mejores fragmentos como “base”
    top_texts = [retrieved[i].chunk["text"].strip() for i in range(min(2, len(retrieved)))]
    answer = (
        f"Basado en la documentación recuperada, la evidencia más relevante es:\n\n"
        + "\n\n---\n\n".join(top_texts)
    )

    if judgment.status == "NEEDS_REVIEW":
        answer = "⚠️ Respuesta provisional (requiere revisión humana):\n\n" + answer

    return FinalAnswer(
        status=judgment.status,
        confidence=judgment.confidence,
        answer=answer,
        citations=cites,
        reasons=judgment.reasons,
    )
