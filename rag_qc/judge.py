from dataclasses import dataclass
from typing import List
from .retrieve import Retrieved

@dataclass
class Judgment:
    confidence: float
    status: str  # OK | NEEDS_REVIEW | NO_EVIDENCE
    reasons: list[str]

def heuristic_contradiction(snippets: List[str]) -> bool:
    # Regla simple inicial: detecta “sí/no” o “debe/no debe” cerca del mismo tema
    joined = " ".join(snippets).lower()
    patterns = [
        ("debe", "no debe"),
        ("permitido", "no permitido"),
        ("required", "not required"),
        ("must", "must not"),
        ("allowed", "not allowed"),
    ]
    return any(a in joined and b in joined for a, b in patterns)

def judge(retrieved: List[Retrieved]) -> Judgment:
    if not retrieved:
        return Judgment(confidence=0.0, status="NO_EVIDENCE", reasons=["Sin evidencia recuperada."])

    top = retrieved[0].score
    # Como embeddings normalizados + IP ~ cos, top suele estar en [0,1]
    # Umbrales conservadores (ajustables)
    reasons = []
    conf = max(0.0, min(1.0, (top - 0.2) / 0.6))  # map 0.2..0.8 -> 0..1

    snippets = [r.chunk["text"][:400] for r in retrieved[:5]]
    if heuristic_contradiction(snippets):
        conf *= 0.55
        reasons.append("Posible contradicción entre fuentes recuperadas.")

    # Si el top score es bajo, probablemente es “match flojo”
    if top < 0.35:
        return Judgment(confidence=conf, status="NO_EVIDENCE", reasons=reasons + ["Score bajo: evidencia débil."])

    if conf < 0.55:
        return Judgment(confidence=conf, status="NEEDS_REVIEW", reasons=reasons + ["Confianza insuficiente: requiere revisión humana."])

    return Judgment(confidence=conf, status="OK", reasons=reasons or ["Evidencia consistente."])
