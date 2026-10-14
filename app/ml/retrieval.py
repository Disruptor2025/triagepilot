"""Retrieval of similar historical failures (the "R" in RAG).

TF-IDF + cosine similarity keeps this dependency-free and explainable. The interface
(`index`, `search`) is deliberately small so it can be swapped for embeddings in a
vector database (pgvector, Chroma) without touching the API layer.
"""
from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Above this similarity, a new failure is reported as a likely duplicate of an existing one.
DUPLICATE_THRESHOLD = 0.80


@dataclass
class Hit:
    doc_id: int
    score: float


class FailureIndex:
    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self._ids: list[int] = []
        self._matrix = None

    def index(self, ids: list[int], texts: list[str]) -> None:
        self._ids = list(ids)
        self._matrix = self._vectorizer.fit_transform(texts) if texts else None

    def __len__(self) -> int:
        return len(self._ids)

    def search(self, text: str, k: int = 5, exclude_id: int | None = None) -> list[Hit]:
        if self._matrix is None:
            return []
        scores = cosine_similarity(self._vectorizer.transform([text]), self._matrix)[0]
        ranked = sorted(zip(self._ids, scores, strict=True), key=lambda p: p[1], reverse=True)
        hits = [Hit(doc_id=i, score=round(float(s), 4)) for i, s in ranked if i != exclude_id and s > 0]
        return hits[:k]
