from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from enterprise_ai.rag.corpus import DEFAULT_CHUNKS


class TfidfRetriever:
    """Lightweight lexical retriever (TF–IDF + cosine similarity). Runs fully offline."""

    def __init__(self, chunks: list[str] | None = None) -> None:
        self._chunks = chunks or list(DEFAULT_CHUNKS)
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=8192,
        )
        self._matrix = self._vectorizer.fit_transform(self._chunks)

    def retrieve(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        q = self._vectorizer.transform([query])
        sims = cosine_similarity(q, self._matrix).ravel()
        idx = np.argsort(-sims)[:top_k]
        return [(self._chunks[int(i)], float(sims[int(i)])) for i in idx if sims[int(i)] > 0]
