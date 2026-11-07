"""Hybrid retrieval: BM25 keyword scores plus embedding similarity, merged with reciprocal rank fusion."""

import math
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from belge.metin import Chunk, stems


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


@dataclass
class Hit:
    chunk: Chunk
    score: float
    keyword_rank: int | None
    vector_rank: int | None


class BM25:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(stems(c.text)) for c in chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0
        df = Counter(term for doc in self.docs for term in doc)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = stems(query)
        out = []
        for doc, length in zip(self.docs, self.lengths):
            s = 0.0
            for t in terms:
                if t in doc:
                    tf = doc[t]
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / (self.avg or 1)))
            out.append(s)
        return out


class Index:
    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self.vectors: np.ndarray | None = None
        self.bm25 = BM25([])

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self.chunks.extend(chunks)
        self.bm25 = BM25(self.chunks)
        if self.embedder is not None:
            new = _unit(np.array(self.embedder.embed([c.text for c in chunks]), dtype=float))
            self.vectors = new if self.vectors is None else np.vstack([self.vectors, new])

    def remove_source(self, source: str) -> int:
        keep = [i for i, c in enumerate(self.chunks) if c.source != source]
        removed = len(self.chunks) - len(keep)
        self.chunks = [self.chunks[i] for i in keep]
        if self.vectors is not None:
            self.vectors = self.vectors[keep] if keep else None
        self.bm25 = BM25(self.chunks)
        return removed

    def search(self, query: str, k: int = 4, rrf_k: int = 60) -> list[Hit]:
        if not self.chunks:
            return []
        keyword = self.bm25.scores(query)
        keyword_order = [i for i in np.argsort(keyword)[::-1] if keyword[i] > 0]
        vector_order: list[int] = []
        if self.embedder is not None and self.vectors is not None:
            q = _unit(np.array(self.embedder.embed([query]), dtype=float))[0]
            similarity = self.vectors @ q
            vector_order = [int(i) for i in np.argsort(similarity)[::-1] if similarity[i] > 0.2]

        fused: dict[int, float] = {}
        ranks: dict[int, list[int | None]] = {}
        for rank, i in enumerate(keyword_order):
            fused[i] = fused.get(i, 0) + 1 / (rrf_k + rank + 1)
            ranks.setdefault(i, [None, None])[0] = rank + 1
        for rank, i in enumerate(vector_order):
            fused[i] = fused.get(i, 0) + 1 / (rrf_k + rank + 1)
            ranks.setdefault(i, [None, None])[1] = rank + 1
        best = sorted(fused, key=fused.get, reverse=True)[:k]
        return [Hit(self.chunks[i], fused[i], ranks[i][0], ranks[i][1]) for i in best]


def _unit(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return matrix / norms
