"""BM25 ranker.

Implemented in-package rather than pulled from a dependency so that
tokenization and IDF computation are pinned and deterministic.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence

from ..identity.normalize import tokenize
from ..types import Work

__all__ = ["BM25Ranker"]


class BM25Ranker:
    name = "bm25"

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._query: Counter[str] = Counter()

    def fit(self, included: Sequence[Work]) -> None:
        self._query = Counter()
        for w in sorted(included, key=lambda x: x.key):
            self._query.update(tokenize(f"{w.title} {w.abstract or ''}"))

    def score(self, candidates: Sequence[Work]) -> Mapping[str, float]:
        docs = {
            w.key: tokenize(f"{w.title} {w.abstract or ''}")
            for w in sorted(candidates, key=lambda x: x.key)
        }
        if not docs or not self._query:
            return {k: 0.0 for k in docs}

        n = len(docs)
        avgdl = sum(len(d) for d in docs.values()) / n if n else 0.0
        df: Counter[str] = Counter()
        for toks in docs.values():
            df.update(set(toks))

        idf = {
            term: math.log(1.0 + (n - df[term] + 0.5) / (df[term] + 0.5))
            for term in df
        }

        out: dict[str, float] = {}
        for key in sorted(docs):
            toks = docs[key]
            tf = Counter(toks)
            dl = len(toks)
            total = 0.0
            for term, qw in sorted(self._query.items()):
                if term not in tf:
                    continue
                f = tf[term]
                denom = f + self.k1 * (1 - self.b + self.b * (dl / avgdl if avgdl else 0))
                total += idf.get(term, 0.0) * (f * (self.k1 + 1)) / (denom or 1.0) * math.log1p(qw)
            out[key] = total
        return out
