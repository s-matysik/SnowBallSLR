"""Optional embedding ranker (extra ``[embeddings]``).

The model name *and* its resolved revision are pinned into the manifest. A run
aborts if the resolved revision differs from the one recorded -- an embedding
model silently updating underneath a pipeline is exactly the kind of
irreproducibility this library exists to prevent.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..errors import ConfigError
from ..types import Work

__all__ = ["EmbeddingRanker"]


class EmbeddingRanker:
    name = "embedding"

    def __init__(self, model_name: str, revision: str | None = None) -> None:
        self.model_name = model_name
        self.revision = revision
        self._model = None
        self._centroid = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ConfigError(
                "embedding ranker requires the [embeddings] extra: "
                "pip install 'snowballslr[embeddings]'"
            ) from exc
        model = SentenceTransformer(self.model_name)
        model.eval()
        torch.set_grad_enabled(False)
        self._model = model
        return model

    def descriptor(self) -> dict[str, str | None]:
        return {"model": self.model_name, "revision": self.revision}

    def _encode(self, texts: Sequence[str]):
        import numpy as np

        model = self._load()
        vectors = model.encode(
            list(texts),
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype="float32")

    def fit(self, included: Sequence[Work]) -> None:
        import numpy as np

        ordered = sorted(included, key=lambda w: w.key)
        if not ordered:
            self._centroid = None
            return
        texts = [f"{w.title} {w.abstract or ''}".strip() for w in ordered]
        vectors = self._encode(texts)
        centroid = vectors.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        self._centroid = centroid / norm if norm > 0 else centroid

    def score(self, candidates: Sequence[Work]) -> Mapping[str, float]:
        ordered = sorted(candidates, key=lambda w: w.key)
        if self._centroid is None or not ordered:
            return {w.key: 0.0 for w in ordered}
        texts = [f"{w.title} {w.abstract or ''}".strip() for w in ordered]
        vectors = self._encode(texts)
        sims = vectors @ self._centroid
        return {w.key: float(s) for w, s in zip(ordered, sims, strict=False)}
