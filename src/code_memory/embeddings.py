from __future__ import annotations

import os
import struct
from typing import List, Optional

from fastembed import TextEmbedding

from .config import DEFAULT_EMBED_MODEL, EMBED_DIM_CONFIGURED, EMBED_MODEL_NAME, MODEL_CACHE_DIR, logger


class EmbeddingModel:
    """Wrapper for fastembed on CPU, with lazy loading."""

    def __init__(self, model_name: str = EMBED_MODEL_NAME, embed_dim: Optional[int] = None):
        os.environ.setdefault("FASTEMBED_CACHE_PATH", str(MODEL_CACHE_DIR))
        os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR))
        self.model_name = model_name
        self._model: Optional[TextEmbedding] = None
        self._dim = embed_dim
        if not EMBED_DIM_CONFIGURED and model_name != DEFAULT_EMBED_MODEL:
            logger.warning(
                "CODE_MEMORY_EMBED_DIM not set for model %s; defaulting to %s.",
                model_name,
                self._dim,
            )
        if self._dim is None:
            logger.info("Embedding model configured: %s (cache: %s)", model_name, MODEL_CACHE_DIR)
        else:
            logger.info("Embedding model configured: %s (dim=%s cache: %s)", model_name, self._dim, MODEL_CACHE_DIR)

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._load()
        return int(self._dim or 0)

    def embed(self, text: str) -> List[float]:
        self._load()
        if not self._model:
            # Deterministic offline embedding: stable across runs; not semantic, but enables vector pipeline.
            import hashlib

            dim = int(self._dim or 384) or 384
            digest = hashlib.sha256((text or "").encode("utf-8")).digest()
            out: List[float] = []
            for i in range(dim):
                b = digest[i % len(digest)]
                out.append((b / 255.0) * 2.0 - 1.0)
            return out
        vec = list(self._model.embed([text]))[0]
        try:
            # fastembed may return a numpy array; normalize to a plain list of floats.
            return list(map(float, list(vec)))
        except Exception:
            return list(vec)

    def _load(self) -> None:
        if self._model is not None:
            return
        logger.info("Loading embedding model: %s (cache: %s)", self.model_name, MODEL_CACHE_DIR)
        try:
            self._model = TextEmbedding(model_name=self.model_name)
            actual_dim = len(next(self._model.embed(["init"])))
            if self._dim is None:
                self._dim = actual_dim
            elif self._dim != actual_dim:
                raise ValueError(
                    f"Embedding dim mismatch: configured={self._dim} actual={actual_dim}. "
                    "Update CODE_MEMORY_EMBED_DIM or rebuild the DB."
                )
            logger.info("Embedding model ready: dim=%s", self._dim)
        except Exception as exc:
            # Deterministic offline fallback (keeps the server functional without model downloads).
            logger.warning("fastembed load failed (%s); using deterministic hash embeddings.", exc)
            if self._dim is None:
                self._dim = 384
            self._model = None


def serialize_vector(vec: List[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)
