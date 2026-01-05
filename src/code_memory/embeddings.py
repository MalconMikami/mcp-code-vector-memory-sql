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
            raise RuntimeError("Embedding model not available.")
        return list(self._model.embed([text]))[0]

    def _load(self) -> None:
        if self._model is not None:
            return
        logger.info("Loading embedding model: %s (cache: %s)", self.model_name, MODEL_CACHE_DIR)
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


def serialize_vector(vec: List[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)

