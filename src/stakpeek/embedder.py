from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"

# bge models work best when queries (not documents) carry this prefix.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    """Turns text into L2-normalised vectors, so dot product == cosine similarity."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 64,
        )

    def embed_query(self, query: str) -> np.ndarray:
        return self._model.encode(QUERY_PREFIX + query, normalize_embeddings=True)