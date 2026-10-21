from pathlib import Path

import numpy as np

from stakpeek.chunker import Chunk
from stakpeek.index import Index


def _chunk(name: str) -> Chunk:
    return Chunk("a.c", 1, 2, "function", name, "x")


def _index() -> Index:
    vectors = np.array([[1, 0], [0, 1], [0.7071, 0.7071]], dtype=np.float32)
    return Index([_chunk("a"), _chunk("b"), _chunk("c")], vectors, "fake")


def test_search_returns_best_match_first() -> None:
    hits = _index().search(np.array([0, 1], dtype=np.float32), k=2)

    assert [h.chunk.name for h in hits] == ["b", "c"]


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    index = _index()
    index.save(tmp_path)
    loaded = Index.load(tmp_path)

    assert loaded.chunks == index.chunks
    assert np.allclose(loaded.vectors, index.vectors)
    assert loaded.model_name == "fake"