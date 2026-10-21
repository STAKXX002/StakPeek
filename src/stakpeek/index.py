from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from stakpeek.chunker import Chunk, chunk_file
from stakpeek.embedder import Embedder
from stakpeek.walker import walk_repo

INDEX_DIRNAME = ".stakpeek"

# Mild preference for real functions over headers and plain text chunks.
KIND_WEIGHTS: dict[str, float] = {
    "function": 1.0,
    "declarations": 0.93,
    "text": 0.93,
}


def _embedding_text(chunk: Chunk) -> str:
    """Prefix the file path and function name so they influence the vector."""
    label = f"{chunk.path} :: {chunk.name}" if chunk.name else chunk.path
    return f"{label}\n{chunk.text}"


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class Index:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray,
                 model_name: str) -> None:
        self.chunks = chunks
        self.vectors = vectors  # shape (n_chunks, dim), rows L2-normalised
        self.model_name = model_name
        self._weights = np.array(
            [KIND_WEIGHTS.get(c.kind, 1.0) for c in chunks], dtype=np.float32
        )

    @classmethod
    def build(cls, repo: Path, embedder: Embedder) -> Index:
        chunks = [c for f in walk_repo(repo) for c in chunk_file(f, repo)]
        vectors = embedder.embed_documents([_embedding_text(c) for c in chunks])
        return cls(chunks, vectors, embedder.model_name)

    def search(self, query_vec: np.ndarray, k: int = 5) -> list[Hit]:
        scores = (self.vectors @ query_vec) * self._weights
        top = np.argsort(scores)[::-1][:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in top]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        meta = {"model": self.model_name,
                "chunks": [asdict(c) for c in self.chunks]}
        (directory / "chunks.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, directory: Path) -> Index:
        vectors = np.load(directory / "vectors.npy")
        meta = json.loads((directory / "chunks.json").read_text())
        chunks = [Chunk(**d) for d in meta["chunks"]]
        return cls(chunks, vectors, meta["model"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="stakpeek")
    sub = parser.add_subparsers(dest="cmd", required=True)

    build = sub.add_parser("build", help="index a repository")
    build.add_argument("repo", type=Path)

    search = sub.add_parser("search", help="search an indexed repository")
    search.add_argument("repo", type=Path)
    search.add_argument("query")
    search.add_argument("-k", type=int, default=5)

    args = parser.parse_args(argv)
    repo = args.repo.expanduser().resolve()
    index_dir = Path(INDEX_DIRNAME) / repo.name
    embedder = Embedder()

    if args.cmd == "build":
        index = Index.build(repo, embedder)
        index.save(index_dir)
        print(f"Indexed {len(index.chunks)} chunks -> {index_dir}")
        return

    index = Index.load(index_dir)
    if index.model_name != embedder.model_name:
        raise SystemExit("Index was built with a different model; rebuild it.")
    for hit in index.search(embedder.embed_query(args.query), args.k):
        c = hit.chunk
        print(f"{hit.score:.3f}  {c.path}:{c.start_line}-{c.end_line}  "
              f"{c.name or c.kind}")


if __name__ == "__main__":
    main()