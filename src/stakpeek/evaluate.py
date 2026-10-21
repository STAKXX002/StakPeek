from __future__ import annotations

import json
import sys
from pathlib import Path

from stakpeek.embedder import Embedder
from stakpeek.index import INDEX_DIRNAME, Index

TOP_K = 10
PASS_RANK = 5


def first_hit_rank(names: list[str], expected: list[str]) -> int | None:
    """1-based rank of the first expected name, or None if none appear."""
    for rank, name in enumerate(names, start=1):
        if name in expected:
            return rank
    return None


def main() -> None:
    repo = Path(sys.argv[1]).expanduser().resolve()
    questions = json.loads(Path(sys.argv[2]).read_text())
    index = Index.load(Path(INDEX_DIRNAME) / repo.name)
    embedder = Embedder()

    passed = 0
    reciprocal_ranks: list[float] = []
    for item in questions:
        hits = index.search(embedder.embed_query(item["q"]), TOP_K)
        rank = first_hit_rank([h.chunk.name for h in hits], item["expect"])
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        if rank is not None and rank <= PASS_RANK:
            passed += 1
        print(f"{rank if rank else 'MISS':>5}  {item['q']}")

    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    print(f"\nfirst hit in top {PASS_RANK}: {passed}/{len(questions)}   MRR: {mrr:.3f}")


if __name__ == "__main__":
    main()