from collections.abc import Iterator

import numpy as np

from stakpeek.answer import SYSTEM_PROMPT, ask, build_prompt
from stakpeek.chunker import Chunk
from stakpeek.index import Hit, Index


class FakeLLM:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def stream(self, system: str, user: str) -> Iterator[str]:
        self.calls.append((system, user))
        yield "hello "
        yield "world"


class FakeEmbedder:
    def embed_query(self, query: str) -> np.ndarray:
        return np.array([0, 1], dtype=np.float32)


def _chunk(name: str, text: str = "x") -> Chunk:
    return Chunk("a.c", 1, 2, "function", name, text)


def test_ask_retrieves_builds_prompt_and_streams() -> None:
    vectors = np.array([[1, 0], [0, 1]], dtype=np.float32)
    index = Index([_chunk("far"), _chunk("near")], vectors, "fake")
    llm = FakeLLM()

    result = ask("what does near do?", index, FakeEmbedder(), llm, k=2)

    assert [h.chunk.name for h in result.sources] == ["near", "far"]
    assert "".join(result.stream) == "hello world"
    system, user = llm.calls[0]
    assert system == SYSTEM_PROMPT
    assert "[a.c:1-2]" in user
    assert "what does near do?" in user


def test_prompt_respects_character_budget() -> None:
    hits = [Hit(_chunk(f"f{i}", "y" * 1000), 1.0) for i in range(3)]

    _, used = build_prompt("q", hits, max_chars=1500)

    assert len(used) == 1


def test_prompt_always_includes_best_chunk_even_if_too_big() -> None:
    hits = [Hit(_chunk("big", "z" * 5000), 1.0)]

    prompt, used = build_prompt("q", hits, max_chars=500)

    assert len(used) == 1
    assert "big" in prompt


def test_stream_normalises_citation_characters() -> None:
    class OddLLM:
        def stream(self, system: str, user: str) -> Iterator[str]:
            yield "see \u3010a.c:1\u20112\u3011"

    vectors = np.array([[0, 1]], dtype=np.float32)
    index = Index([_chunk("f")], vectors, "fake")

    result = ask("q", index, FakeEmbedder(), OddLLM())

    assert "".join(result.stream) == "see [a.c:1-2]"
