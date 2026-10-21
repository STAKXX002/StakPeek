from __future__ import annotations

import argparse
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from stakpeek.chunker import Chunk
from stakpeek.embedder import Embedder
from stakpeek.index import INDEX_DIRNAME, Hit, Index
from stakpeek.llm import LLM, LLMConfig, LLMError, OpenAICompatLLM

DEFAULT_K = 10

# Free tiers limit tokens per minute, so keep the prompt small (~4k tokens).
MAX_CONTEXT_CHARS = 14_000

SYSTEM_PROMPT = """\
You answer questions about a source-code repository using ONLY the code \
excerpts provided by the user.

Rules:
- Cite the excerpts you rely on in square brackets, exactly as labelled, \
for example [Core/Src/hatch.c:55-58]. Use plain ASCII only: ordinary square \
brackets and an ordinary hyphen, never other bracket styles or special dashes.
- If the excerpts do not contain enough information to answer, say so \
plainly. Do not guess or invent code that is not shown.
- If the question's wording does not exactly match a name in the code \
(for example "emergency stop"), list every function or command in the \
excerpts that could be what the user means, and say which one you chose and why.
- Be concise and concrete. Name functions, variables and registers as they \
appear in the code.
"""

# Models sometimes emit fullwidth brackets and special hyphens; map them to ASCII.
_ASCII_FIXES = str.maketrans({
    "\u3010": "[", "\u3011": "]",                  # fullwidth brackets
    "\u2010": "-", "\u2011": "-", "\u2013": "-",   # hyphen variants, en dash
})


@dataclass(frozen=True)
class Answer:
    sources: list[Hit]       # the chunks actually placed in the prompt
    stream: Iterator[str]    # the answer text; the model is called when you iterate


def _format_chunk(chunk: Chunk) -> str:
    label = f"{chunk.kind} {chunk.name}".strip()
    header = f"[{chunk.path}:{chunk.start_line}-{chunk.end_line}] ({label})"
    return f"{header}\n{chunk.text}\n"


def build_prompt(question: str, hits: list[Hit],
                 max_chars: int = MAX_CONTEXT_CHARS) -> tuple[str, list[Hit]]:
    """Pack as many retrieved chunks as fit in the budget, best match first."""
    parts: list[str] = []
    used: list[Hit] = []
    remaining = max_chars
    for hit in hits:
        block = _format_chunk(hit.chunk)
        if len(block) > remaining:
            if used:
                continue                  # a smaller, lower-ranked chunk may still fit
            block = block[:remaining]     # always include something
        parts.append(block)
        used.append(hit)
        remaining -= len(block)
    excerpts = "\n".join(parts)
    return f"Code excerpts:\n\n{excerpts}\nQuestion: {question}", used


def _normalise(pieces: Iterator[str]) -> Iterator[str]:
    for piece in pieces:
        yield piece.translate(_ASCII_FIXES)


def ask(question: str, index: Index, embedder: Embedder, llm: LLM,
        k: int = DEFAULT_K) -> Answer:
    hits = index.search(embedder.embed_query(question), k)
    prompt, used = build_prompt(question, hits)
    return Answer(sources=used,
                  stream=_normalise(llm.stream(SYSTEM_PROMPT, prompt)))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="stakpeek-ask")
    parser.add_argument("repo", type=Path)
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=DEFAULT_K)
    args = parser.parse_args(argv)

    repo = args.repo.expanduser().resolve()
    index = Index.load(Path(INDEX_DIRNAME) / repo.name)
    try:
        llm = OpenAICompatLLM(LLMConfig.from_env())
        result = ask(args.question, index, Embedder(), llm, args.k)
        print("Sources:")
        for hit in result.sources:
            c = hit.chunk
            print(f"  {c.path}:{c.start_line}-{c.end_line}  {c.name or c.kind}")
        print()
        for piece in result.stream:
            print(piece, end="", flush=True)
        print()
    except LLMError as err:
        raise SystemExit(f"\nError: {err}")


if __name__ == "__main__":
    main()
