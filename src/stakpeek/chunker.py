from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator

import tree_sitter_c as tsc
from tree_sitter import Language, Node, Parser

C_LANGUAGE = Language(tsc.language())
C_SUFFIXES = {".c", ".h"}

# Groups of top-level declarations longer than this get split into windows.
MAX_DECL_LINES = 80

_PREPROC_BLOCKS = {
    "preproc_if", "preproc_ifdef", "preproc_else",
    "preproc_elif", "preproc_elifdef",
}


@dataclass(frozen=True)
class Chunk:
    path: str
    start_line: int  # 1-based, inclusive
    end_line: int    # 1-based, inclusive
    kind: str        # "function" | "declarations" | "text"
    name: str        # function name, or "" for other kinds
    text: str


def _function_name(node: Node) -> str:
    """Walk down nested declarators (pointers, parens) to the identifier."""
    decl = node.child_by_field_name("declarator")
    while decl is not None and decl.type != "identifier":
        decl = decl.child_by_field_name("declarator")
    return decl.text.decode() if decl is not None else "<anonymous>"


def _top_level_nodes(parent: Node) -> Iterator[Node]:
    """Yield top-level nodes, looking inside #if/#ifdef/#else blocks."""
    guards: list[Node] = []
    if parent.type in _PREPROC_BLOCKS:
        guards = [
            g for g in (parent.child_by_field_name("condition"),
                        parent.child_by_field_name("name"))
            if g is not None
        ]
    for child in parent.children:
        if not child.is_named or child in guards:
            continue  # skip "#if"/"#endif" tokens and the condition itself
        if child.type in _PREPROC_BLOCKS:
            yield from _top_level_nodes(child)
        else:
            yield child


def chunk_lines(path: str, text: str, size: int = 60, overlap: int = 10,
                first_line: int = 1) -> list[Chunk]:
    """Overlapping windows of lines. Fallback for non-C files."""
    lines = text.splitlines()
    step = size - overlap
    chunks: list[Chunk] = []
    for start in range(0, max(len(lines), 1), step):
        window = lines[start:start + size]
        if not window:
            break
        chunks.append(Chunk(
            path,
            first_line + start,
            first_line + start + len(window) - 1,
            "text",
            "",
            "\n".join(window),
        ))
        if start + size >= len(lines):
            break
    return chunks


def chunk_c(path: str, source: bytes) -> list[Chunk]:
    """One chunk per function; everything between functions is grouped."""
    tree = Parser(C_LANGUAGE).parse(source)
    chunks: list[Chunk] = []
    pending: list[Node] = []  # top-level nodes waiting to be grouped

    def make(kind: str, name: str, first: Node, last: Node) -> Chunk:
        text = source[first.start_byte:last.end_byte].decode(errors="replace")
        return Chunk(path, first.start_point[0] + 1, last.end_point[0] + 1,
                     kind, name, text)

    def flush() -> None:
        if not pending:
            return
        chunk = make("declarations", "", pending[0], pending[-1])
        if chunk.end_line - chunk.start_line + 1 > MAX_DECL_LINES:
            for piece in chunk_lines(path, chunk.text,
                                     first_line=chunk.start_line):
                chunks.append(replace(piece, kind="declarations"))
        else:
            chunks.append(chunk)
        pending.clear()

    for node in _top_level_nodes(tree.root_node):
        if node.type != "function_definition":
            pending.append(node)
            continue
        # Pull a comment sitting directly above the function into its chunk.
        top = node
        while (pending and pending[-1].type == "comment"
               and pending[-1].end_point[0] + 1 >= top.start_point[0]):
            top = pending.pop()
        flush()
        chunks.append(make("function", _function_name(node), top, node))
    flush()
    return chunks


def chunk_file(path: Path, root: Path) -> list[Chunk]:
    rel = str(path.relative_to(root))
    data = path.read_bytes()
    if path.suffix.lower() in C_SUFFIXES:
        return chunk_c(rel, data)
    return chunk_lines(rel, data.decode(errors="replace"))


if __name__ == "__main__":
    import sys

    from stakpeek.walker import walk_repo

    repo = Path(sys.argv[1])
    total = 0
    for file in walk_repo(repo):
        for c in chunk_file(file, repo):
            total += 1
            print(f"{c.path}:{c.start_line}-{c.end_line}  {c.kind}  {c.name}")
    print(f"\n{total} chunks")