from __future__ import annotations

from collections.abc import Sequence
from fnmatch import fnmatch
from pathlib import Path
from typing import Iterator

# Directories we never descend into.
DEFAULT_EXCLUDE_DIRS: frozenset[str] = frozenset({
    ".git", ".venv", "node_modules", "build", "Drivers", "__pycache__",
    ".settings", ".vscode", ".idea", ".pytest_cache",
})

# File types worth indexing.
DEFAULT_EXTENSIONS: frozenset[str] = frozenset({
    ".c", ".h", ".cpp", ".hpp", ".s", ".ld", ".py", ".rs",
    ".md", ".txt", ".cmake", ".json", ".ioc",
})

MAX_FILE_BYTES = 500_000  # skip huge generated files
IGNORE_FILE = ".stakpeekignore"


def load_ignore_patterns(root: Path) -> list[str]:
    """Read glob patterns from <root>/.stakpeekignore (one per line)."""
    ignore_file = root / IGNORE_FILE
    if not ignore_file.is_file():
        return []
    return [
        line.strip()
        for line in ignore_file.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _is_ignored(rel: Path, patterns: Sequence[str]) -> bool:
    """A pattern matches either the relative path or just the file name."""
    return any(fnmatch(rel.as_posix(), p) or fnmatch(rel.name, p)
               for p in patterns)


def walk_repo(
    root: Path,
    exclude_dirs: frozenset[str] = DEFAULT_EXCLUDE_DIRS,
    extensions: frozenset[str] = DEFAULT_EXTENSIONS,
    ignore_patterns: Sequence[str] | None = None,
) -> Iterator[Path]:
    """Yield files under `root` that are worth indexing."""
    patterns = (load_ignore_patterns(root)
                if ignore_patterns is None else ignore_patterns)
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in exclude_dirs for part in rel.parts):
            continue
        if path.suffix.lower() not in extensions:
            continue
        if _is_ignored(rel, patterns):
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            continue
        yield path


if __name__ == "__main__":
    import sys

    repo = Path(sys.argv[1])
    for file in walk_repo(repo):
        print(file.relative_to(repo))