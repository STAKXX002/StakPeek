from __future__ import annotations

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


def walk_repo(
    root: Path,
    exclude_dirs: frozenset[str] = DEFAULT_EXCLUDE_DIRS,
    extensions: frozenset[str] = DEFAULT_EXTENSIONS,
) -> Iterator[Path]:
    """Yield files under `root` that are worth indexing."""
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in exclude_dirs for part in path.relative_to(root).parts):
            continue
        if path.suffix.lower() not in extensions:
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            continue
        yield path


if __name__ == "__main__":
    import sys

    repo = Path(sys.argv[1])
    for file in walk_repo(repo):
        print(file.relative_to(repo))
