from pathlib import Path

from stakpeek.walker import walk_repo


def test_skips_excluded_dirs_and_binaries(tmp_path: Path) -> None:
    (tmp_path / "Core").mkdir()
    (tmp_path / "Core" / "main.c").write_text("int main(void) { return 0; }")
    (tmp_path / "Drivers").mkdir()
    (tmp_path / "Drivers" / "hal.c").write_text("// vendor code")
    (tmp_path / "firmware.bin").write_bytes(b"\x00\x01")

    found = {p.name for p in walk_repo(tmp_path)}

    assert found == {"main.c"}


def test_skips_editor_and_cache_dirs(tmp_path: Path) -> None:
    (tmp_path / ".vscode").mkdir()
    (tmp_path / ".vscode" / "settings.json").write_text("{}")
    (tmp_path / "simulation" / ".pytest_cache").mkdir(parents=True)
    (tmp_path / "simulation" / ".pytest_cache" / "README.md").write_text("x")
    (tmp_path / "keep.c").write_text("int x;")

    assert {p.name for p in walk_repo(tmp_path)} == {"keep.c"}


def test_respects_stakpeekignore(tmp_path: Path) -> None:
    (tmp_path / ".stakpeekignore").write_text(
        "# generated config\n*_hal_conf.h\nsyscalls.c\n"
    )
    (tmp_path / "Core" / "Inc").mkdir(parents=True)
    (tmp_path / "Core" / "Inc" / "stm32f4xx_hal_conf.h").write_text("#define X 1")
    (tmp_path / "Core" / "Src").mkdir()
    (tmp_path / "Core" / "Src" / "syscalls.c").write_text("int y;")
    (tmp_path / "Core" / "Src" / "main.c").write_text("int main(void);")

    assert {p.name for p in walk_repo(tmp_path)} == {"main.c"}