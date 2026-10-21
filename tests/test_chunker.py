from stakpeek.chunker import chunk_c

SAMPLE = b"""#include <stdint.h>

static int counter;

/* Adds two numbers. */
int add(int a, int b)
{
    return a + b;
}

static void *get_ptr(void)
{
    return 0;
}
"""

IFDEF_SAMPLE = b"""#if defined(FOO)
void a(void)
{
}
#else
void b(void)
{
}
#endif
"""


def test_functions_become_chunks() -> None:
    chunks = chunk_c("x.c", SAMPLE)
    funcs = {c.name: c for c in chunks if c.kind == "function"}

    assert set(funcs) == {"add", "get_ptr"}
    assert "Adds two numbers" in funcs["add"].text  # leading comment attached
    assert funcs["add"].start_line == 5


def test_other_top_level_code_is_grouped() -> None:
    decls = [c for c in chunk_c("x.c", SAMPLE) if c.kind == "declarations"]

    assert len(decls) == 1
    assert "counter" in decls[0].text


def test_functions_inside_preprocessor_blocks_are_found() -> None:
    names = {c.name for c in chunk_c("x.c", IFDEF_SAMPLE) if c.kind == "function"}

    assert names == {"a", "b"}