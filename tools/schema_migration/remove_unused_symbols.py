# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
# mypy: disable-error-code="union-attr"
"""Remove top-level symbols in 'onnx/defs' that are no longer used.

One-off migration tool, to be run after 'remove_empty_generators.py'. The
candidates are the file-local symbols in 'onnx/defs/<area>/*.cc' ('static'
symbols, 'const' and 'constexpr' variables, and type aliases) and the symbols
declared in 'onnx/defs/<area>/*.h' and 'onnx/defs/doc_strings.h', except for
those marked 'ONNX_API'. A candidate is unused if its name does not occur
outside of comments and of its declarations and definitions: for file-local
symbols in its own file, otherwise in 'onnx' and 'tests'. Unused symbols are removed until there
are none left. Files without any symbols are deleted together with their
includes and their entries in the 'CMakeLists.txt' of their directory.

Requires 'tree-sitter' and 'tree-sitter-cpp'. Run 'pixi run lint' afterwards.

Usage:
    python tools/schema_migration/remove_unused_symbols.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from registrations import DEFS_DIR, PARSER, REPO_ROOT, walk
from remove_symbols import declared_name, remove, top_level

if TYPE_CHECKING:
    from pathlib import Path

    from tree_sitter import Node

_SYMBOLS = ("declaration", "function_definition", "alias_declaration")
# Top-level nodes of a file without any content.
_SCAFFOLDING = (
    "comment",
    "preproc_include",
    "preproc_call",
    "identifier",
    "namespace_identifier",
    "nested_namespace_specifier",
)


@dataclass(frozen=True)
class Symbol:
    name: str
    # The file of a file-local symbol, None for a symbol declared in a header.
    static_in: Path | None


def candidate_files() -> list[Path]:
    areas = sorted(DEFS_DIR.glob("*/*.cc")) + sorted(DEFS_DIR.glob("*/*.h"))
    return [*areas, DEFS_DIR / "doc_strings.h", DEFS_DIR / "doc_strings.cc"]


def searched_files() -> list[Path]:
    sources = [
        p
        for root in (REPO_ROOT / "onnx", REPO_ROOT / "tests")
        for p in root.rglob("*")
        if p.suffix in (".cc", ".h", ".cpp")
    ]
    return sorted(sources)


def symbols(path: Path) -> list[tuple[str, int, int]]:
    """Return the top-level symbols of `path` with their spans."""
    if path.suffix not in (".cc", ".h"):
        return []
    return [
        (name, node.start_byte, node.end_byte)
        for node in top_level(PARSER.parse(path.read_bytes()).root_node)
        if node.type in _SYMBOLS and (name := declared_name(node)) is not None
    ]


def comments(path: Path) -> list[tuple[int, int]]:
    if path.suffix not in (".cc", ".h", ".cpp"):
        return []
    root = PARSER.parse(path.read_bytes()).root_node
    return [(n.start_byte, n.end_byte) for n in walk(root) if n.type == "comment"]


def is_file_local(node: Node) -> bool:
    """Return whether a top-level symbol in a source file has internal linkage."""
    if node.type == "alias_declaration" or node.text.startswith(b"static"):
        return True
    # Variables declared `const` or `constexpr` without `extern`.
    qualifiers = {c.text for c in node.children if c.type == "type_qualifier"}
    return node.type == "declaration" and bool(qualifiers & {b"const", b"constexpr"})


def candidates() -> set[Symbol]:
    out = set()
    for path in sorted(candidate_files(), key=lambda p: p.suffix != ".h"):
        if not path.exists():
            continue
        for node in top_level(PARSER.parse(path.read_bytes()).root_node):
            if node.type not in _SYMBOLS or b"ONNX_API" in node.text:
                continue
            name = declared_name(node)
            if name is None or "::" in name:
                continue
            if path.suffix == ".h":
                out.add(Symbol(name, None))
            elif is_file_local(node) and Symbol(name, None) not in out:
                out.add(Symbol(name, path))
    return out


def unused(cands: set[Symbol], files: list[Path]) -> set[Symbol]:
    """Return the candidates whose names only occur in their own declarations."""
    names = {s.name for s in cands}
    used: set[Symbol] = set()
    for path in files:
        spans: dict[str, list[tuple[int, int]]] = {}
        for name, start, end in symbols(path):
            spans.setdefault(name, []).append((start, end))
        skipped = comments(path)
        referenced = {
            name
            for m in re.finditer(rb"\w+", path.read_bytes())
            if (name := m.group().decode()) in names
            and not any(s <= m.start() < e for s, e in spans.get(name, []) + skipped)
        }
        used |= {
            s for s in cands if s.name in referenced and s.static_in in (None, path)
        }
    return cands - used


def is_empty(path: Path) -> bool:
    root = PARSER.parse(path.read_bytes()).root_node
    return all(n.type in _SCAFFOLDING for n in top_level(root) if n.is_named)


def remove_file(path: Path) -> None:
    """Delete `path`, its includes and its CMake entry."""
    path.unlink()
    rel = path.relative_to(REPO_ROOT).as_posix()
    include = re.compile(rf'^#include "{re.escape(rel)}"\n', re.MULTILINE)
    for other in searched_files():
        if include.search(text := other.read_text()):
            other.write_text(include.sub("", text))
    cmake_lists = path.parent / "CMakeLists.txt"
    entry = re.compile(rf"^\s+{re.escape(path.name)}\n", re.MULTILINE)
    cmake_lists.write_text(entry.sub("", cmake_lists.read_text()))
    print(f"deleted {rel}")


def main() -> None:
    files = searched_files()
    removed: set[str] = set()
    while found := unused(candidates(), files):
        for path in candidate_files():
            if not path.exists():
                continue
            names = {
                s.name for s in found if s.static_in is None or s.static_in == path
            }
            src, hits = remove(path.read_bytes(), names)
            if hits:
                path.write_bytes(src)
        removed |= {s.name for s in found}
    print(f"removed {len(removed)} symbols")
    for path in candidate_files():
        if path.exists() and is_empty(path):
            remove_file(path)


if __name__ == "__main__":
    main()
