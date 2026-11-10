# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
# mypy: disable-error-code="union-attr,assignment"
"""Remove the operator registrations that the generator emits.

One-off migration tool, to be run after 'move_function_names.py'.
Every registration in 'onnx/defs' must be 'generated::<domain>::<Name>_ver<N>()'
without further builder calls; anything else is reported as an error. The
registrations are removed together with the comments directly preceding them
and the include of the generated schema functions.
Run 'remove_unused_symbols.py' afterwards to remove what is left unused.

Requires 'tree-sitter' and 'tree-sitter-cpp'. Run 'pixi run lint' afterwards.

Usage:
    python tools/schema_migration/remove_registrations.py
"""

from __future__ import annotations

from registrations import (
    PARSER,
    REPO_ROOT,
    RegistrationError,
    location,
    registrations,
    source_files,
)

INCLUDE = b'#include "onnx/defs/generated/schemas.h"\n'


def main() -> None:
    total = 0
    for path in source_files():
        src = path.read_bytes()
        spans = []
        for reg in registrations(path, PARSER.parse(src).root_node):
            if reg.calls or not reg.base.text.startswith(b"generated::"):
                raise RegistrationError(f"{location(path, reg.node)}: not generated")
            statement = reg.node
            while statement.parent.type not in ("translation_unit", "declaration_list"):
                statement = statement.parent
            first = statement
            while (
                (prev := first.prev_sibling) is not None
                and prev.type == "comment"
                and prev.end_point[0] + 1 >= first.start_point[0]
            ):
                first = prev
            start = src.rfind(b"\n", 0, first.start_byte) + 1
            end = statement.end_byte
            while src[end : end + 1] in (b";", b"\n"):
                end += 1
            spans.append((start, end))
        for start, end in sorted(spans, reverse=True):
            src = src[:start] + src[end:]
        if spans:
            path.write_bytes(src.replace(INCLUDE, b""))
            print(f"{path.relative_to(REPO_ROOT)}: {len(spans)} registrations")
        total += len(spans)
    print(f"removed {total} registrations")


if __name__ == "__main__":
    main()
