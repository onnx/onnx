# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Generate C++ functions that return an `OpSchema` from the operator spec files.

Usage (from the `tools` directory):
    python -m operator_specs --input SPECS_DIR --output DIR

Writes `DIR/schemas.h` and one `DIR/<domain>/<op>.cc` per operator, where
`<domain>` has '.' replaced by '_' and `<op>` is lowercase. Files are only
rewritten if their content changes.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import jinja2

from operator_specs.models import resolve

TEMPLATE_DIR = Path(__file__).parent / "templates"


def generate(input_dir: Path, output_dir: Path) -> dict[Path, str]:
    """Return the content of all generated files by path."""
    paths_by_op: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for path in sorted(input_dir.glob("*/v*/*.md")):
        paths_by_op[path.parent.parent.name, path.stem].append(path)
    root = input_dir.parent
    ops = [resolve(paths, root) for _, paths in sorted(paths_by_op.items())]

    env = jinja2.Environment(  # noqa: S701 (generates C++, not HTML)
        loader=jinja2.FileSystemLoader(TEMPLATE_DIR),
        undefined=jinja2.StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    source = env.get_template("schema.cc.jinja")
    files = {
        output_dir / op.namespace / f"{op.name.lower()}.cc": source.render(op=op)
        for op in ops
    }
    files[output_dir / "schemas.h"] = env.get_template("schemas.h.jinja").render(
        ops=ops
    )
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for path, content in generate(args.input.resolve(), args.output).items():
        if path.exists() and path.read_text() == content:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


if __name__ == "__main__":
    main()
