# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Generate C++ functions that fill an `OpSchema` from the operator spec files.

Usage (from the `tools` directory):
    python -m operator_specs --input SPECS_DIR --output DIR

Writes `DIR/schemas.h`, one `DIR/<domain>/<op>.cc` per operator, where
`<domain>` has '.' replaced by '_' and `<op>` is lowercase, and the operator
set classes and their registration functions `DIR/opsets_*.h` included by
`onnx/defs/operator_sets*.h`. Files are only rewritten if their content
changes.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import jinja2

from operator_specs.models import Operator, resolve

TEMPLATE_DIR = Path(__file__).parent / "templates"


@dataclass(frozen=True)
class OpsetHeader:
    """An operator set header included by `onnx/defs/operator_sets*.h`."""

    prefix: str
    """Domain in the class names `OpSet_<prefix>_ver<N>`."""
    domains: set[str]
    """Spec domains whose operators the header contains."""
    register: str
    """Name of the function that registers all operator sets."""
    versioned: bool
    """Whether to emit the registration overload for a target version."""


OPSET_HEADERS = {
    "opsets_onnx.h": OpsetHeader(
        "Onnx", {"ai.onnx"}, "RegisterOnnxOperatorSetSchema", versioned=True
    ),
    "opsets_onnx_ml.h": OpsetHeader(
        "OnnxML", {"ai.onnx.ml"}, "RegisterOnnxMLOperatorSetSchema", versioned=False
    ),
    "opsets_onnx_preview.h": OpsetHeader(
        "OnnxPreview",
        {"ai.onnx.preview", "ai.onnx.preview.training"},
        "RegisterOnnxPreviewOperatorSetSchema",
        versioned=False,
    ),
}


def opsets(ops: list[Operator], domains: set[str]) -> list[tuple[int, list[str]]]:
    """Return the sorted operator names by opset version of `domains`."""
    names: dict[int, list[str]] = defaultdict(list)
    for op in ops:
        if op.domain in domains:
            for v in op.versions:
                names[v.since_version].append(op.name)
    return [(v, sorted(names[v])) for v in sorted(names)]


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
    opsets_template = env.get_template("opsets.h.jinja")
    for name, header in OPSET_HEADERS.items():
        files[output_dir / name] = opsets_template.render(
            prefix=header.prefix,
            register=header.register,
            versioned=header.versioned,
            opsets=opsets(ops, header.domains),
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
