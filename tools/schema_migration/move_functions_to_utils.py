# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
# mypy: disable-error-code="union-attr,assignment,arg-type"
"""Move the named functions passed to function setters into '<area>/utils.{h,cc}'.

One-off migration tool, to be run after 'name_functions.py'. Every type and
shape inference, data propagation and context-dependent function body builder
function in
'onnx/defs/<area>/{defs,old,utils}.cc' is defined in '<area>/utils.cc' and
declared in '<area>/utils.h', within the namespace
'ONNX_NAMESPACE::defs::<area>::utils'. This includes the functions of the
public API ('ONNX_API'), which need to be restored as wrappers afterwards.

File-local functions and variables that the moved functions depend on are
moved along with them. They stay 'static' unless code remaining in
'{defs,old}.cc' uses them, in which case they are declared in the header.
References in the remaining code and forward declarations in 'tests/cpp' are
updated accordingly.

Requires 'tree-sitter' and 'tree-sitter-cpp'. Run 'pixi run lint' afterwards.

Usage:
    python tools/schema_migration/move_functions_to_utils.py
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import tree_sitter_cpp
from tree_sitter import Language, Node, Parser

if TYPE_CHECKING:
    from collections.abc import Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFS_DIR = REPO_ROOT / "onnx" / "defs"
TESTS_DIR = REPO_ROOT / "tests" / "cpp"
PARSER = Parser(Language(tree_sitter_cpp.language()))

HEADER = "// Copyright (c) ONNX Project Contributors\n//\n// SPDX-License-Identifier: Apache-2.0\n"
INFERENCE_SIGNATURE = re.compile(
    r"(?:(?:static|inline|ONNX_API)\s+)*void\s+\w+\s*\(\s*(?:ONNX_NAMESPACE::)?"
    r"(?:Inference|DataPropagation)Context\s*&\s*\w*\s*\)",
    re.DOTALL,
)
BUILDER_SIGNATURE = re.compile(
    r"(?:(?:static|inline|ONNX_API)\s+)*bool\s+\w+\s*\(\s*"
    r"const\s+FunctionBodyBuildContext\s*&\s*\w*\s*,\s*"
    r"const\s+OpSchema\s*&\s*\w*\s*,\s*"
    r"FunctionProto\s*&\s*\w*\s*\)",
    re.DOTALL,
)
NOLINT_LINKAGE = "// NOLINTNEXTLINE(misc-use-internal-linkage)\n"

# Includes copied from the source files are kept if the moved code matches the pattern.
INCLUDE_USES = {
    "<algorithm>": r"std::(sort|max|min|find|find_if|all_of|any_of|none_of|count|copy|fill|"
    r"transform|reverse|max_element|min_element|equal|unique|swap|clamp|accumulate)\b",
    "<cassert>": r"\bassert\(",
    "<cmath>": r"std::(floor|ceil|sqrt|pow|abs|fabs|log|exp|round|isnan|isinf)|\b(floor|ceil|sqrtf?)\(",
    "<functional>": r"std::function",
    "<limits>": r"numeric_limits",
    "<optional>": r"std::optional|std::nullopt",
    "<string>": r"std::string\b|to_string",
    "<unordered_map>": r"unordered_map",
    "<unordered_set>": r"unordered_set",
    "<utility>": r"std::(move|pair|make_pair|swap)",
    "<vector>": r"std::vector",
    '"onnx/common/assertions.h"': r"ONNX_ASSERT|fail_check|ONNX_THROW",
    '"onnx/common/safe_math.h"': r"Safe\w+|safe_\w+|checked_\w+|\bCheck\w*Overflow",
    '"onnx/defs/data_propagators.h"': r"Propagator|appendDimToTensorShapeProto|axisIsZero|MathOpTwoIntegers",
    '"onnx/defs/function.h"': r"FunctionBuilder|FunctionBodyBuildContext",
    # Streaming protos into error messages requires the printer.
    '"onnx/defs/printer.h"': r"fail_shape_inference|fail_type_inference",
    '"onnx/defs/tensor_proto_util.h"': r"ParseData|ToTensor",
}


def walk(node: Node):
    yield node
    for child in node.children:
        yield from walk(child)


def top_level(node: Node):
    """Yield top-level items, descending into namespaces and preprocessor conditionals."""
    for child in node.children:
        if child.type == "namespace_definition":
            yield from top_level(child.child_by_field_name("body"))
        elif child.type in ("preproc_ifdef", "preproc_if"):
            yield from top_level(child)
        else:
            yield child


def item_name(node: Node) -> str | None:
    """Return the name of a function definition or variable declaration."""
    if node.type not in ("function_definition", "declaration"):
        return None
    decl = node.child_by_field_name("declarator")
    target = (
        "function_declarator" if node.type == "function_definition" else "identifier"
    )
    while decl is not None and decl.type != target:
        decl = decl.child_by_field_name("declarator")
    if decl is None:
        return None
    if node.type == "function_definition":
        decl = decl.child_by_field_name("declarator")
    return str(decl.text.decode().split("::")[-1])


def name_of(node: Node) -> str:
    name = item_name(node)
    if name is None:
        raise ValueError(f"unnamed item: {node.text.decode()[:80]}")
    return name


def signature(node: Node, src: bytes) -> str:
    return (
        src[node.start_byte : node.child_by_field_name("body").start_byte]
        .decode()
        .strip()
    )


def is_moved_function(node: Node, src: bytes) -> bool:
    if node.type != "function_definition" or node.child_by_field_name("body") is None:
        return False
    sig = signature(node, src)
    return any(p.fullmatch(sig) for p in (INFERENCE_SIGNATURE, BUILDER_SIGNATURE))


def is_public(node: Node, src: bytes) -> bool:
    return node.type == "function_definition" and "ONNX_API" in signature(node, src)


def extent(node: Node, src: bytes) -> tuple[int, int]:
    """Return the byte range of `node`'s lines, including directly preceding comments."""
    first = node
    prev = node.prev_sibling
    while (
        prev is not None
        and prev.type == "comment"
        and prev.end_point[0] + 1 >= first.start_point[0]
    ):
        first = prev
        prev = prev.prev_sibling
    start = src.rfind(b"\n", 0, first.start_byte) + 1
    end = src.find(b"\n", node.end_byte)
    end = len(src) if end < 0 else end + 1
    if src[end : end + 1] == b"\n":
        end += 1
    return start, end


def identifiers(node: Node) -> set[str]:
    return {n.text.decode() for n in walk(node) if n.type == "identifier"}


def strip_storage(sig: str) -> str:
    return re.sub(r"^(?:(?:static|inline|ONNX_API)\s+)+", "", sig)


def exported_definition(node: Node, src: bytes, start: int, end: int) -> str:
    """Return the text of a function definition without internal linkage."""
    text = src[start:end].decode().replace(NOLINT_LINKAGE, "")
    head = src[node.start_byte : node.child_by_field_name("body").start_byte].decode()
    return text.replace(head, strip_storage(head.strip()) + " ", 1)


def namespace_of(node: Node) -> str:
    names = []
    while node is not None:
        if node.type == "namespace_definition":
            names.append(node.child_by_field_name("name").text.decode())
        node = node.parent
    return "::".join(reversed(names))


def add_include(text: str, header: str) -> str:
    line = f'#include "{header}"'
    if line in text:
        return text
    last = list(re.finditer(r"^#include .*$", text, re.MULTILINE))[-1]
    return text[: last.end()] + "\n" + line + text[last.end() :]


def append_to_namespace(text: str, namespace: str, addition: str) -> str:
    """Append `addition` to the last block of `namespace`, which is created if missing."""
    ends = list(
        re.finditer(rf"^}} // namespace {re.escape(namespace)}$", text, re.MULTILINE)
    )
    if ends:
        end = ends[-1].start()
        return f"{text[:end].rstrip()}\n\n{addition}\n{text[end:]}"
    block = f"namespace {namespace} {{\n\n{addition}\n}} // namespace {namespace}"
    outer = list(re.finditer(r"^} // namespace ONNX_NAMESPACE$", text, re.MULTILINE))
    if outer:
        end = outer[-1].end()
        return f"{text[:end]}\n\n{block}{text[end:]}"
    return f"{text.rstrip()}\n\n{block}\n"


def sort_includes(text: str, own: str) -> str:
    """Group the include block into the own header, system headers and project headers."""
    lines = text.splitlines(keepends=True)
    idx = [i for i, line in enumerate(lines) if line.startswith("#include ")]
    first, last = idx[0], idx[-1]
    if any(
        line.strip() and not line.startswith("#include ")
        for line in lines[first : last + 1]
    ):
        return text
    includes = list(
        dict.fromkeys(line.strip() for line in lines[first : last + 1] if line.strip())
    )
    own_line = f'#include "{own}"'
    groups = [
        [i for i in includes if i == own_line],
        sorted(i for i in includes if "<" in i),
        sorted(i for i in includes if "<" not in i and i != own_line),
    ]
    block = "\n\n".join("\n".join(g) for g in groups if g) + "\n"
    return "".join(lines[:first]) + block + "".join(lines[last + 1 :])


def remove_empty_namespaces(text: str) -> str:
    return re.sub(
        r"^namespace ONNX_NAMESPACE \{\s*\} // namespace ONNX_NAMESPACE\n\n?",
        "",
        text,
        flags=re.MULTILINE,
    )


def remove_empty_sections(text: str) -> str:
    """Remove section banners that are directly followed by another banner or the namespace end."""
    banner = r"// -{10,}\n// .*\n// -{10,}\n\n"
    return re.sub(rf"{banner}(?=// -{{10,}}\n|}} // namespace)", "", text)


def ensure_utils(area: str, namespace: str) -> None:
    """Create '<area>/utils.{h,cc}' if missing and add them to the CMake target."""
    area_dir = DEFS_DIR / area
    header, source = area_dir / "utils.h", area_dir / "utils.cc"
    if not header.exists():
        header.write_text(
            f'{HEADER}\n#pragma once\n\n#include "onnx/defs/schema.h"\n\n'
            f"namespace {namespace} {{\n}} // namespace {namespace}\n"
        )
    if not source.exists():
        source.write_text(
            f'{HEADER}\n#include "onnx/defs/{area}/utils.h"\n\n'
            f"namespace {namespace} {{\n}} // namespace {namespace}\n"
        )
    cmake = area_dir / "CMakeLists.txt"
    text = cmake.read_text()
    for name in ("utils.cc", "utils.h"):
        if not re.search(rf"^\s+{re.escape(name)}$", text, re.MULTILINE):
            text = text.replace("\n)", f"\n    {name}\n)", 1)
    cmake.write_text(text)


def qualify_references(src: bytes, names: set[str], qualifier: str) -> bytes:
    """Qualify unqualified references to `names` in `src` by `qualifier`."""
    edits = []
    for node in walk(PARSER.parse(src).root_node):
        if node.type != "identifier" or node.text.decode() not in names:
            continue
        name = node.text.decode()
        parent = node.parent
        if parent.type == "function_declarator" or namespace_of(node).endswith(
            qualifier.rstrip(":")
        ):
            continue
        target = node
        if parent.type == "qualified_identifier":
            if parent.child_by_field_name("scope").text.decode() != "ONNX_NAMESPACE":
                continue
            target = parent
        edits.append((target.start_byte, target.end_byte, f"{qualifier}{name}"))
    for start, end, new in sorted(edits, reverse=True):
        src = src[:start] + new.encode() + src[end:]
    return src


def move_area(
    area: str, is_moved: Callable[[Node, bytes], bool] = is_moved_function
) -> dict[str, str]:
    """Move the functions of `area` selected by `is_moved`.

    Return the qualified names of all exported functions.
    """
    namespace = f"ONNX_NAMESPACE::defs::{area}::utils"
    qualifier = f"defs::{area}::utils::"
    area_dir = DEFS_DIR / area
    paths = [area_dir / n for n in ("defs.cc", "old.cc") if (area_dir / n).exists()]
    sources = {p: p.read_bytes() for p in paths}
    items = {p: list(top_level(PARSER.parse(s).root_node)) for p, s in sources.items()}

    exported: set[str] = set()
    header_vars: set[int] = set()
    moved: dict[Path, set[int]] = {}
    for path, nodes in items.items():
        src = sources[path]
        local: dict[str, list[Node]] = {}
        for node in nodes:
            name = item_name(node)
            if name is not None:
                local.setdefault(name, []).append(node)
        seeds = [n for n in nodes if is_moved(n, src)]
        move = {n.id for n in seeds}
        stack = list(seeds)
        while stack:
            node = stack.pop()
            for ident in identifiers(node.child_by_field_name("body") or node):
                for dep in local.get(ident, []):
                    if dep.id not in move and not is_public(dep, src):
                        move.add(dep.id)
                        stack.append(dep)
        moved[path] = move
        exported |= {name_of(n) for n in seeds}
        remaining = set().union(*(identifiers(n) for n in nodes if n.id not in move))
        for node in nodes:
            name = item_name(node)
            if node.id in move and name is not None and name in remaining:
                exported.add(name)
                if node.type != "function_definition":
                    header_vars.add(node.id)

    utils_cc, utils_h = area_dir / "utils.cc", area_dir / "utils.h"
    if any(moved.values()):
        ensure_utils(area, namespace)
    if not utils_cc.exists():
        return {}
    usrc = utils_cc.read_bytes()
    header = utils_h.read_text()
    decls: list[str] = []
    definitions: list[str] = []
    includes: list[str] = []

    # Existing selected functions in 'utils.cc'.
    edits = []
    for node in top_level(PARSER.parse(usrc).root_node):
        if not is_moved(node, usrc):
            continue
        name = name_of(node)
        exported.add(name)
        decls.append(strip_storage(signature(node, usrc)) + ";")
        if namespace_of(node).endswith(f"defs::{area}::utils"):
            body = node.child_by_field_name("body")
            edits.append(
                (
                    node.start_byte,
                    body.start_byte,
                    strip_storage(signature(node, usrc)) + " ",
                )
            )
        else:
            start, end = extent(node, usrc)
            definitions.append(
                exported_definition(node, usrc, start, end).rstrip("\n") + "\n"
            )
            edits.append((start, end, ""))
            header = re.sub(
                rf"^[^\n;]*\b{name}\((?:ONNX_NAMESPACE::)?\w+Context& ctx\);\n",
                "",
                header,
                flags=re.MULTILINE,
            )
    for start, end, new in sorted(edits, reverse=True):
        usrc = usrc[:start] + new.encode() + usrc[end:]

    for path in paths:
        src = sources[path]
        cuts = []
        for node in items[path]:
            if node.id not in moved[path]:
                continue
            start, end = extent(node, src)
            cuts.append((start, end))
            if node.id in header_vars:
                decls.append(
                    re.sub(
                        r"\bstatic constexpr\b",
                        "inline constexpr",
                        src[start:end].decode().strip(),
                    )
                )
                continue
            if node.type == "function_definition" and item_name(node) in exported:
                definitions.append(
                    exported_definition(node, src, start, end).rstrip("\n") + "\n"
                )
                decls.append(strip_storage(signature(node, src)) + ";")
            else:
                definitions.append(src[start:end].decode().rstrip("\n") + "\n")
        for line in re.findall(r"^#include [<\"].*[>\"]$", src.decode(), re.MULTILINE):
            if line not in includes:
                includes.append(line)
        for start, end in sorted(cuts, reverse=True):
            src = src[:start] + src[end:]
        src = qualify_references(src, exported, qualifier)
        text = remove_empty_sections(src.decode())
        if src != sources[path]:
            text = add_include(text, f"onnx/defs/{area}/utils.h")
        path.write_text(text)

    if not decls and not definitions:
        return {}
    text = usrc.decode()
    if definitions:
        block = "\n".join(definitions)
        code = re.sub(r"//.*|^#include .*", "", text + block, flags=re.MULTILINE)
        for line in includes:
            pattern = INCLUDE_USES.get(line.removeprefix("#include "))
            if line not in text and pattern is not None and re.search(pattern, code):
                last = list(re.finditer(r"^#include .*$", text, re.MULTILINE))[-1]
                text = text[: last.end()] + "\n" + line + text[last.end() :]
        text = append_to_namespace(text, namespace, block)
    text = sort_includes(remove_empty_namespaces(text), f"onnx/defs/{area}/utils.h")
    text = qualify_references(text.encode(), exported, qualifier).decode()
    utils_cc.write_text(text)

    new_decls = [
        d
        for d in decls
        if not (m := re.match(r"(void|bool) (\w+)\(", d))
        or not re.search(rf"^{m[1]} {m[2]}\(", header, re.MULTILINE)
    ]
    if new_decls:
        header = append_to_namespace(header, namespace, "\n".join(new_decls) + "\n")
    utils_h.write_text(remove_empty_namespaces(header))
    return {name: qualifier + name for name in exported}


def update_tests(qualified: dict[str, str]) -> None:
    """Replace forward declarations of moved functions in the C++ tests by includes."""
    for path in sorted(TESTS_DIR.glob("*.cc")):
        text = path.read_text()
        declared = re.findall(
            r"^void (\w+)\(InferenceContext& ctx\);\n", text, re.MULTILINE
        )
        declared = [name for name in declared if name in qualified]
        if not declared:
            continue
        for name in declared:
            text = re.sub(
                rf"(?:^//.*\n)*^void {name}\(InferenceContext& ctx\);\n",
                "",
                text,
                flags=re.MULTILINE,
            )
            text = re.sub(rf"(?<![:\w]){name}\(ctx\)", f"{qualified[name]}(ctx)", text)
            area = qualified[name].split("::")[1]
            text = add_include(text, f"onnx/defs/{area}/utils.h")
        text = re.sub(r"(namespace ONNX_NAMESPACE \{\n)\n+", r"\1", text)
        path.write_text(sort_includes(text, ""))


def main() -> None:
    qualified: dict[str, str] = {}
    for area_dir in sorted(DEFS_DIR.iterdir()):
        if (area_dir / "defs.cc").exists():
            qualified |= move_area(area_dir.name)
    update_tests(qualified)


if __name__ == "__main__":
    main()
