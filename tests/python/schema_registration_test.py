# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
"""Guards against defining an operator schema without registering it.

An operator schema is *defined* with one of the ONNX_..._OPERATOR_SET_SCHEMA
macros in an onnx/defs/**/*.cc file, but it is only actually placed in
the schema registry if it is also *registered*: forward-declared and passed
to `fn(...)` inside the matching onnx/defs/operator_sets*.h header, whose
`ForEachSchema` is in turn passed to `RegisterOpSetSchema<...>()` in
`RegisterOnnxOperatorSetSchema()`. Forgetting any of these steps compiles
cleanly and silently drops the schema (see
https://github.com/onnx/onnx/pull/8301, where ReduceLogSum/ReduceLogSumExp
opset 21 were defined but never registered, so opset 21 users silently kept
getting the opset 18 schema).

A debug-build-only runtime assertion in onnx/defs/schema.cc already checks
that the *count* of defined and registered schemas match, but it only runs
when ONNX is built with DEBUG=1, which in CI only happens on two Ubuntu jobs
(ASAN, TSAN); Windows and macOS never build Debug. This module performs two
complementary checks on every platform and every build:

* `test_all_defined_schemas_are_registered` parses the headers and catches a
  missing `fn(...)` call, even when the installed ONNX binary predates the
  sources.
* `test_runtime_registry_matches_defined_schemas` compares the parsed
  definitions against `onnx.defs.get_all_schemas_with_history()`, the
  registry actually built at runtime. This additionally catches what header
  parsing cannot see: an opset class (e.g. a future `OpSet_Onnx_ver30`) that
  is never passed to `RegisterOpSetSchema<...>()`, which silently drops
  *every* schema of that opset even though each `fn(...)` call is present,
  and a schema that ends up registered under the wrong domain.
"""

from __future__ import annotations

import functools
import re
from pathlib import Path

import pytest

from onnx.defs import (
    AI_ONNX_PREVIEW_DOMAIN,
    AI_ONNX_PREVIEW_TRAINING_DOMAIN,
    ONNX_DOMAIN,
    ONNX_ML_DOMAIN,
    get_all_schemas_with_history,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFS_DIR = REPO_ROOT / "onnx" / "defs"

# class_domain: the domain argument of the schema class name, i.e. the class is
# ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(class_domain, ver, name) = name_<class_domain>_ver<ver>.
# registry_domain: the value of OpSchema.domain in the runtime registry for schemas
# defined by the macro.
# header: where the schema must be forward-declared and registered.
_MACRO_INFO = {
    "ONNX_OPERATOR_SET_SCHEMA": ("Onnx", ONNX_DOMAIN, DEFS_DIR / "operator_sets.h"),
    "ONNX_ML_OPERATOR_SET_SCHEMA": (
        "OnnxML",
        ONNX_ML_DOMAIN,
        DEFS_DIR / "operator_sets_ml.h",
    ),
    # ai.onnx.training currently defines no operators, and onnx.defs exports no
    # Python constant for its domain, so spell it out here.
    "ONNX_TRAINING_OPERATOR_SET_SCHEMA": (
        "OnnxTraining",
        "ai.onnx.training",
        DEFS_DIR / "operator_sets_training.h",
    ),
    "ONNX_PREVIEW_OPERATOR_SET_SCHEMA": (
        "OnnxPreview",
        AI_ONNX_PREVIEW_DOMAIN,
        DEFS_DIR / "operator_sets_preview.h",
    ),
    # Preview training schemas share the "OnnxPreview" class-name domain and
    # header with the preview schemas; only the registry domain differs.
    "ONNX_PREVIEW_TRAINING_OPERATOR_SET_SCHEMA": (
        "OnnxPreview",
        AI_ONNX_PREVIEW_TRAINING_DOMAIN,
        DEFS_DIR / "operator_sets_preview.h",
    ),
}
# Macros that currently have no schemas at all (ai.onnx.training defines no operators).
# Every other macro must yield at least one definition and one registration, so a regex
# or glob that silently stops matching fails loudly instead of passing with empty sets.
_MACROS_WITHOUT_SCHEMAS = {"ONNX_TRAINING_OPERATOR_SET_SCHEMA"}

_DEFINITION_RE = re.compile(
    r"\b(" + "|".join(_MACRO_INFO) + r")\(\s*([A-Za-z0-9_]+)\s*,\s*(\d+)\s*,"
)
# Matches every registration call `fn(GetOpSchema<<Class>...>())` regardless of which
# class-name macro produced <Class>; _registered_schemas then decodes the macro
# arguments. Anchored on the full fn(...) call, not just GetOpSchema<...>, so a stray
# reference to a class name elsewhere in the file (e.g. left over in a comment) isn't
# mistaken for a registration. An unknown class-name macro raises instead of being
# silently ignored.
_REGISTRATION_RE = re.compile(
    r"fn\(\s*GetOpSchema\s*<\s*([A-Za-z0-9_]+)\s*\(([^()]*)\)\s*>\s*\(\s*\)\s*\)"
)
# ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(domain, ver, name)
_FULL_CLASS_NAME_ARGS_RE = re.compile(
    r"([A-Za-z0-9_]+)\s*,\s*(\d+)\s*,\s*([A-Za-z0-9_]+)"
)
# ONNX_PREVIEW_OPERATOR_SET_SCHEMA_CLASS_NAME(ver, name) expands to the class name
# with the fixed "OnnxPreview" domain; see onnx/defs/schema.h.
_PREVIEW_CLASS_NAME_ARGS_RE = re.compile(r"(\d+)\s*,\s*([A-Za-z0-9_]+)")
_CLASS_NAME_MACROS = {
    "ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME": _FULL_CLASS_NAME_ARGS_RE,
    "ONNX_PREVIEW_OPERATOR_SET_SCHEMA_CLASS_NAME": _PREVIEW_CLASS_NAME_ARGS_RE,
}
# The class-name domain baked into ONNX_PREVIEW_OPERATOR_SET_SCHEMA_CLASS_NAME.
_PREVIEW_CLASS_DOMAIN = "OnnxPreview"

# Matches C++ // line comments and /* */ block comments, so a commented-out
# definition or fn(...) call doesn't get mistaken for a live one. String and
# character literals are kept intact: doc strings contain URLs ("https://...")
# whose // must not be treated as the start of a line comment.
_LINE_COMMENT_RE = re.compile(r"//[^\n]*")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_QUOTED_RE = re.compile(
    r"""
    "(?:\\.|[^"\\])*"      # string literal
    | '(?:\\.|[^'\\])*'    # character literal
    | R"([A-Za-z0-9_]*)\(  # raw string literal start, R"delim(
""",
    re.VERBOSE,
)


def _strip_comments(text: str) -> str:
    """Remove comments while preserving string literals."""
    out: list[str] = []
    pos = 0
    while pos < len(text):
        ch = text[pos]
        if text.startswith("//", pos):
            match = _LINE_COMMENT_RE.match(text, pos)
            pos = match.end()  # type: ignore[union-attr]
        elif text.startswith("/*", pos):
            match = _BLOCK_COMMENT_RE.match(text, pos)
            if match is None:  # unterminated block comment: drop the rest
                pos = len(text)
            else:
                pos = match.end()
        elif ch in {'"', "'"} or (ch == "R" and text.startswith('R"', pos)):
            match = _QUOTED_RE.match(text, pos)
            if match is None:
                out.append(ch)
                pos += 1
                continue
            if match.group(1) is not None:  # raw string: find )delim"
                close = f'){match.group(1)}"'
                end = text.find(close, match.end())
                pos = len(text) if end == -1 else end + len(close)
                out.append(text[match.start() : pos])
            else:
                out.append(match.group())
                pos = match.end()
        else:
            out.append(ch)
            pos += 1
    return "".join(out)


@functools.cache
def _defined_schemas() -> dict[str, set[tuple[str, int]]]:
    """Returns, per macro, the set of (name, version) pairs defined via that macro."""
    defined: dict[str, set[tuple[str, int]]] = {macro: set() for macro in _MACRO_INFO}
    for source_file in DEFS_DIR.glob("**/*.cc"):
        text = _strip_comments(source_file.read_text(encoding="utf-8"))
        for match in _DEFINITION_RE.finditer(text):
            macro, name, version = match.group(1), match.group(2), int(match.group(3))
            defined[macro].add((name, version))
    return defined


@functools.cache
def _runtime_schemas_by_domain() -> dict[str, set[tuple[str, int]]]:
    """Returns, per registry domain, the (name, version) pairs the runtime registry holds."""
    by_domain: dict[str, set[tuple[str, int]]] = {}
    for schema in get_all_schemas_with_history():
        by_domain.setdefault(schema.domain, set()).add(
            (schema.name, schema.since_version)
        )
    return by_domain


def _registered_schemas(header: Path, class_domain: str) -> set[tuple[str, int]]:
    """Returns the (name, version) pairs actually wired into `header`'s ForEachSchema for `class_domain`."""
    text = _strip_comments(header.read_text(encoding="utf-8"))
    registered = set()
    for match in _REGISTRATION_RE.finditer(text):
        macro, args = match.group(1), match.group(2)
        if macro not in _CLASS_NAME_MACROS:
            raise AssertionError(
                f"{header.relative_to(REPO_ROOT)} registers a schema via the unknown "
                f"class-name macro {macro}(...); extend _CLASS_NAME_MACROS so its "
                "registrations are not silently ignored."
            )
        parsed = _CLASS_NAME_MACROS[macro].fullmatch(args.strip())
        if parsed is None:
            raise AssertionError(
                f"Cannot parse the arguments of {macro}({args}) in "
                f"{header.relative_to(REPO_ROOT)}."
            )
        if macro == "ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME":
            domain, version, name = (
                parsed.group(1),
                int(parsed.group(2)),
                parsed.group(3),
            )
        else:
            domain, version, name = (
                _PREVIEW_CLASS_DOMAIN,
                int(parsed.group(1)),
                parsed.group(2),
            )
        if domain == class_domain:
            registered.add((name, version))
    return registered


@pytest.mark.parametrize("macro", sorted(_MACRO_INFO))
def test_all_defined_schemas_are_registered(macro: str) -> None:
    class_domain, _, header = _MACRO_INFO[macro]
    defined = _defined_schemas()[macro]
    registered = _registered_schemas(header, class_domain)

    if macro in _MACROS_WITHOUT_SCHEMAS:
        assert not defined, (
            f"{macro}(...) now defines schemas {sorted(defined)}; "
            "remove it from _MACROS_WITHOUT_SCHEMAS so this test checks them."
        )
    else:
        assert defined, (
            f"Found no {macro}(...) definitions under onnx/defs/**/*.cc; "
            "the definition regex or file glob likely no longer matches the sources."
        )
        assert registered, (
            f"Found no {class_domain} registrations in {header.relative_to(REPO_ROOT)}; "
            "the registration regex likely no longer matches the header."
        )

    missing = defined - registered
    assert not missing, (
        f"{len(missing)} operator schema(s) defined via {macro}(...) in onnx/defs/**/*.cc "
        f"are missing from {header.relative_to(REPO_ROOT)}: {sorted(missing)}. "
        "Each schema must be forward-declared as "
        f"ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME({class_domain}, <version>, <Name>) and passed to fn(GetOpSchema<...>()) "
        "inside that header's ForEachSchema, or it silently never gets registered "
        "(see https://github.com/onnx/onnx/pull/8301)."
    )


@pytest.mark.parametrize("macro", sorted(_MACRO_INFO))
def test_runtime_registry_matches_defined_schemas(macro: str) -> None:
    """Compares the parsed definitions against the registry built at runtime.

    The header check above only sees the `fn(...)` calls inside one header; it cannot
    see a missing `RegisterOpSetSchema<OpSet_Onnx_verN>()` call in
    `RegisterOnnxOperatorSetSchema()`, which silently drops *every* schema of that
    opset, or a schema registered under the wrong domain. The runtime registry is the
    single source of truth for what users actually get, so require it to hold exactly
    the schemas defined via this macro.
    """
    _, registry_domain, _ = _MACRO_INFO[macro]
    defined = _defined_schemas()[macro]
    runtime = _runtime_schemas_by_domain().get(registry_domain, set())

    unregistered = defined - runtime
    assert not unregistered, (
        f"{len(unregistered)} operator schema(s) defined via {macro}(...) are missing "
        f"from the runtime schema registry for domain {registry_domain!r}: "
        f"{sorted(unregistered)}. Either the fn(GetOpSchema<...>()) call is missing from "
        "the operator_sets*.h header, or the whole opset class was never passed to "
        "RegisterOpSetSchema<...>() in RegisterOnnxOperatorSetSchema() "
        "(see https://github.com/onnx/onnx/issues/8577)."
    )

    unexpected = runtime - defined
    assert not unexpected, (
        f"{len(unexpected)} schema(s) are in the runtime registry for domain "
        f"{registry_domain!r} but were not parsed as {macro}(...) definitions in "
        f"onnx/defs/**/*.cc: {sorted(unexpected)}. Either a schema is registered under "
        "the wrong domain, or this test's definition regex/glob no longer matches the "
        "sources it is supposed to parse."
    )
