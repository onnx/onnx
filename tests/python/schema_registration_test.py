# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
"""Guards against defining an operator schema without registering it.

An operator schema is *defined* with one of the ONNX_..._OPERATOR_SET_SCHEMA
macros in onnx/defs/**/defs.cc or old.cc, but it is only actually placed in
the schema registry if it is also *registered*: forward-declared and passed
to `fn(...)` inside the matching onnx/defs/operator_sets*.h header. Forgetting
the second step compiles cleanly and silently drops the schema (see
https://github.com/onnx/onnx/pull/8301, where ReduceLogSum/ReduceLogSumExp
opset 21 were defined but never registered, so opset 21 users silently kept
getting the opset 18 schema).

A debug-build-only runtime assertion in onnx/defs/schema.cc already checks
that the *count* of defined and registered schemas match, but it only runs
when ONNX is built with DEBUG=1, which in CI only happens on two Ubuntu jobs
(ASAN, TSAN); Windows and macOS never build Debug. This test performs the
same check by parsing source directly, so it runs on every platform and every
build, and reports exactly which (name, version) pairs are missing.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFS_DIR = REPO_ROOT / "onnx" / "defs"

# domain: the first argument of ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(domain, ver, name)
# used by GetOpSchema<...> calls in the header's ForEachSchema.
# header: where the schema must be forward-declared and registered.
_MACRO_DOMAIN_AND_HEADER = {
    "ONNX_OPERATOR_SET_SCHEMA": ("Onnx", DEFS_DIR / "operator_sets.h"),
    "ONNX_ML_OPERATOR_SET_SCHEMA": ("OnnxML", DEFS_DIR / "operator_sets_ml.h"),
    "ONNX_TRAINING_OPERATOR_SET_SCHEMA": (
        "OnnxTraining",
        DEFS_DIR / "operator_sets_training.h",
    ),
    "ONNX_PREVIEW_OPERATOR_SET_SCHEMA": (
        "OnnxPreview",
        DEFS_DIR / "operator_sets_preview.h",
    ),
    "ONNX_PREVIEW_TRAINING_OPERATOR_SET_SCHEMA": (
        "OnnxPreview",
        DEFS_DIR / "operator_sets_preview.h",
    ),
}

_DEFINITION_RE = re.compile(
    r"\b("
    + "|".join(_MACRO_DOMAIN_AND_HEADER)
    + r")\(\s*([A-Za-z0-9_]+)\s*,\s*(\d+)\s*,"
)
# Registered via the usual GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(domain, ver, name)>
_REGISTRATION_RE = re.compile(
    r"GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME\(\s*([A-Za-z0-9_]+)\s*,\s*(\d+)\s*,\s*([A-Za-z0-9_]+)\s*\)>"
)
# Preview headers alternatively use an alias with (ver, name) instead of (domain, ver, name);
# see ONNX_PREVIEW_OPERATOR_SET_SCHEMA_CLASS_NAME in onnx/defs/schema.h.
_PREVIEW_REGISTRATION_RE = re.compile(
    r"GetOpSchema<ONNX_PREVIEW_OPERATOR_SET_SCHEMA_CLASS_NAME\(\s*(\d+)\s*,\s*([A-Za-z0-9_]+)\s*\)>"
)


def _defined_schemas() -> dict[str, set[tuple[str, int]]]:
    """Returns, per macro, the set of (name, version) pairs defined via that macro."""
    defined: dict[str, set[tuple[str, int]]] = {
        macro: set() for macro in _MACRO_DOMAIN_AND_HEADER
    }
    for source_file in (*DEFS_DIR.glob("**/defs.cc"), *DEFS_DIR.glob("**/old.cc")):
        text = source_file.read_text(encoding="utf-8")
        for match in _DEFINITION_RE.finditer(text):
            macro, name, version = match.group(1), match.group(2), int(match.group(3))
            defined[macro].add((name, version))
    return defined


def _registered_schemas(header: Path, domain: str) -> set[tuple[str, int]]:
    """Returns the (name, version) pairs actually wired into `header`'s ForEachSchema for `domain`."""
    text = header.read_text(encoding="utf-8")
    registered = set()
    for match in _REGISTRATION_RE.finditer(text):
        dom, version, name = match.group(1), int(match.group(2)), match.group(3)
        if dom == domain:
            registered.add((name, version))
    if domain == "OnnxPreview":
        for match in _PREVIEW_REGISTRATION_RE.finditer(text):
            version, name = int(match.group(1)), match.group(2)
            registered.add((name, version))
    return registered


@pytest.mark.parametrize("macro", sorted(_MACRO_DOMAIN_AND_HEADER))
def test_all_defined_schemas_are_registered(macro: str) -> None:
    domain, header = _MACRO_DOMAIN_AND_HEADER[macro]
    defined = _defined_schemas()[macro]
    registered = _registered_schemas(header, domain)

    missing = defined - registered
    assert not missing, (
        f"{len(missing)} operator schema(s) defined via {macro}(...) in onnx/defs/**/{{defs,old}}.cc "
        f"are missing from {header.relative_to(REPO_ROOT)}: {sorted(missing)}. "
        "Each schema must be forward-declared as "
        f"ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME({domain}, <version>, <Name>) and passed to fn(GetOpSchema<...>()) "
        "inside that header's ForEachSchema, or it silently never gets registered "
        "(see https://github.com/onnx/onnx/pull/8301)."
    )
