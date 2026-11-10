# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
"""Guards against operator schemas that are specified but not registered.

Every operator schema has a spec file
`operator-specifications/<domain>/v<N>/<op>.md`, from which its operator set and
registration are generated. These tests compare the spec files against the registry
actually built at runtime (see https://github.com/onnx/onnx/issues/8577).
"""

from __future__ import annotations

import functools
import re
from pathlib import Path

import pytest

import onnx
from onnx.defs import (
    AI_ONNX_PREVIEW_DOMAIN,
    AI_ONNX_PREVIEW_TRAINING_DOMAIN,
    ONNX_DOMAIN,
    ONNX_ML_DOMAIN,
    get_all_schemas_with_history,
)

SPECS_DIR = Path(__file__).resolve().parents[2] / "operator-specifications"

# ai.onnx.training currently defines no operators, and onnx.defs exports no Python
# constant for its domain, so spell it out here.
DOMAINS = [
    ONNX_DOMAIN,
    ONNX_ML_DOMAIN,
    "ai.onnx.training",
    AI_ONNX_PREVIEW_DOMAIN,
    AI_ONNX_PREVIEW_TRAINING_DOMAIN,
]

_NAME_RE = re.compile(r'^name = "(\w+)"$', re.MULTILINE)


@functools.cache
def _specified_schemas(domain: str) -> set[tuple[str, int]]:
    """Return the (name, version) pairs of the spec files of `domain`."""
    specified = set()
    # The registry denotes ai.onnx by the empty string.
    for path in (SPECS_DIR / (domain or "ai.onnx")).glob("v*/*.md"):
        match = _NAME_RE.search(path.read_text(encoding="utf-8"))
        assert match is not None, f"no 'name' found in {path}"
        specified.add((match.group(1), int(path.parent.name[1:])))
    return specified


def _registered_schemas(domain: str) -> set[tuple[str, int]]:
    """Return the (name, version) pairs of `domain` in the runtime registry."""
    return {
        (schema.name, schema.since_version)
        for schema in get_all_schemas_with_history()
        if schema.domain == domain
    }


@pytest.mark.parametrize("domain", DOMAINS)
def test_runtime_registry_matches_spec_files(domain: str) -> None:
    if domain == ONNX_ML_DOMAIN and not onnx.ONNX_ML:
        pytest.skip("ML schemas are not registered in ONNX_ML=0 builds")

    specified = _specified_schemas(domain)
    registered = _registered_schemas(domain)

    unregistered = specified - registered
    assert not unregistered, (
        f"{len(unregistered)} specified schema(s) of domain {domain!r} are missing "
        f"from the runtime registry: {sorted(unregistered)}."
    )
    unexpected = registered - specified
    assert not unexpected, (
        f"{len(unexpected)} schema(s) of domain {domain!r} in the runtime registry "
        f"have no spec file: {sorted(unexpected)}."
    )
