# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import pathlib
from typing import Any

import pytest

pytest.importorskip("sphinx")


def _load_onnx_sphinx() -> Any:
    source = (
        pathlib.Path(__file__).parents[2]
        / "docs"
        / "docsgen"
        / "source"
        / "onnx_sphinx.py"
    )
    spec = importlib.util.spec_from_file_location("onnx_sphinx", source)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


onnx_sphinx = _load_onnx_sphinx()


def test_get_latest_operator_schema_without_domain() -> None:
    schemas = onnx_sphinx.get_operator_schemas("Add", version="last")

    assert len(schemas) == 1
    assert schemas[0].name == "Add"
    assert schemas[0].since_version == max(
        schema.since_version
        for schema in onnx_sphinx.get_operator_schemas("Add", version=None)
    )


def test_operator_generation_removes_stale_files(tmp_path: pathlib.Path) -> None:
    stale_operator = tmp_path / "onnx_Stale.md"
    stale_diff = tmp_path / "text_diff_Stale.rst"
    unrelated = tmp_path / "keep.txt"
    stale_operator.touch()
    stale_diff.touch()
    unrelated.touch()

    onnx_sphinx._clean_operator_docs(tmp_path)

    assert not stale_operator.exists()
    assert not stale_diff.exists()
    assert unrelated.exists()
