# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

pytest.importorskip("sphinx")

from docs.docsgen.source import onnx_sphinx


@pytest.mark.parametrize(
    "op_name",
    [
        "BatchNormalization",
        "GreaterOrEqual",
        "InstanceNormalization",
        "LessOrEqual",
        "Range",
        "SoftmaxCrossEntropyLoss",
    ],
)
def test_get_markdown_doc_includes_backend_examples(op_name: str) -> None:
    docs, _, example_count = onnx_sphinx.get_markdown_doc(
        ".", op_name=op_name, domain="", example=True
    )

    assert example_count > 0
    assert "### Examples" in docs


def test_get_markdown_doc_keeps_example_source_unescaped() -> None:
    docs, _, example_count = onnx_sphinx.get_markdown_doc(
        ".", op_name="Attention", domain="", example=True
    )

    assert example_count > 0
    assert "&#34;" not in docs.split("### Examples", maxsplit=1)[1]


@pytest.mark.parametrize(
    ("op_name", "domain"),
    [
        ("Adagrad", "ai.onnx.preview.training"),
        ("Adam", "ai.onnx.preview.training"),
        ("FlexAttention", "ai.onnx.preview"),
        ("Momentum", "ai.onnx.preview.training"),
    ],
)
def test_get_markdown_doc_finds_top_level_domain_examples(
    op_name: str, domain: str
) -> None:
    docs, _, example_count = onnx_sphinx.get_markdown_doc(
        ".", op_name=op_name, domain=domain, example=True
    )

    assert example_count > 0
    assert "### Examples" in docs


def test_get_onnx_example_does_not_fall_back_for_unrelated_domains() -> None:
    assert onnx_sphinx.get_onnx_example("Add", "ai.onnx.ml") == {}
