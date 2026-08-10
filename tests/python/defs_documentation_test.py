# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from onnx.defs import OpSchema
from onnx.defs._documentation import generate_formal_parameter_tags


@pytest.mark.parametrize(
    ("option", "is_homogeneous", "expected"),
    [
        (OpSchema.FormalParameterOption.Single, True, ""),
        (OpSchema.FormalParameterOption.Optional, True, " (optional)"),
        (OpSchema.FormalParameterOption.Variadic, True, " (variadic)"),
        (
            OpSchema.FormalParameterOption.Variadic,
            False,
            " (variadic, heterogeneous)",
        ),
    ],
)
def test_generate_formal_parameter_tags(
    option: OpSchema.FormalParameterOption,
    is_homogeneous: bool,
    expected: str,
) -> None:
    parameter = OpSchema.FormalParameter(
        "input",
        "T",
        "An input.",
        param_option=option,
        is_homogeneous=is_homogeneous,
    )

    assert generate_formal_parameter_tags(parameter) == expected


def test_generate_formal_parameter_differentiation_tag() -> None:
    parameter = OpSchema.FormalParameter(
        "input",
        "T",
        "An input.",
        differentiation_category=OpSchema.DifferentiationCategory.Differentiable,
    )

    assert generate_formal_parameter_tags(parameter) == " (differentiable)"
