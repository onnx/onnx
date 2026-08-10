# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from onnx.defs import OpSchema


def generate_formal_parameter_tags(
    formal_parameter: OpSchema.FormalParameter,
) -> str:
    """Generate the tags displayed beside an operator input or output."""
    tags: list[str] = []
    if OpSchema.FormalParameterOption.Optional == formal_parameter.option:
        tags.append("optional")
    elif OpSchema.FormalParameterOption.Variadic == formal_parameter.option:
        tags.append("variadic")
        if not formal_parameter.is_homogeneous:
            tags.append("heterogeneous")

    differentiable = OpSchema.DifferentiationCategory.Differentiable
    non_differentiable = OpSchema.DifferentiationCategory.NonDifferentiable
    if differentiable == formal_parameter.differentiation_category:
        tags.append("differentiable")
    elif non_differentiable == formal_parameter.differentiation_category:
        tags.append("non-differentiable")

    return "" if not tags else f" ({', '.join(tags)})"
