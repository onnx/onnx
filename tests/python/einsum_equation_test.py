# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from onnx import TensorProto, helper, shape_inference


@pytest.mark.parametrize("version", [12, 28])
@pytest.mark.parametrize(
    ("equation", "input_shapes", "expected_shape"),
    [
        ("->", [()], ()),
        (",->", [(), ()], ()),
        (",,->", [(), (), ()], ()),
        (",i->i", [(), (3,)], (3,)),
        ("i,->i", [(3,), ()], (3,)),
        (",i,", [(), (3,), ()], (3,)),
        (" i , -> i ", [(3,), ()], (3,)),
        ("ij,jk->ik", [(2, 3), (3, 4)], (2, 4)),
    ],
)
def test_equation_operand_slices(
    version: int,
    equation: str,
    input_shapes: list[tuple[int, ...]],
    expected_shape: tuple[int, ...],
) -> None:
    inputs = [
        helper.make_tensor_value_info(f"X{i}", TensorProto.FLOAT, shape)
        for i, shape in enumerate(input_shapes)
    ]
    graph = helper.make_graph(
        [
            helper.make_node(
                "Einsum", [x.name for x in inputs], ["Y"], equation=equation
            )
        ],
        "einsum_equation",
        inputs,
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", version)])
    inferred = shape_inference.infer_shapes(model, strict_mode=True)
    assert inferred.graph.output[0] == helper.make_tensor_value_info(
        "Y", TensorProto.FLOAT, expected_shape
    )
