# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from onnx import ModelProto, TensorProto, checker, helper, numpy_helper
from onnx.reference import ReferenceEvaluator


def make_sink_model(storage_type: int = TensorProto.FLOAT) -> ModelProto:
    """Build the final-logit composition, not a registered ONNX function."""
    if storage_type not in (
        TensorProto.FLOAT16,
        TensorProto.FLOAT,
        TensorProto.DOUBLE,
    ):
        raise ValueError("The example supports float16, float32, and float64.")
    compute_type = (
        TensorProto.DOUBLE if storage_type == TensorProto.DOUBLE else TensorProto.FLOAT
    )
    constants = {
        "head_shape": [1, -1, 1, 1],
        "row_axes": [0, 1, 2],
        "one": [1],
        "starts": [0],
        "ends": [-1],
        "axes": [-1],
    }
    nodes = [
        helper.make_node("Cast", ["L"], ["Lf"], to=compute_type),
        helper.make_node("Cast", ["S"], ["Sf"], to=compute_type),
        helper.make_node("Cast", ["V"], ["Vf"], to=compute_type),
        helper.make_node("Shape", ["Lf"], ["logit_shape"]),
        helper.make_node("Gather", ["logit_shape", "row_axes"], ["row_shape"], axis=0),
        helper.make_node("Concat", ["row_shape", "one"], ["sink_shape"], axis=0),
        helper.make_node("Reshape", ["Sf", "head_shape"], ["S_head"]),
        helper.make_node("Expand", ["S_head", "sink_shape"], ["S_full"]),
        helper.make_node("Concat", ["Lf", "S_full"], ["Z"], axis=-1),
        helper.make_node("Softmax", ["Z"], ["P_all"], axis=-1),
        helper.make_node("Slice", ["P_all", "starts", "ends", "axes", "one"], ["P"]),
        helper.make_node("MatMul", ["P", "Vf"], ["Yf"]),
        helper.make_node("Cast", ["Yf"], ["Y"], to=storage_type),
    ]
    graph = helper.make_graph(
        nodes,
        "attention_sink_composition",
        [
            helper.make_tensor_value_info("L", storage_type, ["B", "Hq", "Q", "K"]),
            helper.make_tensor_value_info("S", storage_type, ["Hq"]),
            helper.make_tensor_value_info("V", storage_type, ["B", "Hq", "K", "Dv"]),
        ],
        [
            helper.make_tensor_value_info("Y", storage_type, ["B", "Hq", "Q", "Dv"]),
            helper.make_tensor_value_info("P", compute_type, ["B", "Hq", "Q", "K"]),
        ],
        [
            numpy_helper.from_array(np.array(value, dtype=np.int64), name)
            for name, value in constants.items()
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=7
    )
    checker.check_model(model, full_check=True)
    return model


def main() -> None:
    model = make_sink_model()
    feeds = {
        "L": np.zeros((1, 2, 1, 2), dtype=np.float32),
        "S": np.array([0, np.log(2)], dtype=np.float32),
        "V": np.array([[[[2], [4]], [[2], [4]]]], dtype=np.float32),
    }
    output, probabilities = ReferenceEvaluator(model).run(None, feeds)
    np.testing.assert_allclose(output.flatten(), [2, 1.5])
    np.testing.assert_allclose(probabilities.sum(axis=-1).flatten(), [2 / 3, 1 / 2])
    print("Output:", output.flatten())
    print("Real probability mass:", probabilities.sum(axis=-1).flatten())


if __name__ == "__main__":
    main()
