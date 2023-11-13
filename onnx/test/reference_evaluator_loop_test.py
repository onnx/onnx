# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
# type: ignore
import unittest

import numpy as np

from onnx import TensorProto
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_opsetid,
    make_tensor_value_info,
)
from onnx.numpy_helper import from_array
from onnx.reference import ReferenceEvaluator


def create_model():
    """
    The following model is equivalent to the following function.

    .. code-block:: python

        from onnx import TensorProto
        from onnx.helper import make_tensor

        from onnxscript import script
        from onnxscript.onnx_opset import opset15 as op
        from onnxscript.onnx_types import FLOAT

        @script()
        def loop_range_cond_only(A: FLOAT["N"]) -> FLOAT["N"]:
            T = A
            cond = op.Constant(value=make_tensor("true", TensorProto.BOOL, [1], [1]))
            while cond:
                T = T + A
                cond = op.ReduceSum(T) > -10
            return T

        model = loop_range_cond_only.to_model_proto()
    """
    opset_imports = [
        make_opsetid("", 15),
    ]
    inputs = []
    outputs = []
    nodes = []
    initializers = []
    sparse_initializers = []
    functions = []
    inputs.append(make_tensor_value_info("A", TensorProto.FLOAT, shape=("N",)))
    nodes.append(
        make_node(
            "Constant",
            [],
            ["cond"],
            value=from_array(np.array([True], dtype=np.bool_), name="value"),
        )
    )
    nodes.append(
        make_node(
            "Constant",
            [],
            ["true"],
            value=from_array(np.array(True, dtype=np.bool_), name="value"),
        )
    )

    def _make_local_graph_body():
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        inputs.append(
            make_tensor_value_info("infinite_loop", TensorProto.INT64, shape=[])
        )
        inputs.append(make_tensor_value_info("cond", TensorProto.BOOL, shape=[]))
        inputs.append(make_tensor_value_info("T", TensorProto.UNDEFINED, []))
        nodes.append(make_node("Add", ["T", "A"], ["T_0"]))
        nodes.append(make_node("ReduceSum", ["T_0"], ["tmp"]))
        nodes.append(
            make_node(
                "Constant",
                [],
                ["int64_m10"],
                value=from_array(np.array(-10, dtype=np.int64), name="value"),
            )
        )
        nodes.append(make_node("CastLike", ["int64_m10", "tmp"], ["int64_m10_cast"]))
        nodes.append(make_node("Greater", ["tmp", "int64_m10_cast"], ["cond_1"]))
        nodes.append(make_node("Identity", ["cond_1"], ["cond_out"]))
        outputs.append(make_tensor_value_info("cond_out", TensorProto.BOOL, shape=[]))
        outputs.append(make_tensor_value_info("T_0", TensorProto.UNDEFINED, []))
        graph = make_graph(
            nodes,
            "loop_body",
            inputs,
            outputs,
            initializers,
            sparse_initializer=sparse_initializers,
        )
        return graph

    body = _make_local_graph_body()
    nodes.append(make_node("Loop", ["", "true", "A"], ["T_2"], body=body))
    outputs.append(make_tensor_value_info("T_2", TensorProto.FLOAT, shape=("N",)))
    graph = make_graph(
        nodes,
        "loop_range_cond_only",
        inputs,
        outputs,
        initializers,
        sparse_initializer=sparse_initializers,
    )
    model = make_model(graph, functions=functions, opset_imports=opset_imports)
    return model


class TestReferenceEvaluatorLoop(unittest.TestCase):
    def test_loop_fft(self):
        model = create_model()
        session = ReferenceEvaluator(model, verbose=10)
        session.run(None, {"A": -np.arange(10).astype(np.float32)})


if __name__ == "__main__":
    unittest.main(verbosity=2)
