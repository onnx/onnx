# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
# type: ignore

"""You can run a specific test by using the following syntax.

::

    python onnx/test/reference_evaluator_test.py TestReferenceEvaluator.test_function_attribute_nested_graph
"""

from __future__ import annotations

import unittest
from os import getenv

import onnx
import onnx.helper as oh
from onnx.checker import check_model
from onnx.reference import ReferenceEvaluator

# TODO (https://github.com/microsoft/onnxruntime/issues/14932): Get max supported version from onnxruntime directly
# For now, bump the version in CIs whenever there is a new onnxruntime release
ORT_MAX_IR_SUPPORTED_VERSION = int(getenv("ORT_MAX_IR_SUPPORTED_VERSION", "8"))
ORT_MAX_ONNX_OPSET_SUPPORTED_VERSION = int(
    getenv("ORT_MAX_ONNX_OPSET_SUPPORTED_VERSION", "18")
)


def get_torch():
    try:
        import torch

        return torch  # noqa: TRY300
    except ImportError:
        return None


class TestReferenceEvaluatorWithTorch(unittest.TestCase):
    @staticmethod
    def _linear_regression(opset=None):
        X = oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [None, None])
        A = oh.make_tensor_value_info("A", onnx.TensorProto.FLOAT, [None, None])
        B = oh.make_tensor_value_info("B", onnx.TensorProto.FLOAT, [None, None])
        Y = oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [None])
        node1 = oh.make_node("MatMul", ["X", "A"], ["XA"])
        node2 = oh.make_node("Add", ["XA", "B"], ["Y"])
        graph = oh.make_graph([node1, node2], "lr", [X, A, B], [Y])

        f = lambda x, a, b: a @ a + b  # noqa: ARG005, E731
        if opset is None:
            onnx_model = oh.make_model(graph)
        else:
            onnx_model = oh.make_model(
                graph, opset_imports=[oh.make_opsetid("", opset)]
            )
        try:
            check_model(onnx_model)
        except Exception as e:
            raise AssertionError(f"checker fails for\n{onnx_model}") from e
        return onnx_model, f

    @unittest.skipIf(not get_torch(), reason="torch not installed")
    def test_reference_evaluator_lr(self):
        torch = get_torch()
        lr, f = TestReferenceEvaluatorWithTorch._linear_regression()
        x = torch.Tensor([[0, 1], [2, 3]]).to(torch.float32)
        a = torch.Tensor([1, 1]).to(torch.float32)
        b = torch.Tensor([11]).to(torch.float32)
        expected = f(x, a, b)
        sess = ReferenceEvaluator(lr)
        got = sess.run(None, {"X": a, "A": a, "B": b})[0]
        self.assertIsInstance(got, torch.Tensor)
        torch.testing.assert_close(expected, got)


if __name__ == "__main__":
    unittest.main(verbosity=2)
