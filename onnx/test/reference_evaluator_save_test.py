# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
# type: ignore
# pylint: disable=C3001,C0302,C0415,R0904,R0913,R0914,R0915,W0221,W0707

import os
import unittest

import numpy as np

from onnx import TensorProto
from onnx.checker import check_model
from onnx.defs import onnx_opset_version
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_opsetid,
    make_tensor_value_info,
)
from onnx.reference import ReferenceEvaluator
from onnx.backend.test.loader import load_model_tests
from onnx.reference.reference_backend import create_reference_backend


class TestReferenceEvaluatorSave(unittest.TestCase):
    @staticmethod
    def _linear_regression(min_value=-1.0, max_value=1.0):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        A = make_tensor_value_info("A", TensorProto.FLOAT, [None, None])
        B = make_tensor_value_info("B", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        node1 = make_node("MatMul", ["X", "A"], ["XA"])
        node2 = make_node("Add", ["XA", "B"], ["Y_clip"])
        node2 = make_node("Add", ["XA", "B"], ["Y"])
        graph = make_graph([node1, node2], "lr", [X, A, B], [Y])
        onnx_model = make_model(
            graph, opset_imports=[make_opsetid("", onnx_opset_version())]
        )
        try:
            check_model(onnx_model)
        except Exception as e:
            raise AssertionError(f"checker fails for\n{onnx_model}") from e
        return onnx_model

    def test_reference_evaluator_save(self):
        onx = self._linear_regression()
        root = "."
        path = os.path.join(root, "reference_evaluator_test_save")
        ref = ReferenceEvaluator(onx, save_intermediate=path)
        x = np.array([[0, 1], [2, 3]], dtype=np.float32)
        a = np.array([1, 1], dtype=np.float32)
        b = np.array([11], dtype=np.float32)
        ref.run(None, {"X": x, "A": a, "B": b})
        self.assertTrue(os.path.exists(path))
        examples = load_model_tests(root, "reference_evaluator_test_save")
        self.assertEqual(len(examples), 2)

        backend = create_reference_backend(
            path_to_test=root, kind="reference_evaluator_test_save"
        )
        backend.exclude("cuda")
        tests = backend.tests()
        names = []
        for m in dir(tests):
            if m.startswith("test_"):
                test = getattr(tests, m)
                try:
                    test()
                    names.append(m)
                except unittest.case.SkipTest:
                    continue
        self.assertEqual(names, ["test_node_0_MatMul_cpu", "test_node_1_Add_cpu"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
