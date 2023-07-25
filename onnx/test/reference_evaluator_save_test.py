# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import unittest

import numpy as np

from onnx import TensorProto
from onnx.backend.test.loader import load_model_tests
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
from onnx.reference.reference_backend import (
    ReferenceEvaluatorBackend,
    create_reference_backend,
)

try:
    from onnxruntime import InferenceSession
except ImportError:
    # onnxruntime not installed.
    InferenceSession = None


class TestReferenceEvaluatorSave(unittest.TestCase):
    @staticmethod
    def _linear_regression(previous_opset=False):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        A = make_tensor_value_info("A", TensorProto.FLOAT, [None, None])
        B = make_tensor_value_info("B", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        node1 = make_node("MatMul", ["X", "A"], ["XA"])
        node2 = make_node("Add", ["XA", "B"], ["Y"])
        graph = make_graph([node1, node2], "lr", [X, A, B], [Y])
        onnx_model = make_model(
            graph,
            opset_imports=[
                make_opsetid("", onnx_opset_version() - (1 if previous_opset else 0))
            ],
        )
        try:
            check_model(onnx_model)
        except Exception as e:
            raise AssertionError(f"checker fails for\n{onnx_model}") from e
        return onnx_model

    def test_reference_evaluator_save(self):
        onx = self._linear_regression()
        with tempfile.TemporaryDirectory() as root:
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
            for att in dir(tests):
                if att.startswith("test_"):
                    test = getattr(tests, att)
                    try:
                        test()
                        names.append(att)
                    except unittest.case.SkipTest:
                        continue
            self.assertEqual(names, ["test_node_0_MatMul_cpu", "test_node_1_Add_cpu"])

    def test_reference_evaluator_custom_runtime_save(self):
        onx = self._linear_regression()
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "reference_evaluator_test_save_ref")
            ref = ReferenceEvaluator(onx, save_intermediate=path)
            x = np.array([[0, 1], [2, 3]], dtype=np.float32)
            a = np.array([1, 1], dtype=np.float32)
            b = np.array([11], dtype=np.float32)
            ref.run(None, {"X": x, "A": a, "B": b})
            self.assertTrue(os.path.exists(path))
            examples = load_model_tests(root, "reference_evaluator_test_save_ref")
            self.assertEqual(len(examples), 2)

            class NewRef(ReferenceEvaluator):
                n_inits = 0
                n_calls = 0

                def __init__(self, *args, **kwargs):
                    NewRef.n_inits += 1
                    super().__init__(*args, **kwargs)

                def run(self, *args, **kwargs):
                    NewRef.n_calls += 1
                    return ReferenceEvaluator.run(self, *args, **kwargs)

            new_cls = ReferenceEvaluatorBackend[NewRef]  # type: ignore[type-arg]
            self.assertEqual(new_cls.__name__, "ReferenceEvaluatorBackendNewRef")
            self.assertTrue(issubclass(new_cls.cls_inference, NewRef))

            backend = create_reference_backend(
                new_cls, path_to_test=root, kind="reference_evaluator_test_save_ref"
            )
            backend.exclude("cuda")
            tests = backend.tests()
            names = []
            for att in dir(tests):
                if att.startswith("test_"):
                    test = getattr(tests, att)
                    try:
                        test()
                        names.append(att)
                    except unittest.case.SkipTest:
                        continue
            self.assertEqual(names, ["test_node_0_MatMul_cpu", "test_node_1_Add_cpu"])
            self.assertEqual(NewRef.n_inits, 2)
            self.assertEqual(NewRef.n_calls, 2)

    @unittest.skipIf(InferenceSession is None, reason="onnxruntime not available")
    def test_reference_evaluator_onnxruntime_runtime_save(self):
        onx = self._linear_regression(True)
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "reference_evaluator_test_save_ort")
            ref = ReferenceEvaluator(onx, save_intermediate=path)
            x = np.array([[0, 1], [2, 3]], dtype=np.float32)
            a = np.array([1, 1], dtype=np.float32)
            b = np.array([11], dtype=np.float32)
            ref.run(None, {"X": x, "A": a, "B": b})
            self.assertTrue(os.path.exists(path))
            examples = load_model_tests(root, "reference_evaluator_test_save_ort")
            self.assertEqual(len(examples), 2)

            class NewRef(InferenceSession):
                n_inits = 0
                n_calls = 0

                def __init__(self, model, *args, providers=None, **kwargs):
                    NewRef.n_inits += 1
                    super().__init__(
                        model.SerializeToString(),
                        *args,
                        providers=providers or ["CPUExecutionProvider"],
                        **kwargs,
                    )

                def run(self, *args, **kwargs):
                    NewRef.n_calls += 1
                    return InferenceSession.run(self, *args, **kwargs)

                @property
                def input_names(self):
                    inputs = self.get_inputs()
                    return [i.name for i in inputs]

                @property
                def output_names(self):
                    outputs = self.get_outputs()
                    return [o.name for o in outputs]

            new_cls = ReferenceEvaluatorBackend[NewRef]  # type: ignore[type-arg]
            self.assertEqual(new_cls.__name__, "ReferenceEvaluatorBackendNewRef")
            self.assertTrue(issubclass(new_cls.cls_inference, NewRef))

            backend = create_reference_backend(
                new_cls, path_to_test=root, kind="reference_evaluator_test_save_ort"
            )
            backend.exclude("cuda")
            tests = backend.tests()
            names = []
            for att in dir(tests):
                if att.startswith("test_"):
                    test = getattr(tests, att)
                    try:
                        test()
                        names.append(att)
                    except unittest.case.SkipTest:
                        continue
            self.assertEqual(names, ["test_node_0_MatMul_cpu", "test_node_1_Add_cpu"])
            self.assertEqual(NewRef.n_inits, 2)
            self.assertEqual(NewRef.n_calls, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
