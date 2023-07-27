# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import unittest

import numpy as np

from onnx import ModelProto, TensorProto, helper, numpy_helper
from onnx.backend.test.loader import load_model_tests
from onnx.checker import check_model
from onnx.defs import onnx_opset_version
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
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [None, None])
        B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [None, None])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        node1 = helper.make_node("MatMul", ["X", "A"], ["XA"])
        node2 = helper.make_node("Add", ["XA", "B"], ["Y"])
        graph = helper.make_graph([node1, node2], "lr", [X, A, B], [Y])
        onnx_model = helper.make_model(
            graph,
            opset_imports=[
                helper.make_opsetid(
                    "", onnx_opset_version() - (1 if previous_opset else 0)
                )
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

    def _get_loop_model(self) -> ModelProto:
        return helper.make_model(
            opset_imports=[helper.make_operatorsetid("", 18)],
            ir_version=8,
            graph=helper.make_graph(
                name="test-loop",
                inputs=[
                    helper.make_tensor_value_info(
                        "input_0", TensorProto.INT32, shape=[1]
                    ),
                    helper.make_tensor_value_info(
                        "while_maximum_iterations_0", TensorProto.INT64, shape=[]
                    ),
                    helper.make_tensor_value_info(
                        "const_fold_opt__18", TensorProto.INT64, shape=[1]
                    ),
                    helper.make_tensor_value_info(
                        "const_fold_opt__17", TensorProto.FLOAT, shape=[]
                    ),
                    helper.make_tensor_value_info(
                        "Const_0", TensorProto.INT32, shape=[1]
                    ),
                ],
                outputs=[
                    helper.make_tensor_value_info(
                        "output_0", TensorProto.INT32, shape=[1]
                    )
                ],
                initializer=[
                    numpy_helper.from_array(
                        np.array(9223372036854775807, dtype=np.int64),
                        name="while_maximum_iterations_0",
                    ),
                    numpy_helper.from_array(
                        np.array([-1], dtype=np.int64), name="const_fold_opt__18"
                    ),
                    numpy_helper.from_array(
                        np.array(10, dtype=np.int32), name="const_fold_opt__17"
                    ),
                    numpy_helper.from_array(
                        np.array([1], dtype=np.int32), name="Const_0"
                    ),
                    numpy_helper.from_array(np.array([0], dtype=np.int32), name="zero"),
                ],
                nodes=[
                    helper.make_node(
                        "Cast",
                        inputs=["input_0"],
                        outputs=["while_cond_158_while_Less__13_0"],
                        name="while_cond_158_while_Less__13",
                        domain="",
                        to=TensorProto.INT32,
                    ),
                    helper.make_node(
                        "Less",
                        inputs=[
                            "while_cond_158_while_Less__13_0",
                            "const_fold_opt__17",
                        ],
                        outputs=["while_cond_158_while_Less_0"],
                        name="while_cond_158_while_Less",
                        domain="",
                    ),
                    helper.make_node(
                        "Squeeze",
                        inputs=["while_cond_158_while_Less_0"],
                        outputs=["while_cond_158_while_Squeeze_0"],
                        name="while_cond_158_while_Squeeze",
                        domain="",
                    ),
                    helper.make_node(
                        "Loop",
                        inputs=[
                            "while_maximum_iterations_0",
                            "while_cond_158_while_Squeeze_0",
                            "input_0",
                            "Const_0",
                        ],
                        outputs=["while_loop_0", "while_loop_1"],
                        name="while_loop",
                        body=helper.make_graph(
                            name="while_body",
                            inputs=[
                                helper.make_tensor_value_info(
                                    "while_while_loop_counter_0",
                                    TensorProto.INT64,
                                    shape=[],
                                ),
                                helper.make_tensor_value_info(
                                    "cond__15_0", TensorProto.BOOL, shape=[]
                                ),
                                helper.make_tensor_value_info(
                                    "while_placeholder_0", TensorProto.INT32, shape=[1]
                                ),
                                helper.make_tensor_value_info(
                                    "while_add_const_0_0", TensorProto.INT32, shape=[1]
                                ),
                            ],
                            outputs=[
                                helper.make_tensor_value_info(
                                    "cond___while_Identity_graph_outputs_Identity__3_0",
                                    TensorProto.BOOL,
                                    shape=[],
                                ),
                                helper.make_tensor_value_info(
                                    "while_Identity_2_0", TensorProto.INT32, shape=[1]
                                ),
                                helper.make_tensor_value_info(
                                    "while_add_const_0_0", TensorProto.INT32, shape=[1]
                                ),
                            ],
                            # Cannot use the same name as both a subgraph initializer and subgraph input: while_while_loop_counter_0
                            initializer=[
                                numpy_helper.from_array(
                                    np.array(10, dtype=np.int64),
                                    name="while_while_loop_counter_0",
                                )
                            ],
                            nodes=[
                                helper.make_node(
                                    "Add",
                                    inputs=[
                                        "while_placeholder_0",
                                        "while_add_const_0_0",
                                    ],
                                    outputs=["while_Identity_2_0"],
                                    name="while_Add",
                                ),
                                helper.make_node(
                                    "CastLike",
                                    inputs=["while_Identity_2_0", "zero"],
                                    outputs=["cond___while_Less__13_0"],
                                    name="cond___while_Less__13",
                                    domain="",
                                ),
                                helper.make_node(
                                    "Less",
                                    inputs=[
                                        "cond___while_Less__13_0",
                                        "while_while_loop_counter_0",
                                    ],
                                    outputs=["cond___while_Less_0"],
                                    name="cond___while_Less",
                                    domain="",
                                ),
                                helper.make_node(
                                    "Squeeze",
                                    inputs=["cond___while_Less_0"],
                                    outputs=[
                                        "cond___while_Identity_graph_outputs_Identity__3_0"
                                    ],
                                    name="cond___while_Squeeze",
                                    domain="",
                                ),
                            ],
                        ),
                    ),
                    helper.make_node(
                        "Unsqueeze",
                        inputs=["while_loop_0", "zero"],
                        outputs=["Reshape_tensor_0"],
                        name="Reshape_tensor",
                    ),
                    helper.make_node(
                        "Reshape",
                        inputs=["Reshape_tensor_0", "const_fold_opt__18"],
                        outputs=["output_0"],
                        name="Reshape",
                    ),
                ],
            ),
        )

    def test_reference_evaluator_loop_save(self):
        onx = self._get_loop_model()
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "test_reference_evaluator_loop_save")
            ref = ReferenceEvaluator(onx, save_intermediate=path)
            ref.run(
                None,
                {
                    "input_0": np.array([5], dtype=np.int32),
                    "while_maximum_iterations_0": np.array(10, dtype=np.int32),
                },
            )
            self.assertTrue(os.path.exists(path))
            examples = load_model_tests(root, "test_reference_evaluator_loop_save")
            self.assertEqual(len(examples), 6)

            backend = create_reference_backend(
                path_to_test=root, kind="test_reference_evaluator_loop_save"
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
            self.assertEqual(
                names,
                [
                    "test_node_0_Cast_cpu",
                    "test_node_1_Less_cpu",
                    "test_node_2_Squeeze_cpu",
                    "test_node_3_Loop_cpu",
                    "test_node_4_Unsqueeze_cpu",
                    "test_node_5_Reshape_cpu",
                ],
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
