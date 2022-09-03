# SPDX-License-Identifier: Apache-2.0

import unittest
from contextlib import redirect_stdout
from io import StringIO
from textwrap import dedent

import numpy as np  # type: ignore
from numpy.testing import assert_almost_equal

from onnx import ModelProto, TensorProto, checker, parser
from onnx.checker import check_model
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_opsetid,
    make_tensor_value_info,
)
from onnx.numpy_helper import from_array

try:
    import onnx.runtime as rt
except ImportError:
    # will disappear in the final version of the PR.
    import os
    import sys

    sys.path.insert(
        0,
        os.path.normpath(
            os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "..", "onnx")
        ),
    )
    import runtime as rt


class TestRuntimeInference(unittest.TestCase):
    m2_def = """
        <
            ir_version: 7,
            opset_import: [ "": 10, "com.microsoft": 1]
        >
        agraph (float[N, M] B01, float[N, M] B11, float[N, M] B21) => (float[N, M] D0)
        {
            C0 = Add(B01, B11)
            C1 = Sub(B11, B21)
            D0 = Mul(C0, C1)
        }
        """

    @staticmethod
    def _load_model(m_def: str) -> ModelProto:
        """
        Parses a model from a string representation, including checking
        the model for correctness
        """
        m = parser.parse_model(m_def)
        checker.check_model(m)
        return m

    @staticmethod
    def _linear_regression(clip=False, opset=None, min_value=-1.0, max_value=1.0):  # noqa
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        A = make_tensor_value_info("A", TensorProto.FLOAT, [None, None])
        B = make_tensor_value_info("B", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        node1 = make_node("MatMul", ["X", "A"], ["XA"])
        if clip:
            node2 = make_node("Add", ["XA", "B"], ["Y_clip"])
            if opset is not None and opset < 11:
                if min_value:
                    if max_value:
                        node3 = make_node(
                            "Clip", ["Y_clip"], ["Y"], min=min_value, max=max_value
                        )
                    else:
                        node3 = make_node("Clip", ["Y_clip"], ["Y"], min=min_value)
                elif max_value:
                    node3 = make_node("Clip", ["Y_clip"], ["Y"], max=max_value)
                else:
                    node3 = make_node("Clip", ["Y_clip"], ["Y"])
                graph = make_graph([node1, node2, node3], "lr", [X, A, B], [Y])
            else:
                mi = (
                    from_array(np.array([min_value], dtype=np.float32), name="mi")
                    if min_value
                    else None
                )
                ma = (
                    from_array(np.array([max_value], dtype=np.float32), name="ma")
                    if max_value
                    else None
                )
                inputs = ["Y_clip", "mi" if mi else "", "ma" if ma else ""]
                node3 = make_node("Clip", inputs, ["Y"])
                initializer = [_ for _ in [mi, ma] if _]
                graph = make_graph(
                    [node1, node2, node3], "lr", [X, A, B], [Y], initializer=initializer
                )
            f = lambda x, a, b: np.clip(a @ a + b, min_value, max_value)  # noqa
        else:
            node2 = make_node("Add", ["XA", "B"], ["Y"])
            graph = make_graph([node1, node2], "lr", [X, A, B], [Y])
            f = lambda x, a, b: a @ a + b  # noqa
        if opset is None:
            onnx_model = make_model(graph)
        else:
            onnx_model = make_model(graph, opset_imports=[make_opsetid("", opset)])
        try:
            check_model(onnx_model)
        except Exception as e:
            raise AssertionError(f"checker fails for\n{str(onnx_model)}") from e
        return onnx_model, f

    def test_inference_exceptions(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        with self.assertRaises(TypeError):
            rt.Inference(X)

    def test_inference_no_attribute(self):
        m = TestRuntimeInference._load_model(TestRuntimeInference.m2_def)
        checker.check_model(m)
        sess = rt.Inference(m)
        self.assertEqual(sess.input_names, ["B01", "B11", "B21"])
        self.assertEqual(sess.output_names, ["D0"])
        self.assertEqual(sess.opsets, {"": 10, "com.microsoft": 1})
        x = np.array([[0, 1], [2, 3]], dtype=np.float32)
        y = np.array([[4, 5], [6, 7]], dtype=np.float32)
        z = np.array([[-4, -5], [-6, -7]], dtype=np.float32)
        res = sess.run(None, {"B01": x, "B11": y, "B21": z})[0]
        expected = (x + y) * (y - z)
        assert_almost_equal(expected, res)

    def test_inference_no_attribute_verbose(self):
        m = TestRuntimeInference._load_model(TestRuntimeInference.m2_def)
        x = np.array([[0, 1], [2, 3]], dtype=np.float32)
        y = np.array([[4, 5], [6, 7]], dtype=np.float32)
        z = np.array([[-4, -5], [-6, -7]], dtype=np.float32)

        with self.subTest(level=2):
            sess = rt.Inference(m, verbose=2)
            stdout = StringIO()
            with redirect_stdout(stdout):
                sess.run(None, {"B01": x, "B11": y, "B21": z})
            out = stdout.getvalue()
            log = "Add(B01, B11) -> C0\nSub(B11, B21) -> C1\nMul(C0, C1) -> D0\n"
            self.assertEqual(log, out)

        with self.subTest(level=3):
            sess = rt.Inference(m, verbose=3)
            stdout = StringIO()
            with redirect_stdout(stdout):
                sess.run(None, {"B01": x, "B11": y, "B21": z})
            out = stdout.getvalue()
            log = dedent(
                """
                Add(B01, B11) -> C0
                 + C0: float32:(2, 2) in [4.0, 10.0]
                Sub(B11, B21) -> C1
                 + C1: float32:(2, 2) in [8.0, 14.0]
                Mul(C0, C1) -> D0
                 + D0: float32:(2, 2) in [32.0, 140.0]
                """
            ).lstrip("\n")
            self.assertEqual(log, out)

        with self.subTest(level=4):
            sess = rt.Inference(m, verbose=4)
            stdout = StringIO()
            with redirect_stdout(stdout):
                sess.run(None, {"B01": x, "B11": y, "B21": z})
            out = stdout.getvalue()
            log = dedent(
                """
                Add(B01, B11) -> C0
                 + C0: float32:(2, 2):[4.0, 6.0, 8.0, 10.0]
                Sub(B11, B21) -> C1
                 + C1: float32:(2, 2):[8.0, 10.0, 12.0, 14.0]
                Mul(C0, C1) -> D0
                 + D0: float32:(2, 2):[32.0, 60.0, 96.0, 140.0]
                """
            ).lstrip("\n")
            self.assertEqual(log, out)

        with self.subTest(level=15):
            sess = rt.Inference(m, verbose=15)
            stdout = StringIO()
            with redirect_stdout(stdout):
                sess.run(None, {"B01": x, "B11": y, "B21": z})
            out = stdout.getvalue()
            log = dedent(
                """
                Add(B01, B11) -> C0
                -- begin Add.run(2 inputs)
                -- done Add.run -> 1 outputs
                 + C0: float32:(2, 2):[4.0, 6.0, 8.0, 10.0]
                Sub(B11, B21) -> C1
                -- begin Sub.run(2 inputs)
                -- done Sub.run -> 1 outputs
                 + C1: float32:(2, 2):[8.0, 10.0, 12.0, 14.0]
                Mul(C0, C1) -> D0
                -- begin Mul.run(2 inputs)
                -- done Mul.run -> 1 outputs
                 + D0: float32:(2, 2):[32.0, 60.0, 96.0, 140.0]
                """
            ).lstrip("\n")
            self.assertEqual(log, out)

    def test_inference_lr(self):
        lr, f = TestRuntimeInference._linear_regression()
        x = np.array([[0, 1], [2, 3]], dtype=np.float32)
        a = np.array([1, 1], dtype=np.float32)
        b = np.array([11], dtype=np.float32)
        expected = f(x, a, b)
        sess = rt.Inference(lr)
        got = sess.run(None, {"X": a, "A": a, "B": b})[0]
        assert_almost_equal(expected, got)

    def test_inference_lr_clip(self):
        with self.subTest(opt="min+max"):
            lr, f = TestRuntimeInference._linear_regression(clip=True)
            x = np.array([[0, 1], [2, 3]], dtype=np.float32)
            a = np.array([1, 1], dtype=np.float32)
            b = np.array([11], dtype=np.float32)
            expected = f(x, a, b)
            sess = rt.Inference(lr)
            last_node = sess.rt_nodes_[-1]
            self.assertEqual(last_node.__class__.__name__, "Clip_11")
            got = sess.run(None, {"X": a, "A": a, "B": b})[0]
            assert_almost_equal(expected, got)

        with self.subTest(opt="max"):
            lr, f = TestRuntimeInference._linear_regression(clip=True, min_value=None)
            x = np.array([[0, 1], [2, 3]], dtype=np.float32)
            a = np.array([1, 1], dtype=np.float32)
            b = np.array([11], dtype=np.float32)
            expected = f(x, a, b)
            sess = rt.Inference(lr)
            last_node = sess.rt_nodes_[-1]
            self.assertEqual(last_node.__class__.__name__, "Clip_11")
            got = sess.run(None, {"X": a, "A": a, "B": b})[0]
            assert_almost_equal(expected, got)

        with self.subTest(opt="min"):
            lr, f = TestRuntimeInference._linear_regression(clip=True, max_value=None)
            x = np.array([[0, 1], [2, 3]], dtype=np.float32)
            a = np.array([1, 1], dtype=np.float32)
            b = np.array([11], dtype=np.float32)
            expected = f(x, a, b)
            sess = rt.Inference(lr)
            last_node = sess.rt_nodes_[-1]
            self.assertEqual(last_node.__class__.__name__, "Clip_11")
            got = sess.run(None, {"X": a, "A": a, "B": b})[0]
            assert_almost_equal(expected, got)

    def test_inference_lr_clip_6(self):
        with self.subTest(opt="min+max"):
            lr, f = TestRuntimeInference._linear_regression(clip=True, opset=10)
            x = np.array([[0, 1], [2, 3]], dtype=np.float32)
            a = np.array([1, 1], dtype=np.float32)
            b = np.array([11], dtype=np.float32)
            expected = f(x, a, b)
            sess = rt.Inference(lr)
            last_node = sess.rt_nodes_[-1]
            self.assertEqual(last_node.__class__.__name__, "Clip_6")
            self.assertEqual(last_node.min, -1)
            self.assertEqual(last_node.max, 1)
            got = sess.run(None, {"X": a, "A": a, "B": b})[0]
            assert_almost_equal(expected, got)

        with self.subTest(opt="max"):
            lr, f = TestRuntimeInference._linear_regression(
                clip=True, opset=10, min_value=None
            )
            x = np.array([[0, 1], [2, 3]], dtype=np.float32)
            a = np.array([1, 1], dtype=np.float32)
            b = np.array([11], dtype=np.float32)
            expected = f(x, a, b)
            sess = rt.Inference(lr)
            last_node = sess.rt_nodes_[-1]
            self.assertEqual(last_node.__class__.__name__, "Clip_6")
            self.assertEqual(last_node.max, 1)
            self.assertFalse("min" in last_node.__dict__)
            got = sess.run(None, {"X": a, "A": a, "B": b})[0]
            assert_almost_equal(expected, got)

        with self.subTest(opt="max"):
            lr, f = TestRuntimeInference._linear_regression(
                clip=True, opset=10, max_value=None
            )
            x = np.array([[0, 1], [2, 3]], dtype=np.float32)
            a = np.array([1, 1], dtype=np.float32)
            b = np.array([11], dtype=np.float32)
            expected = f(x, a, b)
            sess = rt.Inference(lr)
            last_node = sess.rt_nodes_[-1]
            self.assertEqual(last_node.__class__.__name__, "Clip_6")
            self.assertEqual(last_node.min, -1)
            self.assertFalse("max" in last_node.__dict__)
            got = sess.run(None, {"X": a, "A": a, "B": b})[0]
            assert_almost_equal(expected, got)

    def test_nested_local_functions(self):
        m = parser.parse_model(
            """
            <
              ir_version: 8,
              opset_import: [ "" : 14, "local" : 1],
              producer_name: "test",
              producer_version: "1.0",
              model_version: 1,
              doc_string: "Test preprocessing model"
            >
            agraph (uint8[H, W, C] x) => (uint8[H, W, C] x_processed)
            {
                x_processed = local.func(x)
            }

            <
              opset_import: [ "" : 14 ],
              domain: "local",
              doc_string: "function 1"
            >
            f1 (x) => (y) {
                y = Identity(x)
            }

            <
              opset_import: [ "" : 14 ],
              domain: "local",
              doc_string: "function 2"
            >
            f2 (x) => (y) {
                y = Identity(x)
            }

            <
              opset_import: [ "" : 14, "local" : 1 ],
              domain: "local",
              doc_string: "Preprocessing function."
            >
            func (x) => (y) {
                x1 = local.f1(x)
                y = local.f2(x1)
            }
        """
        )

        sess = rt.Inference(m)
        x = np.array([0, 1, 3], dtype=np.uint8).reshape((1, 1, 3))
        result = sess.run(None, {"x": x})[0]
        expected = x
        assert_almost_equal(expected, result)


if __name__ == "__main__":
    # TestRuntimeInference().test_inference_lr_clip_11()
    unittest.main()
