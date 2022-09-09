# SPDX-License-Identifier: Apache-2.0
# type: ignore
# pylint: disable=C3001,R0914,W0221

import unittest
from contextlib import redirect_stdout
from io import StringIO
from textwrap import dedent
from typing import Any, List

import numpy as np  # type: ignore
from numpy.testing import assert_almost_equal  # type: ignore

import onnx.runtime as rt
from onnx import AttributeProto, FunctionProto, ModelProto, TensorProto, checker, parser
from onnx.checker import check_model
from onnx.defs import onnx_opset_version
from onnx.helper import (
    make_function,
    make_graph,
    make_model,
    make_node,
    make_opsetid,
    make_sequence_type_proto,
    make_tensor,
    make_tensor_sequence_value_info,
    make_tensor_value_info,
    make_value_info,
)
from onnx.numpy_helper import from_array
from onnx.runtime.op_run import OpRun


def make_sequence_value_info(name, elem_type, shape):
    if isinstance(elem_type, int):
        return make_tensor_sequence_value_info(name, elem_type, shape)
    s_type = make_sequence_type_proto(elem_type)
    return make_value_info(name, s_type, shape)


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
    def _linear_regression(clip=False, opset=None, min_value=-1.0, max_value=1.0):
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
            f = lambda x, a, b: np.clip(a @ a + b, min_value, max_value)  # noqa: E731
        else:
            node2 = make_node("Add", ["XA", "B"], ["Y"])
            graph = make_graph([node1, node2], "lr", [X, A, B], [Y])
            f = lambda x, a, b: a @ a + b  # noqa: E731
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

    def test_inference_no_attribute_bytes(self):
        m = TestRuntimeInference._load_model(TestRuntimeInference.m2_def)
        checker.check_model(m)
        sess = rt.Inference(m.SerializeToString())
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
            self.assertEqual(last_node.min, -3.4028234663852886e38)
            got = sess.run(None, {"X": a, "A": a, "B": b})[0]
            assert_almost_equal(expected, got)

        with self.subTest(opt="min"):
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
            self.assertEqual(last_node.max, 3.4028234663852886e38)
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

    def test_reduce_sum_11(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        node1 = make_node("ReduceSum", ["X"], ["Y"], axes=[1], keepdims=1)
        graph = make_graph([node1], "rs", [X], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 11)])
        check_model(onnx_model)
        x = np.arange(60).reshape((3, 4, 5)).astype(np.float32)
        expected = x.sum(axis=1, keepdims=1)
        sess = rt.Inference(onnx_model)
        got = sess.run(None, {"X": x})[0]
        assert_almost_equal(expected, got)

    def test_reduce_sum_13(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        A = make_tensor_value_info("A", TensorProto.INT64, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        node1 = make_node("ReduceSum", ["X", "A"], ["Y"], keepdims=1)
        graph = make_graph([node1], "rs", [X, A], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 13)])
        check_model(onnx_model)
        x = np.arange(60).reshape((3, 4, 5)).astype(np.float32)
        a = np.array([1], dtype=np.int64)
        expected = x.sum(axis=1, keepdims=1)
        sess = rt.Inference(onnx_model)
        got = sess.run(None, {"X": x, "A": a})[0]
        assert_almost_equal(expected, got)

    def test_reduce_sum_13_empty_axes(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        A = make_tensor_value_info("A", TensorProto.INT64, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        node1 = make_node("ReduceSum", ["X", "A"], ["Y"], keepdims=1)
        graph = make_graph([node1], "rs", [X, A], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 13)])
        check_model(onnx_model)
        x = np.arange(60).reshape((3, 4, 5)).astype(np.float32)
        a = np.array([], dtype=np.int64)
        expected = x.sum(keepdims=1)
        sess = rt.Inference(onnx_model)
        got = sess.run(None, {"X": x, "A": a})[0]
        assert_almost_equal(expected, got)

    def test_reduce_sum_13_empty_axes_noop(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        node1 = make_node("ReduceSum", ["X"], ["Y"], keepdims=1, noop_with_empty_axes=1)
        graph = make_graph([node1], "rs", [X], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 13)])
        check_model(onnx_model)
        x = np.arange(60).reshape((3, 4, 5)).astype(np.float32)
        sess = rt.Inference(onnx_model)
        got = sess.run(None, {"X": x})[0]
        assert_almost_equal(x, got)

    def test_reduce_greater(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        Z = make_tensor_value_info("Z", TensorProto.FLOAT, [None])
        node1 = make_node("Greater", ["X", "Y"], ["Z"])
        graph = make_graph([node1], "g", [X, Y], [Z])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 13)])
        check_model(onnx_model)
        x = np.arange(4).reshape((2, 2)).astype(np.float32)
        y = np.array([2], dtype=np.float32)
        expected = x > y
        sess = rt.Inference(onnx_model)
        got = sess.run(None, {"X": x, "Y": y})[0]
        assert_almost_equal(expected, got)

    def test_reduce_greater_or_equal(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        Z = make_tensor_value_info("Z", TensorProto.FLOAT, [None])
        node1 = make_node("GreaterOrEqual", ["X", "Y"], ["Z"])
        graph = make_graph([node1], "g", [X, Y], [Z])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 13)])
        check_model(onnx_model)
        x = np.arange(4).reshape((2, 2)).astype(np.float32)
        y = np.array([2], dtype=np.float32)
        expected = x >= y
        sess = rt.Inference(onnx_model)
        got = sess.run(None, {"X": x, "Y": y})[0]
        assert_almost_equal(expected, got)

    def test_if(self):
        C = make_tensor_value_info("C", TensorProto.FLOAT, [None])
        bthen = make_node(
            "Constant",
            [],
            ["C"],
            value_floats=from_array(np.array([1], dtype=np.float32)),
        )
        bthen_body = make_graph([bthen], "gthen", [], [C])

        C = make_tensor_value_info("C", TensorProto.FLOAT, [None])
        belse = make_node(
            "Constant",
            [],
            ["C"],
            value_floats=from_array(np.array([0], dtype=np.float32)),
        )
        belse_body = make_graph([belse], "gelse", [], [C])

        zero = from_array(np.array([0], dtype=np.float32), name="zero")
        greater = make_node("Greater", ["X", "zero"], ["G"])
        node_if = make_node(
            "If",
            ["G"],
            ["Z"],
            then_branch=bthen_body,
            else_branch=belse_body,
        )
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Z = make_tensor_value_info("Z", TensorProto.FLOAT, [None])
        graph = make_graph([greater, node_if], "g", [X], [Z], initializer=[zero])
        model_def = make_model(graph)

        sess = rt.Inference(model_def)
        self.assertEqual(str(sess), "Inference(X) -> Z")

        x = np.array([1, 2], dtype=np.float32)
        got = sess.run(None, {"X": x})[0]
        assert_almost_equal(np.array([1], dtype=np.float32), got)

        x = np.array([-1, -2], dtype=np.float32)
        got = sess.run(None, {"X": x})[0]
        assert_almost_equal(np.array([0], dtype=np.float32), got)

    def test_if_function(self):
        then_out = make_tensor_value_info("then_out", TensorProto.FLOAT, [5])
        else_out = make_tensor_value_info("else_out", TensorProto.FLOAT, [5])

        x = np.array([1, 2, 3, 4, 5]).astype(np.float32)
        y = np.array([5, 4, 3, 2, 1]).astype(np.float32)

        then_const_node = make_node(
            "Constant", inputs=[], outputs=["then_out"], value=from_array(x)
        )
        else_const_node = make_node(
            "Constant", inputs=[], outputs=["else_out"], value=from_array(y)
        )
        then_body = make_graph([then_const_node], "then_body", [], [then_out])
        else_body = make_graph([else_const_node], "else_body", [], [else_out])
        if_node = make_node(
            "If",
            inputs=["f_cond"],
            outputs=["f_res"],
            then_branch=then_body,
            else_branch=else_body,
        )

        f = FunctionProto()
        f.domain = "custom"
        f.name = "fn"
        f.input.extend(["f_cond"])
        f.output.extend(["f_res"])
        f.node.extend([if_node])
        opset = onnx_opset_version()
        f.opset_import.extend([make_opsetid("", opset)])

        graph = make_graph(
            nodes=[make_node("fn", domain="custom", inputs=["cond"], outputs=["res"])],
            name="graph",
            inputs=[make_tensor_value_info("cond", TensorProto.BOOL, [])],
            outputs=[make_tensor_value_info("res", TensorProto.FLOAT, [5])],
        )

        m = make_model(
            graph,
            producer_name="test",
            opset_imports=[make_opsetid("", opset), make_opsetid("custom", 1)],
        )
        m.functions.extend([f])

        sess = rt.Inference(m)
        result = sess.run(None, {"cond": np.array(True)})
        expected = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        assert_almost_equal(expected, result[0])

    def test_function_attribute(self):
        opset = onnx_opset_version()
        new_domain = "custom"
        opset_imports = [make_opsetid("", opset), make_opsetid(new_domain, 1)]
        cst = make_node("Constant", [], ["B"])

        att = AttributeProto()
        att.name = "value"
        att.ref_attr_name = "bias"
        att.type = AttributeProto.TENSOR
        cst.attribute.append(att)

        node1 = make_node("MatMul", ["X", "A"], ["XA"])
        node2 = make_node("Add", ["XA", "B"], ["Y"])

        linear_regression = make_function(
            new_domain,
            "LinearRegression",
            ["X", "A"],
            ["Y"],
            [cst, node1, node2],
            opset_imports,
            ["bias"],
        )

        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        A = make_tensor_value_info("A", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])

        graph = make_graph(
            [
                make_node(
                    "LinearRegression",
                    ["X", "A"],
                    ["Y1"],
                    domain=new_domain,
                    bias=make_tensor("former_B", TensorProto.FLOAT, [1], [0.67]),
                ),
                make_node("Abs", ["Y1"], ["Y"]),
            ],
            "example",
            [X, A],
            [Y],
        )

        onnx_model = make_model(
            graph, opset_imports=opset_imports, functions=[linear_regression]
        )
        sess = rt.Inference(onnx_model)
        x = np.arange(6).reshape((3, 2)).astype(np.float32)
        a = np.array([1, -1], dtype=np.float32)
        result = sess.run(None, {"X": x, "A": a})[0]
        expected = np.abs(x @ a + 0.67)
        assert_almost_equal(expected, result)

    def test_custom_node(self):
        class _InvAlpha:

            op_domain = "custom"

            def __init__(self, onnx_node, run_params):  # type: ignore
                self.onnx_node = onnx_node
                self.run_params = run_params

            def _run(self, x):  # type: ignore
                return (1 / (x + self.alpha),)

        class InvAlpha_(OpRun):
            def __init__(self, onnx_node, run_params):  # type: ignore
                OpRun.__init__(self, onnx_node, run_params)

            def _run(self, x):  # type: ignore
                return (1 / (x + self.alpha),)

        class InvAlpha(OpRun):

            op_domain = "custom"

            def __init__(self, onnx_node, run_params):  # type: ignore
                OpRun.__init__(self, onnx_node, run_params)

            def _run(self, x):  # type: ignore
                return (1 / (x + self.alpha),)

        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        node1 = make_node("InvAlpha", ["X"], ["Y"], alpha=0.5, domain="custom")
        graph = make_graph([node1], "rs", [X], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("custom", 1)])
        x = np.arange(60).reshape((3, 4, 5)).astype(np.float32) + 1
        with self.assertRaises(NotImplementedError):
            rt.Inference(onnx_model)
        with self.assertRaises(TypeError):
            rt.Inference(onnx_model, new_ops=[_InvAlpha])
        with self.assertRaises(AttributeError):
            rt.Inference(onnx_model, new_ops=[InvAlpha_])
        with self.assertRaises(ValueError):
            rt.Inference(onnx_model, new_ops=[InvAlpha, InvAlpha])
        sess = rt.Inference(onnx_model, new_ops=[InvAlpha])
        got = sess.run(None, {"X": x})[0]
        expected = 1 / (x + 0.5)
        assert_almost_equal(expected, got)

    def test_loop(self):
        # Given a tensor x of values [x1, ..., xN],
        # Return a sequence of tensors of
        #   [[x1], [x1, x2], ..., [x1, ..., xN]]

        cond_in = make_tensor_value_info("cond_in", TensorProto.BOOL, [])
        cond_out = make_tensor_value_info("cond_out", TensorProto.BOOL, [])
        iter_count = make_tensor_value_info("iter_count", TensorProto.INT64, [])
        seq_in = make_tensor_sequence_value_info("seq_in", TensorProto.FLOAT, None)
        seq_out = make_tensor_sequence_value_info("seq_out", TensorProto.FLOAT, None)

        x = np.array([1, 2, 3, 4, 5]).astype(np.float32)

        x_const_node = make_node(
            "Constant",
            inputs=[],
            outputs=["x"],
            value=make_tensor(
                name="const_tensor_x",
                data_type=TensorProto.FLOAT,
                dims=x.shape,
                vals=x.flatten().astype(float),
            ),
        )

        one_const_node = make_node(
            "Constant",
            inputs=[],
            outputs=["one"],
            value=make_tensor(
                name="const_tensor_one",
                data_type=TensorProto.INT64,
                dims=(),
                vals=[1],
            ),
        )

        zero_const_node = make_node(
            "Constant",
            inputs=[],
            outputs=["slice_start"],
            value=make_tensor(
                name="const_tensor_zero",
                data_type=TensorProto.INT64,
                dims=(1,),
                vals=[0],
            ),
        )

        axes_node = make_node(
            "Constant",
            inputs=[],
            outputs=["axes"],
            value=make_tensor(
                name="const_tensor_axes",
                data_type=TensorProto.INT64,
                dims=(),
                vals=[0],
            ),
        )

        add_node = make_node("Add", inputs=["iter_count", "one"], outputs=["end"])

        end_unsqueeze_node = make_node(
            "Unsqueeze", inputs=["end", "axes"], outputs=["slice_end"]
        )

        slice_node = make_node(
            "Slice", inputs=["x", "slice_start", "slice_end"], outputs=["slice_out"]
        )

        insert_node = make_node(
            "SequenceInsert", inputs=["seq_in", "slice_out"], outputs=["seq_out"]
        )

        identity_node = make_node("Identity", inputs=["cond_in"], outputs=["cond_out"])

        loop_body = make_graph(
            [
                identity_node,
                x_const_node,
                one_const_node,
                zero_const_node,
                add_node,
                axes_node,
                end_unsqueeze_node,
                slice_node,
                insert_node,
            ],
            "loop_body",
            [iter_count, cond_in, seq_in],
            [cond_out, seq_out],
        )

        node = make_node(
            "Loop",
            inputs=["trip_count", "cond", "seq_empty"],
            outputs=["seq_res"],
            body=loop_body,
        )
        node_concat = make_node(
            "ConcatFromSequence",
            inputs=["seq_res"],
            outputs=["res"],
            axis=0,
            new_axis=0,
        )

        trip_count = np.array(5).astype(np.int64)
        seq_empty = []  # type: List[Any]
        # seq_res = [x[:int(i)] for i in x]
        cond = np.array(1).astype(np.bool_)

        model_def = make_model(
            graph=make_graph(
                name="loop_test",
                inputs=[
                    make_tensor_value_info(
                        "trip_count", TensorProto.INT64, trip_count.shape
                    ),
                    make_tensor_value_info("cond", TensorProto.BOOL, cond.shape),
                    make_sequence_value_info("seq_empty", TensorProto.FLOAT, []),
                ],
                outputs=[make_tensor_value_info("res", TensorProto.FLOAT, None)],
                nodes=[node, node_concat],
            )
        )

        expected = np.array(
            [1.0, 1.0, 2.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            dtype=np.float32,
        )
        oinf = rt.Inference(model_def)
        inputs = {"trip_count": trip_count, "cond": cond, "seq_empty": seq_empty}
        got = oinf.run(None, inputs)
        assert_almost_equal(expected, got[0])

    def test_onnxt_runtime_bernoulli(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        node1 = make_node("Bernoulli", ["X"], ["Y"], seed=0.0)
        graph = make_graph([node1], "g", [X], [Y])
        onnx_model = make_model(graph)
        check_model(onnx_model)
        sess = rt.Inference(onnx_model)
        got = sess.run(None, {"X": np.zeros((2, 4), dtype=np.float32) + 0.5})[0]
        self.assertEqual(got.shape, (2, 4))
        self.assertEqual(got.dtype, np.float32)
        self.assertGreater(got.min(), -1e-5)
        self.assertLess(got.max(), 1 + 1e-5)

    def test_onnxt_runtime_random_uniform(self):
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        node1 = make_node("RandomUniform", [], ["Y"], seed=0.0, shape=[2, 4])
        graph = make_graph([node1], "g", [], [Y])
        onnx_model = make_model(graph)
        check_model(onnx_model)
        sess = rt.Inference(onnx_model)
        got = sess.run(None, {})[0]
        self.assertEqual(got.shape, (2, 4))
        self.assertEqual(got.dtype, np.float32)
        self.assertGreater(got.min(), 0)
        self.assertLess(got.max(), 1)

    def test_onnxt_runtime_random_uniform_like(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        node1 = make_node("RandomUniformLike", ["X"], ["Y"], seed=0.0)
        graph = make_graph([node1], "g", [X], [Y])
        onnx_model = make_model(graph)
        check_model(onnx_model)
        sess = rt.Inference(onnx_model)
        got = sess.run(None, {"X": np.zeros((2, 4), dtype=np.float32)})[0]
        self.assertEqual(got.shape, (2, 4))
        self.assertEqual(got.dtype, np.float32)
        self.assertGreater(got.min(), 0)
        self.assertLess(got.max(), 1)

    def test_onnxt_runtime_random_normal(self):
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        node1 = make_node("RandomNormal", [], ["Y"], seed=0.0, shape=[2, 4])
        graph = make_graph([node1], "g", [], [Y])
        onnx_model = make_model(graph)
        check_model(onnx_model)
        sess = rt.Inference(onnx_model)
        got = sess.run(None, {})[0]
        self.assertEqual(got.shape, (2, 4))
        self.assertEqual(got.dtype, np.float32)

    def test_onnxt_runtime_random_normal_like(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        node1 = make_node("RandomNormalLike", ["X"], ["Y"], seed=0.0)
        graph = make_graph([node1], "g", [X], [Y])
        onnx_model = make_model(graph)
        check_model(onnx_model)
        sess = rt.Inference(onnx_model)
        got = sess.run(None, {"X": np.zeros((2, 4), dtype=np.float32)})[0]
        self.assertEqual(got.shape, (2, 4))
        self.assertEqual(got.dtype, np.float32)


if __name__ == "__main__":
    # TestRuntimeInference().test_custom_node()
    unittest.main(verbosity=2)
