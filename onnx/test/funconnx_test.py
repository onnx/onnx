# SPDX-License-Identifier: Apache-2.0
# type: ignore
# pylint: disable=C3001,C0302,C0415,R0904,R0913,R0914,R0915,W0221,W0707

import unittest
from contextlib import redirect_stdout
from functools import wraps
from io import StringIO
from textwrap import dedent
from typing import Any, List

import numpy as np  # type: ignore
from numpy.testing import assert_allclose  # type: ignore

from onnx import AttributeProto, FunctionProto, ModelProto, TensorProto, checker, parser
from onnx.checker import check_model
from onnx.defs import onnx_opset_version
from onnx.funconnx import ProtoRun
from onnx.funconnx.aionnx import load_op
from onnx.funconnx.aionnx._op_list import Celu
from onnx.funconnx.aionnx.op_celu import _vcelu1
from onnx.funconnx.aionnx.op_col2im import (
    _col2im_naive_implementation_2d,
    col2im_naive_implementation,
)
from onnx.funconnx.aionnx_preview_training._op_list import Adam
from onnx.funconnx.op_run import OpRun
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


def skip_if_no_onnxruntime(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            import onnxruntime  # pylint: disable=W0611
        except ImportError:
            raise unittest.SkipTest("onnxruntime not installed")
        fn(*args, **kwargs)

    return wrapper


def skip_if_no_torch(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            import torch  # pylint: disable=W0611
        except ImportError:
            raise unittest.SkipTest("torch not installed")
        fn(*args, **kwargs)

    return wrapper


def make_sequence_value_info(name, elem_type, shape):
    if isinstance(elem_type, int):
        return make_tensor_sequence_value_info(name, elem_type, shape)
    s_type = make_sequence_type_proto(elem_type)
    return make_value_info(name, s_type, shape)


class TestRuntimeProtoRun(unittest.TestCase):
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

    def test_ProtoRun_exceptions(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        with self.assertRaises(TypeError):
            ProtoRun(X)

    def test_ProtoRun_no_attribute(self):
        m = TestRuntimeProtoRun._load_model(TestRuntimeProtoRun.m2_def)
        checker.check_model(m)
        sess = ProtoRun(m)
        self.assertEqual(sess.input_names, ["B01", "B11", "B21"])
        self.assertEqual(sess.output_names, ["D0"])
        self.assertEqual(sess.opsets, {"": 10, "com.microsoft": 1})
        x = np.array([[0, 1], [2, 3]], dtype=np.float32)
        y = np.array([[4, 5], [6, 7]], dtype=np.float32)
        z = np.array([[-4, -5], [-6, -7]], dtype=np.float32)
        res = sess.run(None, {"B01": x, "B11": y, "B21": z})[0]
        expected = (x + y) * (y - z)
        assert_allclose(expected, res)

    def test_ProtoRun_no_attribute_bytes(self):
        m = TestRuntimeProtoRun._load_model(TestRuntimeProtoRun.m2_def)
        checker.check_model(m)
        sess = ProtoRun(m.SerializeToString())
        self.assertEqual(sess.input_names, ["B01", "B11", "B21"])
        self.assertEqual(sess.output_names, ["D0"])
        self.assertEqual(sess.opsets, {"": 10, "com.microsoft": 1})
        x = np.array([[0, 1], [2, 3]], dtype=np.float32)
        y = np.array([[4, 5], [6, 7]], dtype=np.float32)
        z = np.array([[-4, -5], [-6, -7]], dtype=np.float32)
        res = sess.run(None, {"B01": x, "B11": y, "B21": z})[0]
        expected = (x + y) * (y - z)
        assert_allclose(expected, res)

    def test_ProtoRun_no_attribute_verbose(self):
        m = TestRuntimeProtoRun._load_model(TestRuntimeProtoRun.m2_def)
        x = np.array([[0, 1], [2, 3]], dtype=np.float32)
        y = np.array([[4, 5], [6, 7]], dtype=np.float32)
        z = np.array([[-4, -5], [-6, -7]], dtype=np.float32)

        with self.subTest(level=2):
            sess = ProtoRun(m, verbose=2)
            stdout = StringIO()
            with redirect_stdout(stdout):
                sess.run(None, {"B01": x, "B11": y, "B21": z})
            out = stdout.getvalue()
            log = "Add(B01, B11) -> C0\nSub(B11, B21) -> C1\nMul(C0, C1) -> D0\n"
            self.assertEqual(log, out)

        with self.subTest(level=3):
            sess = ProtoRun(m, verbose=3)
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
            sess = ProtoRun(m, verbose=4)
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
            sess = ProtoRun(m, verbose=15)
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

    def test_ProtoRun_lr(self):
        lr, f = TestRuntimeProtoRun._linear_regression()
        x = np.array([[0, 1], [2, 3]], dtype=np.float32)
        a = np.array([1, 1], dtype=np.float32)
        b = np.array([11], dtype=np.float32)
        expected = f(x, a, b)
        sess = ProtoRun(lr)
        got = sess.run(None, {"X": a, "A": a, "B": b})[0]
        assert_allclose(expected, got)

    def test_ProtoRun_lr_clip(self):
        with self.subTest(opt="min+max"):
            lr, f = TestRuntimeProtoRun._linear_regression(clip=True)
            x = np.array([[0, 1], [2, 3]], dtype=np.float32)
            a = np.array([1, 1], dtype=np.float32)
            b = np.array([11], dtype=np.float32)
            expected = f(x, a, b)
            sess = ProtoRun(lr)
            last_node = sess.rt_nodes_[-1]
            self.assertEqual(last_node.__class__.__name__, "Clip_11")
            got = sess.run(None, {"X": a, "A": a, "B": b})[0]
            assert_allclose(expected, got)

        with self.subTest(opt="max"):
            lr, f = TestRuntimeProtoRun._linear_regression(clip=True, min_value=None)
            x = np.array([[0, 1], [2, 3]], dtype=np.float32)
            a = np.array([1, 1], dtype=np.float32)
            b = np.array([11], dtype=np.float32)
            expected = f(x, a, b)
            sess = ProtoRun(lr)
            last_node = sess.rt_nodes_[-1]
            self.assertEqual(last_node.__class__.__name__, "Clip_11")
            got = sess.run(None, {"X": a, "A": a, "B": b})[0]
            assert_allclose(expected, got)

        with self.subTest(opt="min"):
            lr, f = TestRuntimeProtoRun._linear_regression(clip=True, max_value=None)
            x = np.array([[0, 1], [2, 3]], dtype=np.float32)
            a = np.array([1, 1], dtype=np.float32)
            b = np.array([11], dtype=np.float32)
            expected = f(x, a, b)
            sess = ProtoRun(lr)
            last_node = sess.rt_nodes_[-1]
            self.assertEqual(last_node.__class__.__name__, "Clip_11")
            got = sess.run(None, {"X": a, "A": a, "B": b})[0]
            assert_allclose(expected, got)

    def test_ProtoRun_lr_clip_6(self):
        with self.subTest(opt="min+max"):
            lr, f = TestRuntimeProtoRun._linear_regression(clip=True, opset=10)
            x = np.array([[0, 1], [2, 3]], dtype=np.float32)
            a = np.array([1, 1], dtype=np.float32)
            b = np.array([11], dtype=np.float32)
            expected = f(x, a, b)
            sess = ProtoRun(lr)
            last_node = sess.rt_nodes_[-1]
            self.assertEqual(last_node.__class__.__name__, "Clip_6")
            self.assertEqual(last_node.min, -1)
            self.assertEqual(last_node.max, 1)
            got = sess.run(None, {"X": a, "A": a, "B": b})[0]
            assert_allclose(expected, got)

        with self.subTest(opt="max"):
            lr, f = TestRuntimeProtoRun._linear_regression(
                clip=True, opset=10, min_value=None
            )
            x = np.array([[0, 1], [2, 3]], dtype=np.float32)
            a = np.array([1, 1], dtype=np.float32)
            b = np.array([11], dtype=np.float32)
            expected = f(x, a, b)
            sess = ProtoRun(lr)
            last_node = sess.rt_nodes_[-1]
            self.assertEqual(last_node.__class__.__name__, "Clip_6")
            self.assertEqual(last_node.max, 1)
            self.assertEqual(last_node.min, -3.4028234663852886e38)
            got = sess.run(None, {"X": a, "A": a, "B": b})[0]
            assert_allclose(expected, got)

        with self.subTest(opt="min"):
            lr, f = TestRuntimeProtoRun._linear_regression(
                clip=True, opset=10, max_value=None
            )
            x = np.array([[0, 1], [2, 3]], dtype=np.float32)
            a = np.array([1, 1], dtype=np.float32)
            b = np.array([11], dtype=np.float32)
            expected = f(x, a, b)
            sess = ProtoRun(lr)
            last_node = sess.rt_nodes_[-1]
            self.assertEqual(last_node.__class__.__name__, "Clip_6")
            self.assertEqual(last_node.min, -1)
            self.assertEqual(last_node.max, 3.4028234663852886e38)
            got = sess.run(None, {"X": a, "A": a, "B": b})[0]
            assert_allclose(expected, got)

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

        sess = ProtoRun(m)
        x = np.array([0, 1, 3], dtype=np.uint8).reshape((1, 1, 3))
        result = sess.run(None, {"x": x})[0]
        expected = x
        assert_allclose(expected, result)

    def test_reduce_sum_11(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        node1 = make_node("ReduceSum", ["X"], ["Y"], axes=[1], keepdims=1)
        graph = make_graph([node1], "rs", [X], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 11)])
        check_model(onnx_model)
        x = np.arange(60).reshape((3, 4, 5)).astype(np.float32)
        expected = x.sum(axis=1, keepdims=1)
        sess = ProtoRun(onnx_model)
        got = sess.run(None, {"X": x})[0]
        assert_allclose(expected, got)

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
        sess = ProtoRun(onnx_model)
        got = sess.run(None, {"X": x, "A": a})[0]
        assert_allclose(expected, got)

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
        sess = ProtoRun(onnx_model)
        got = sess.run(None, {"X": x, "A": a})[0]
        assert_allclose(expected, got)

    def test_reduce_sum_13_empty_axes_noop(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        node1 = make_node("ReduceSum", ["X"], ["Y"], keepdims=1, noop_with_empty_axes=1)
        graph = make_graph([node1], "rs", [X], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 13)])
        check_model(onnx_model)
        x = np.arange(60).reshape((3, 4, 5)).astype(np.float32)
        sess = ProtoRun(onnx_model)
        got = sess.run(None, {"X": x})[0]
        assert_allclose(x, got)

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
        sess = ProtoRun(onnx_model)
        got = sess.run(None, {"X": x, "Y": y})[0]
        assert_allclose(expected, got)

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
        sess = ProtoRun(onnx_model)
        got = sess.run(None, {"X": x, "Y": y})[0]
        assert_allclose(expected, got)

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

        sess = ProtoRun(model_def)
        self.assertEqual(str(sess), "ProtoRun(X) -> Z")

        x = np.array([1, 2], dtype=np.float32)
        got = sess.run(None, {"X": x})[0]
        assert_allclose(np.array([1], dtype=np.float32), got)

        x = np.array([-1, -2], dtype=np.float32)
        got = sess.run(None, {"X": x})[0]
        assert_allclose(np.array([0], dtype=np.float32), got)

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

        sess = ProtoRun(m)
        result = sess.run(None, {"cond": np.array(True)})
        expected = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        assert_allclose(expected, result[0])

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
        sess = ProtoRun(onnx_model)
        x = np.arange(6).reshape((3, 2)).astype(np.float32)
        a = np.array([1, -1], dtype=np.float32)
        result = sess.run(None, {"X": x, "A": a})[0]
        expected = np.abs(x @ a + 0.67)
        assert_allclose(expected, result)

    def test_custom_node(self):
        class _InvAlpha:

            op_domain = "custom"

            def __init__(self, onnx_node, run_params):  # type: ignore
                self.onnx_node = onnx_node
                self.run_params = run_params

            def _run(self, x):  # type: ignore
                return (1 / (x + self.alpha),)

        class InvAlpha_(OpRun):
            def _run(self, x):  # type: ignore
                return (1 / (x + self.alpha),)

        class InvAlpha(OpRun):

            op_domain = "custom"

            def _run(self, x, alpha=None):  # type: ignore
                alpha = alpha or self.alpha  # type: ignore
                return (1 / (x + alpha),)

        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        node1 = make_node("InvAlpha", ["X"], ["Y"], alpha=0.5, domain="custom")
        graph = make_graph([node1], "rs", [X], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("custom", 1)])
        x = np.arange(60).reshape((3, 4, 5)).astype(np.float32) + 1
        with self.assertRaises(NotImplementedError):
            ProtoRun(onnx_model)

        node1 = make_node("_InvAlpha", ["X"], ["Y"], alpha=0.5, domain="custom")
        graph = make_graph([node1], "rs", [X], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("custom", 1)])
        with self.assertRaises(TypeError):
            ProtoRun(onnx_model, new_ops=[_InvAlpha])

        node1 = make_node("InvAlpha_", ["X"], ["Y"], alpha=0.5, domain="custom")
        graph = make_graph([node1], "rs", [X], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("custom", 1)])
        with self.assertRaises(NotImplementedError):
            ProtoRun(onnx_model, new_ops=[InvAlpha_])

        node1 = make_node("InvAlpha", ["X"], ["Y"], alpha=0.5, domain="custom")
        graph = make_graph([node1], "rs", [X], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("custom", 1)])
        with self.assertRaises(ValueError):
            ProtoRun(onnx_model, new_ops=[InvAlpha, InvAlpha])
        sess = ProtoRun(onnx_model, new_ops=[InvAlpha])
        got = sess.run(None, {"X": x})[0]
        expected = 1 / (x + 0.5)
        assert_allclose(expected, got)

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
        oinf = ProtoRun(model_def)
        inputs = {"trip_count": trip_count, "cond": cond, "seq_empty": seq_empty}
        got = oinf.run(None, inputs)
        assert_allclose(expected, got[0])

    def test_onnxt_runtime_bernoulli(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        node1 = make_node("Bernoulli", ["X"], ["Y"], seed=0.0)
        graph = make_graph([node1], "g", [X], [Y])
        onnx_model = make_model(graph)
        check_model(onnx_model)
        sess = ProtoRun(onnx_model)
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
        sess = ProtoRun(onnx_model)
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
        sess = ProtoRun(onnx_model)
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
        sess = ProtoRun(onnx_model)
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
        sess = ProtoRun(onnx_model)
        got = sess.run(None, {"X": np.zeros((2, 4), dtype=np.float32)})[0]
        self.assertEqual(got.shape, (2, 4))
        self.assertEqual(got.dtype, np.float32)

    def test_eval_celu(self):
        inst = Celu.create(alpha=0.5)
        self.assertEqual(inst.alpha, 0.5)
        x = np.array([[0, 1], [-1, 2]], dtype=np.float32)
        y = Celu.eval(x, alpha=0.5)
        expected = _vcelu1(x, alpha=0.5)
        assert_allclose(expected, y)

    def test_eval_celu_load_op(self):
        celu = load_op("", "Celu")
        self.assertEqual(celu.op_domain, "")
        inst = celu.create(alpha=0.5)
        self.assertEqual(inst.alpha, 0.5)
        x = np.array([[0, 1], [-1, 2]], dtype=np.float32)
        y = celu.eval(x, alpha=0.5)
        expected = _vcelu1(x, alpha=0.5)
        assert_allclose(expected, y)

    def test_create_adam(self):
        inst = Adam.create(alpha=0.5)
        self.assertEqual(inst.alpha, 0.5)

    @skip_if_no_onnxruntime
    def test_conv(self):
        import onnxruntime as ort

        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None, None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None, None, None])
        B = make_tensor_value_info("B", TensorProto.FLOAT, [None, None, None, None])
        W = make_tensor_value_info("W", TensorProto.FLOAT, [None, None, None, None])
        node = make_node(
            "Conv",
            ["X", "W", "B"],
            ["Y"],
            pads=[1, 1, 1, 1],
            dilations=[1, 1],
            strides=[2, 2],
        )
        graph = make_graph([node], "g", [X, W, B], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 16)])

        sess1 = ort.InferenceSession(onnx_model.SerializeToString())
        sess2 = ProtoRun(onnx_model)

        sH, sW = 5, 6
        for i in range(sH):
            for j in range(sW):
                X = np.zeros((1, 1, sH, sW), dtype=np.float32)
                X[0, 0, i, j] = 1.0
                W = np.zeros((1, 1, 3, 3), dtype=np.float32)
                W[0, 0, :, :] = np.minimum(2 ** np.arange(9).reshape((3, -1)), 256)

                B = np.array([[[[0]]]], dtype=np.float32)
                expected = sess1.run(None, {"X": X, "W": W, "B": B})[0]
                got = sess2.run(None, {"X": X, "W": W, "B": B})[0]
                assert_allclose(expected, got)

    @skip_if_no_onnxruntime
    def test_qlinearconv(self):
        import onnxruntime as ort

        x = make_tensor_value_info("x", TensorProto.UINT8, [None, None, None, None])
        w = make_tensor_value_info("w", TensorProto.UINT8, [None, None, None, None])
        y = make_tensor_value_info("y", TensorProto.UINT8, [None, None, None, None])
        x_scale = make_tensor_value_info("x_scale", TensorProto.FLOAT, [None])
        w_scale = make_tensor_value_info("w_scale", TensorProto.FLOAT, [None])
        y_scale = make_tensor_value_info("y_scale", TensorProto.FLOAT, [None])
        x_zero_point = make_tensor_value_info("x_zero_point", TensorProto.UINT8, [None])
        w_zero_point = make_tensor_value_info("w_zero_point", TensorProto.UINT8, [None])
        y_zero_point = make_tensor_value_info("y_zero_point", TensorProto.UINT8, [None])

        node = make_node(
            "QLinearConv",
            [
                "x",
                "x_scale",
                "x_zero_point",
                "w",
                "w_scale",
                "w_zero_point",
                "y_scale",
                "y_zero_point",
            ],
            ["y"],
        )
        graph = make_graph(
            [node],
            "g",
            [x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point],
            [y],
        )
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 16)])

        sess1 = ort.InferenceSession(onnx_model.SerializeToString())
        sess2 = ProtoRun(onnx_model)

        sH, sW = 3, 3
        for i in range(sH):
            for j in range(sW):
                x = np.zeros((1, 1, sH, sW), dtype=np.uint8)
                x[0, 0, i, j] = 1.0
                with self.subTest(w="1x1", i=i, j=j):
                    w = np.zeros((1, 1, 1, 1), dtype=np.uint8)
                    w[0, 0, :, :] = 1
                    feeds = {
                        "x": x,
                        "x_scale": np.array([1], dtype=np.float32),
                        "x_zero_point": np.array([0], dtype=np.uint8),
                        "w": w,
                        "w_scale": np.array([1], dtype=np.float32),
                        "w_zero_point": np.array([0], dtype=np.uint8),
                        "y_scale": np.array([1], dtype=np.float32),
                        "y_zero_point": np.array([0], np.uint8),
                    }
                    expected = sess1.run(None, feeds)[0]
                    got = sess2.run(None, feeds)[0]
                    try:
                        assert_allclose(expected, got)
                    except AssertionError as e:
                        raise e
                with self.subTest(w="3x3", i=i, j=j):
                    w = np.zeros((1, 1, 3, 3), dtype=np.uint8)
                    w[0, 0, :, :] = np.minimum(2 ** np.arange(9).reshape((3, -1)), 128)
                    feeds = {
                        "x": x,
                        "x_scale": np.array([1], dtype=np.float32),
                        "x_zero_point": np.array([0], dtype=np.uint8),
                        "w": w,
                        "w_scale": np.array([1], dtype=np.float32),
                        "w_zero_point": np.array([0], dtype=np.uint8),
                        "y_scale": np.array([1], dtype=np.float32),
                        "y_zero_point": np.array([0], np.uint8),
                    }
                    expected = sess1.run(None, feeds)[0]
                    got = sess2.run(None, feeds)[0]
                    assert_allclose(expected, got)
                with self.subTest(w="1x1", i=i, j=j):
                    w = np.zeros((1, 1, 1, 1), dtype=np.uint8)
                    w[0, 0, :, :] = 0
                    feeds = {
                        "x": x,
                        "x_scale": np.array([0.00369204697], dtype=np.float32),
                        "x_zero_point": np.array([132], dtype=np.uint8),
                        "w": w,
                        "w_scale": np.array([100.001727945750], dtype=np.float32),
                        "w_zero_point": np.array([255], dtype=np.uint8),
                        "y_scale": np.array([0.00162681262], dtype=np.float32),
                        "y_zero_point": np.array([132], np.uint8),
                    }
                    expected = sess1.run(None, feeds)[0]
                    got = sess2.run(None, feeds)[0]
                    assert_allclose(expected, got)

        x = np.array(
            [
                [255, 174, 162, 25, 203, 168, 58],
                [15, 59, 237, 95, 129, 0, 64],
                [56, 242, 153, 221, 168, 12, 166],
                [232, 178, 186, 195, 237, 162, 237],
                [188, 39, 124, 77, 80, 102, 43],
                [127, 230, 21, 83, 41, 40, 134],
                [255, 154, 92, 141, 42, 148, 247],
            ],
            dtype=np.uint8,
        ).reshape((1, 1, 7, 7))
        x_scale = np.array([0.00369204697], dtype=np.float32)
        x_zero_point = np.array([132], dtype=np.uint8)
        w = np.array([0], dtype=np.uint8).reshape((1, 1, 1, 1))
        w_scale = np.array([0.00172794575], dtype=np.float32)
        w_zero_point = np.array([255], dtype=np.uint8)
        y_scale = np.array([0.00162681262], dtype=np.float32)
        y_zero_point = np.array([123], dtype=np.uint8)

        feeds = {
            "x": x,
            "x_scale": x_scale,
            "x_zero_point": x_zero_point,
            "w": w,
            "w_scale": w_scale,
            "w_zero_point": w_zero_point,
            "y_scale": y_scale,
            "y_zero_point": y_zero_point,
        }
        expected = sess1.run(None, feeds)[0]
        got = sess2.run(None, feeds)[0]
        assert_allclose(expected, got)

    def common_test_im2col(self, kernel_shape, pads, strides, dilations):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None, None, None])
        Y1 = make_tensor_value_info("Y1", TensorProto.FLOAT, [None, None, None, None])
        Y2 = make_tensor_value_info("Y2", TensorProto.FLOAT, [None, None, None, None])
        W = make_tensor_value_info("W", TensorProto.FLOAT, [None, None, None, None])
        node = make_node(
            "Conv", ["X", "W"], ["Y1"], pads=pads, strides=strides, dilations=dilations
        )
        node_shape = make_node("Shape", ["W"], ["shape"])
        node_im = make_node(
            "Im2Col",
            ["X", "shape"],
            ["xim"],
            pads=pads,
            strides=strides,
            dilations=dilations,
            domain="experimental",
        )
        node_flat = make_node("Flatten", ["W"], ["wflat"])
        node_axes = make_node(
            "Constant", [], ["axes"], value=from_array(np.array([0], dtype=np.int64))
        )
        node_qu = make_node("Squeeze", ["wflat", "axes"], ["wsqu"])
        node_gem = make_node("MatMul", ["xim", "wsqu"], ["Y2"])
        graph = make_graph(
            [node, node_shape, node_im, node_axes, node_flat, node_qu, node_gem],
            "g",
            [X, W],
            [Y1, Y2],
        )
        onnx_model = make_model(
            graph, opset_imports=[make_opsetid("", 16), make_opsetid("experimental", 1)]
        )
        graph_conv = make_graph([node], "g", [X, W], [Y1])
        onnx_model_conv = make_model(graph_conv, opset_imports=[make_opsetid("", 16)])
        sess = ProtoRun(onnx_model)

        try:
            import onnxruntime as ort

            sess_conv = ort.InferenceSession(onnx_model_conv.SerializeToString())
        except ImportError:
            sess_conv = None

        sH, sW = 7, 7
        nker = np.prod(kernel_shape)
        for i in range(sH):
            for j in range(sW):
                X = np.zeros((1, 1, sH, sW), dtype=np.float32)
                X[0, 0, i, j] = 1.0
                W = np.zeros(
                    (
                        1,
                        1,
                    )
                    + kernel_shape,
                    dtype=np.float32,
                )
                W[0, 0, :, :] = np.minimum(
                    2 ** np.arange(nker).reshape((kernel_shape[0], -1)), 256
                )

                got = sess.run(None, {"X": X, "W": W})
                if sess_conv is not None:
                    ort_res = sess_conv.run(None, {"X": X, "W": W})[0]
                    assert_allclose(got[1], ort_res)
                try:
                    assert_allclose(got[0], got[1])
                except AssertionError as e:
                    raise AssertionError(
                        f"Discrepancies: pads={pads}, dilations={dilations}, strides={strides}, "
                        f"kernel_shape={kernel_shape}"
                        f"\n{got[0]}\n!=\n{got[1]}"
                    ) from e

    def test_im2col_1x1(self):
        self.common_test_im2col(
            (1, 1), pads=[1, 1, 1, 2], strides=[1, 1], dilations=[1, 1]
        )

    def test_im2col_2x2(self):
        self.common_test_im2col(
            (2, 2), pads=[1, 1, 1, 2], strides=[1, 1], dilations=[1, 1]
        )

    def test_im2col_3x3(self):
        self.common_test_im2col(
            (3, 3), pads=[1, 1, 1, 2], strides=[1, 1], dilations=[1, 1]
        )

    def test_im2col_3x3_pads(self):
        self.common_test_im2col(
            (3, 3), pads=[0, 1, 2, 3], strides=[1, 1], dilations=[1, 1]
        )

    def test_im2col_3x3_strides(self):
        self.common_test_im2col(
            (3, 3), pads=[0, 1, 1, 1], strides=[1, 2], dilations=[1, 1]
        )

    def test_im2col_3x3_dilations(self):
        self.common_test_im2col(
            (3, 3), pads=[1, 1, 1, 2], strides=[1, 1], dilations=[1, 2]
        )

    def test_im2col_5x5(self):
        self.common_test_im2col(
            (5, 5), pads=[1, 1, 1, 2], strides=[1, 1], dilations=[1, 1]
        )

    @skip_if_no_torch
    def test_col2im(self):
        import torch

        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None, None])
        IS = make_tensor_value_info("I", TensorProto.INT64, [None])
        BS = make_tensor_value_info("B", TensorProto.INT64, [None])
        node = make_node(
            "Col2Im",
            ["X", "I", "B"],
            ["Y"],
            pads=[0, 0, 0, 0],
            strides=[1, 1],
            dilations=[1, 1],
        )
        graph = make_graph([node], "g", [X, IS, BS], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 16)])
        sess = ProtoRun(onnx_model)

        X = np.array(
            [
                [
                    [1.0, 6.0, 11.0, 16.0, 21.0],
                    [2.0, 7.0, 12.0, 17.0, 22.0],
                    [3.0, 8.0, 13.0, 18.0, 23.0],
                    [4.0, 9.0, 14.0, 19.0, 24.0],
                    [5.0, 0.0, 15.0, 20.0, 25.0],
                ]
            ]
        ).astype(np.float32)
        image_shape = np.array([5, 5]).astype(np.int64)
        block_shape = np.array([1, 5]).astype(np.int64)

        fold = torch.nn.Fold(output_size=tuple(image_shape), kernel_size=block_shape)

        got = sess.run(None, {"X": X, "B": block_shape, "I": image_shape})
        output = fold(torch.from_numpy(X)).numpy()
        assert_allclose(output, got[0])

    def common_test_col2im(
        self, size, image_shape, block_shape, pads, strides, dilations
    ):
        import torch

        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None, None])
        IS = make_tensor_value_info("I", TensorProto.INT64, [None])
        BS = make_tensor_value_info("B", TensorProto.INT64, [None])
        node = make_node(
            "Col2Im",
            ["X", "I", "B"],
            ["Y"],
            pads=pads,
            strides=strides,
            dilations=dilations,
        )
        graph = make_graph([node], "g", [X, IS, BS], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 16)])
        sess = ProtoRun(onnx_model)

        fold = torch.nn.Fold(
            output_size=tuple(image_shape),
            kernel_size=tuple(block_shape),
            dilation=tuple(dilations),
            padding=min(pads),
            stride=tuple(strides),
        )

        nker = np.prod(block_shape)
        for i in range(nker):
            for j in range(size):
                X = np.zeros((1, nker, size), dtype=np.float32)
                X[0, i, j] = 1.0
                i_shape = np.array(image_shape, dtype=np.int64)
                b_shape = np.array(block_shape, dtype=np.int64)

                output = fold(torch.from_numpy(X)).numpy()
                got = sess.run(None, {"X": X, "B": b_shape, "I": i_shape})
                # print(output)
                # print(got)
                assert_allclose(output, got[0])

    @skip_if_no_torch
    def test_col2im_2x3(self):
        self.common_test_col2im(
            10, (6, 4), (2, 3), pads=[0, 0, 0, 0], strides=[1, 1], dilations=[1, 1]
        )

    @skip_if_no_torch
    def test_col2im_2x3_pads(self):
        self.common_test_col2im(
            28, (6, 4), (2, 3), pads=[1, 1, 1, 1], strides=[1, 1], dilations=[1, 1]
        )

    def test_col2im_2d(self):

        data = np.zeros([6, 28], dtype=np.float32)
        data[0][0] = 1.
        image_shape, kernel_shape, dilations, pads, stride = (
            np.array([6, 4]),
            (2, 3),
            np.array([1, 1]),
            np.array([1, 1, 1, 1]),
            np.array([1, 1]),
        )
        r1 = _col2im_naive_implementation_2d(
            data, image_shape, kernel_shape, dilations, pads, stride
        )
        r2 = col2im_naive_implementation(
            data, image_shape, kernel_shape, dilations, pads, stride
        )
        assert_allclose(r1, r2)


if __name__ == "__main__":
    TestRuntimeProtoRun().test_function_attribute()
    unittest.main(verbosity=2)
