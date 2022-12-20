# SPDX-License-Identifier: Apache-2.0
# type: ignore
# pylint: disable=C3001,C0302,C0415,R0904,R0913,R0914,R0915,W0221,W0707

import unittest
from functools import wraps

import numpy as np  # type: ignore
from numpy.testing import assert_allclose  # type: ignore

from onnx import ONNX_ML, TensorProto
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

TARGET_OPSET = onnx_opset_version() - 2
TARGET_OPSET_ML = 3
OPSETS = [make_opsetid("", TARGET_OPSET), make_opsetid("ai.onnx.ml", TARGET_OPSET_ML)]


def has_onnxruntime():
    try:
        import onnxruntime  # pylint: disable=W0611

        return True
    except ImportError:
        return False


def skip_if_no_onnxruntime(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not has_onnxruntime():
            raise unittest.SkipTest("onnxruntime not installed")  # noqa
        fn(*args, **kwargs)

    return wrapper


class TestReferenceEvaluatorAiOnnxMl(unittest.TestCase):
    @staticmethod
    def _check_ort(onx, feeds, atol=0, rtol=0):
        if not has_onnxruntime():
            return
        from onnxruntime import InferenceSession

        ort = InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        sess = ReferenceEvaluator(onx)
        expected = ort.run(None, feeds)
        got = sess.run(None, feeds)
        if len(expected) != len(got):
            raise AssertionError(
                f"onnxruntime returns a different number of output "
                f"{len(expected)} != {len(sess)} (ReferenceEvaluator)."
            )
        for i, (e, g) in enumerate(zip(expected, got)):
            if e.shape != g.shape:
                raise AssertionError(
                    f"Unexpected shape {g.shape} for output {i} "
                    f"(expecting {e.shape})\n{e!r}\n---\n{g!r}."
                )
            assert_allclose(
                actual=g,
                desired=e,
                atol=atol,
                rtol=rtol,
                err_msg=f"Discrepancies for output {i}.",
            )

    @unittest.skipIf(not ONNX_ML, reason="onnx not compiled with ai.onnx.ml")
    def test_binarizer(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
        node1 = make_node("Binarizer", ["X"], ["Y"], threshold=5.5, domain="ai.onnx.ml")
        graph = make_graph([node1], "ml", [X], [Y])
        onx = make_model(graph, opset_imports=OPSETS)
        check_model(onx)
        x = np.arange(12).reshape((3, 4)).astype(np.float32)
        expected = np.array(
            [[0, 0, 0, 0], [0, 0, 1, 1], [1, 1, 1, 1]], dtype=np.float32
        )
        self._check_ort(onx, {"X": x})
        sess = ReferenceEvaluator(onx)
        got = sess.run(None, {"X": x})[0]
        assert_allclose(expected, got)

    @unittest.skipIf(not ONNX_ML, reason="onnx not compiled with ai.onnx.ml")
    def test_scaler(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
        node1 = make_node(
            "Scaler", ["X"], ["Y"], scale=[0.5], offset=[-4.5], domain="ai.onnx.ml"
        )
        graph = make_graph([node1], "ml", [X], [Y])
        onx = make_model(graph, opset_imports=OPSETS)
        check_model(onx)
        x = np.arange(12).reshape((3, 4)).astype(np.float32)
        expected = np.array(
            [
                [2.25, 2.75, 3.25, 3.75],
                [4.25, 4.75, 5.25, 5.75],
                [6.25, 6.75, 7.25, 7.75],
            ],
            dtype=np.float32,
        )
        self._check_ort(onx, {"X": x})
        sess = ReferenceEvaluator(onx)
        got = sess.run(None, {"X": x})[0]
        assert_allclose(expected, got)

    @unittest.skipIf(not ONNX_ML, reason="onnx not compiled with ai.onnx.ml")
    def test_array_feature_extractor(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        A = make_tensor_value_info("A", TensorProto.INT64, [None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
        node1 = make_node(
            "ArrayFeatureExtractor", ["X", "A"], ["Y"], domain="ai.onnx.ml"
        )
        graph = make_graph([node1], "ml", [X, A], [Y])
        onx = make_model(graph, opset_imports=OPSETS)
        check_model(onx)
        x = np.arange(12).reshape((3, 4)).astype(np.float32)

        expected = np.array([[0, 4, 8]], dtype=np.float32).T
        feeds = {"X": x, "A": np.array([0], dtype=np.int64)}
        self._check_ort(onx, feeds)
        sess = ReferenceEvaluator(onx)
        got = sess.run(None, feeds)[0]
        assert_allclose(expected, got)

        expected = np.array([[0, 4, 8], [1, 5, 9]], dtype=np.float32).T
        feeds = {"X": x, "A": np.array([0, 1], dtype=np.int64)}
        self._check_ort(onx, feeds)
        sess = ReferenceEvaluator(onx)
        got = sess.run(None, feeds)[0]
        assert_allclose(expected, got)

        expected = np.array(
            [[0, 4, 8], [1, 5, 9], [0, 4, 8], [1, 5, 9], [0, 4, 8], [1, 5, 9]],
            dtype=np.float32,
        ).T
        feeds = {"X": x, "A": np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)}
        self._check_ort(onx, feeds)
        sess = ReferenceEvaluator(onx)
        got = sess.run(None, feeds)[0]
        assert_allclose(expected, got)

    @unittest.skipIf(not ONNX_ML, reason="onnx not compiled with ai.onnx.ml")
    def test_normalizer(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
        x = np.arange(12).reshape((3, 4)).astype(np.float32)
        expected = {
            "MAX": x / x.max(axis=1, keepdims=1),
            "L1": x / np.abs(x).sum(axis=1, keepdims=1),
            "L2": x / (x**2).sum(axis=1, keepdims=1) ** 0.5,
        }
        for norm, value in expected.items():
            with self.subTest(norm=norm):
                node1 = make_node(
                    "Normalizer", ["X"], ["Y"], norm=norm, domain="ai.onnx.ml"
                )
                graph = make_graph([node1], "ml", [X], [Y])
                onx = make_model(graph, opset_imports=OPSETS)
                check_model(onx)

                feeds = {"X": x}
                self._check_ort(onx, feeds, atol=1e-6)
                sess = ReferenceEvaluator(onx)
                got = sess.run(None, feeds)[0]
                assert_allclose(value, got, atol=1e-6)

    @unittest.skipIf(not ONNX_ML, reason="onnx not compiled with ai.onnx.ml")
    def test_feature_vectorizer(self):
        X = [
            make_tensor_value_info("X0", TensorProto.FLOAT, [None, None]),
            make_tensor_value_info("X1", TensorProto.FLOAT, [None, None]),
        ]
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
        x = [
            np.arange(9).reshape((3, 3)).astype(np.float32),
            np.arange(9).reshape((3, 3)).astype(np.float32) + 0.5,
        ]
        expected = {
            (1,): np.array([[0], [3], [6]], dtype=np.float32),
            (2,): np.array([[0, 1], [3, 4], [6, 7]], dtype=np.float32),
            (4,): np.array(
                [[0, 1, 2, 0], [3, 4, 5, 0], [6, 7, 8, 0]], dtype=np.float32
            ),
            (1, 1): np.array([[0, 0.5], [3, 3.5], [6, 6.5]], dtype=np.float32),
            (0, 1): np.array([[0.5], [3.5], [6.5]], dtype=np.float32),
        }
        for inputdimensions, value in expected.items():
            att = (
                list(inputdimensions)
                if isinstance(inputdimensions, tuple)
                else inputdimensions
            )
            with self.subTest(inputdimensions=att):
                node1 = make_node(
                    "FeatureVectorizer",
                    [f"X{i}" for i in range(len(att))],
                    ["Y"],
                    inputdimensions=att,
                    domain="ai.onnx.ml",
                )
                graph = make_graph([node1], "ml", X[: len(att)], [Y])
                onx = make_model(graph, opset_imports=OPSETS)
                check_model(onx)

                feeds = {f"X{i}": v for i, v in enumerate(x[: len(att)])}
                self._check_ort(onx, feeds, atol=1e-6)
                sess = ReferenceEvaluator(onx)
                got = sess.run(None, feeds)[0]
                assert_allclose(value, got, atol=1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
