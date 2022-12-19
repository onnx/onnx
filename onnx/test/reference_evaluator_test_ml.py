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
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRun
from onnx.reference.ops import load_op


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
                    f"Unexpected shape {g.shape} for output {i} (expecting {e.shape})."
                )
            assert_allclose(
                actual=g,
                desired=e,
                atol=atol,
                rtol=rtol,
                err_msg=f"Discrepancies for output {i}.",
            )

    def test_scaler(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
        node1 = make_node(
            "Scaler", ["X"], ["Y"], scale=[0.5], offset=[-4.5], domain="ai.onnx.ml"
        )
        graph = make_graph([node1], "ml", [X], [Y])
        onx = make_model(
            graph, opset_imports=[make_opsetid("", 17), make_opsetid("ai.onnx.ml", 3)]
        )
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
