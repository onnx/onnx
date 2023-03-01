# SPDX-License-Identifier: Apache-2.0
# type: ignore
# pylint: disable=C3001,C0302,C0415,R0904,R0913,R0914,R0915,W0221,W0707
"""
You can run a specific test by using the following syntax.

::

    python onnx/test/reference_evaluator_c_test.py TestReferenceEvaluatorC.test_function_attribute_nested_graph
"""

import os
import unittest

import numpy as np  # type: ignore
from numpy.testing import assert_allclose  # type: ignore

from onnx import TensorProto
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_opsetid,
    make_tensor_value_info,
)
from onnx.reference import ReferenceEvaluator
from onnx.reference.c_reference_evaluator import CReferenceEvaluator


class TestReferenceEvaluatorC(unittest.TestCase):
    def test_conv(self):
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

        sess1 = ReferenceEvaluator(onnx_model)
        sess2 = CReferenceEvaluator(onnx_model)

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

    def test_light_model(self):
        model = os.path.join(
            os.path.dirname(__file__),
            "..",
            "backend",
            "test",
            "data",
            "light",
            "light_shufflenet.onnx",
        )
        if not os.path.exists(model):
            raise FileNotFoundError(os.path.abspath(model))
        sess = CReferenceEvaluator(model)
        name = sess.input_names[0]
        shape = [d.dim_value for d in sess.input_types[0].tensor_type.shape.dim]
        img = np.arange(np.prod(shape)).reshape(*shape) / np.prod(shape)
        img = img.astype(np.float32)
        got = sess.run(None, {name: img})
        expected = got[0] * 0 + 1
        expected /= expected.sum().reshape((1, -1))

        self.assertEqual(got[0].shape, (1, 1000))
        self.assertEqual(got[0].dtype, np.float32)
        assert_allclose(expected, got[0], atol=1e-5)

        try:
            from onnxruntime import InferenceSession  # pylint: disable=W0611
        except ImportError:
            return

        sess2 = InferenceSession(model, providers=["CPUExecutionProvider"])
        got2 = sess2.run(None, {name: img})
        assert_allclose(expected, got2[0], atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
