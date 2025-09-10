# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest
import numpy as np

import onnx
from onnx import helper
from onnx.reference import ReferenceEvaluator
from onnx.numpy_helper import to_array


class TestFP6(unittest.TestCase):
    def test_pack_unpack_roundtrip_even(self):
        vals = np.array([0.0, 0.5, 1.0, 2.0], dtype=np.float32)
        tp = helper.make_tensor(
            "t",
            onnx.TensorProto.FLOAT6E2M3,
            [4],
            vals,
            raw=True,
        )
        back = to_array(tp)
        self.assertEqual(back.shape, (4,))

    def test_pack_unpack_roundtrip_odd(self):
        vals = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        tp = helper.make_tensor(
            "t",
            onnx.TensorProto.FLOAT6E3M2,
            [3],
            vals,
            raw=True,
        )
        back = to_array(tp)
        self.assertEqual(back.shape, (3,))

    def _cast_model(self, to_dtype: int) -> onnx.ModelProto:
        return helper.make_model(
            helper.make_graph(
                [helper.make_node("Cast", ["X"], ["Y"], to=to_dtype)],
                "g",
                [helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [6])],
                [helper.make_tensor_value_info("Y", to_dtype, [6])],
            )
        )

    def test_cast_fp6_e2m3(self):
        m = self._cast_model(onnx.TensorProto.FLOAT6E2M3)
        ref = ReferenceEvaluator(m)
        x = np.array([0.0, -0.0, 0.125, 0.2, 1.0, 1000.0], dtype=np.float32)
        (y,) = ref.run(None, {"X": x})
        self.assertEqual(y.shape, x.shape)

    def test_cast_fp6_e3m2(self):
        m = self._cast_model(onnx.TensorProto.FLOAT6E3M2)
        ref = ReferenceEvaluator(m)
        x = np.array([0.0, -0.0, 0.0625, 0.5, 2.0, 1e6], dtype=np.float32)
        (y,) = ref.run(None, {"X": x})
        self.assertEqual(y.shape, x.shape)

    def _qdq_model(self, qdtype: int) -> onnx.ModelProto:
        g = helper.make_graph(
            [
                helper.make_node("QuantizeLinear", ["X", "S"], ["Q"], saturate=1, output_dtype=qdtype),
                helper.make_node("DequantizeLinear", ["Q", "S"], ["Y"], output_dtype=onnx.TensorProto.FLOAT),
            ],
            "g",
            [
                helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [6]),
                helper.make_tensor_value_info("S", onnx.TensorProto.FLOAT, []),
            ],
            [helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [6])],
        )
        return helper.make_model(g)

    def test_qdq_fp6_paths(self):
        ref = ReferenceEvaluator(self._qdq_model(onnx.TensorProto.FLOAT6E2M3))
        x = np.array([0.0, -0.0, 0.125, 1.0, 8.0, 1000.0], dtype=np.float32)
        (y,) = ref.run(None, {"X": x, "S": np.array(1.0, dtype=np.float32)})
        self.assertEqual(y.shape, x.shape)

        ref2 = ReferenceEvaluator(self._qdq_model(onnx.TensorProto.FLOAT6E3M2))
        (y2,) = ref2.run(None, {"X": x, "S": np.array(1.0, dtype=np.float32)})
        self.assertEqual(y2.shape, x.shape)


if __name__ == "__main__":
    unittest.main(verbosity=2)

