# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import unittest

import ml_dtypes
import numpy as np

import onnx
from onnx import helper
from onnx.numpy_helper import to_array
from onnx.reference import ReferenceEvaluator


class TestFP6(unittest.TestCase):
    def test_pack_unpack_roundtrip_even(self):
        vals = np.array([0.0, 0.5, 1.0, 2.0], dtype=np.float32)
        tp = helper.make_tensor(
            "t",
            onnx.TensorProto.FLOAT6E2M3,
            [4],
            vals.astype(ml_dtypes.float6_e2m3fn),
            raw=True,
        )
        # 4 six-bit values pack exactly into 3 bytes.
        self.assertEqual(len(tp.raw_data), 3)
        back = to_array(tp)
        self.assertEqual(back.shape, (4,))
        # All 4 values are exactly representable in E2M3 (bias=1), so the
        # round-trip should be exact, not just shape-preserving.
        np.testing.assert_array_equal(back.astype(np.float32), vals)

    def test_pack_unpack_roundtrip_odd(self):
        vals = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        tp = helper.make_tensor(
            "t",
            onnx.TensorProto.FLOAT6E3M2,
            [3],
            vals.astype(ml_dtypes.float6_e3m2fn),
            raw=True,
        )
        # 3 six-bit values need ceil(3*6/8) = 3 bytes (padded).
        self.assertEqual(len(tp.raw_data), math.ceil(3 * 6 / 8))
        back = to_array(tp)
        self.assertEqual(back.shape, (3,))
        # All 3 values are exactly representable in E3M2 (bias=3) too.
        np.testing.assert_array_equal(back.astype(np.float32), vals)

    def test_to_array_raises_on_truncated_raw_data(self):
        # 4 elements need ceil(4*6/8) = 3 packed bytes; give only 2.
        tp = helper.make_tensor(
            "t",
            onnx.TensorProto.FLOAT6E2M3,
            [4],
            np.zeros(4, dtype=ml_dtypes.float6_e2m3fn),
            raw=True,
        )
        tp.raw_data = tp.raw_data[:2]
        with self.assertRaises(ValueError):
            to_array(tp)

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
        onnx.checker.check_model(m)
        ref = ReferenceEvaluator(m)
        x = np.array([0.0, -0.0, 0.125, 0.2, 1.0, 1000.0], dtype=np.float32)
        (y,) = ref.run(None, {"X": x})
        self.assertEqual(y.shape, x.shape)
        # Expected values computed via ml_dtypes directly -- the same
        # authoritative cast Cast delegates to -- so this catches any
        # regression back to hand-rolled (and previously incorrect) bit math,
        # not just a shape-preserving no-op.
        expected = x.astype(ml_dtypes.float6_e2m3fn).astype(np.float32)
        np.testing.assert_array_equal(y.astype(np.float32), expected)
        # assert_array_equal treats 0.0 == -0.0, so it alone wouldn't catch a
        # regression that canonicalizes -0.0 to +0.0; check the sign bit too.
        np.testing.assert_array_equal(
            np.signbit(y.astype(np.float32)), np.signbit(expected)
        )

    def test_cast_fp6_e3m2(self):
        m = self._cast_model(onnx.TensorProto.FLOAT6E3M2)
        onnx.checker.check_model(m)
        ref = ReferenceEvaluator(m)
        x = np.array([0.0, -0.0, 0.0625, 0.5, 2.0, 1e6], dtype=np.float32)
        (y,) = ref.run(None, {"X": x})
        self.assertEqual(y.shape, x.shape)
        expected = x.astype(ml_dtypes.float6_e3m2fn).astype(np.float32)
        np.testing.assert_array_equal(y.astype(np.float32), expected)
        np.testing.assert_array_equal(
            np.signbit(y.astype(np.float32)), np.signbit(expected)
        )

    def _qdq_model(self, qdtype: int) -> onnx.ModelProto:
        g = helper.make_graph(
            [
                helper.make_node(
                    "QuantizeLinear", ["X", "S"], ["Q"], saturate=1, output_dtype=qdtype
                ),
                helper.make_node(
                    "DequantizeLinear",
                    ["Q", "S"],
                    ["Y"],
                    output_dtype=onnx.TensorProto.FLOAT,
                ),
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
        x = np.array([0.0, -0.0, 0.125, 1.0, 8.0, 1000.0], dtype=np.float32)
        scale = np.array(1.0, dtype=np.float32)

        ref = ReferenceEvaluator(self._qdq_model(onnx.TensorProto.FLOAT6E2M3))
        (y,) = ref.run(None, {"X": x, "S": scale})
        self.assertEqual(y.shape, x.shape)
        expected_e2m3 = x.astype(ml_dtypes.float6_e2m3fn).astype(np.float32)
        np.testing.assert_array_equal(y, expected_e2m3)
        # Note: unlike test_cast_fp6_e2m3/e3m2, this round-trip does not
        # currently preserve the sign of -0.0 (QuantizeLinear's `x +=
        # zero_point` turns -0.0 into +0.0 for the default zero_point=0,
        # a pre-existing issue shared with FLOAT4E2M1) -- see
        # https://github.com/onnx/onnx/issues/8227. No signbit assertion
        # here until that's resolved.

        ref2 = ReferenceEvaluator(self._qdq_model(onnx.TensorProto.FLOAT6E3M2))
        (y2,) = ref2.run(None, {"X": x, "S": scale})
        self.assertEqual(y2.shape, x.shape)
        expected_e3m2 = x.astype(ml_dtypes.float6_e3m2fn).astype(np.float32)
        np.testing.assert_array_equal(y2, expected_e3m2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
