# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

import numpy as np

from onnx import TensorProto, checker, defs, helper
from onnx.reference import ReferenceEvaluator


class TestGemmBias(unittest.TestCase):
    def _check(
        self, opset, dtype, c_values, expected, alpha, beta, trans_a, trans_b, broadcast
    ):
        # The literal product of these matrices is [[9,12,15], [19,26,33]].
        a = np.array([[1, 2], [3, 4]], dtype=dtype)
        b = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
        if trans_a:
            a = a.T.copy()
        if trans_b:
            b = b.T.copy()
        c = np.asarray(c_values, dtype=dtype)
        wanted = np.asarray(expected, dtype=dtype)
        self.assertEqual(wanted.shape, (2, 3))
        self.assertTrue(all(np.isfinite(value).all() for value in (a, b, c, wanted)))
        schema = defs.get_schema("Gemm", opset)
        self.assertEqual(schema.since_version, opset)
        attrs = {"alpha": alpha, "beta": beta, "transA": trans_a, "transB": trans_b}
        if broadcast is not None:
            self.assertEqual(opset, 6)
            attrs["broadcast"] = broadcast
        self.assertTrue(set(attrs).issubset(schema.attributes))
        if opset == 6 and not broadcast:
            self.assertEqual(c.shape, wanted.shape)
        else:
            self.assertEqual(np.broadcast_shapes(c.shape, wanted.shape), wanted.shape)
        tensor_type = helper.np_dtype_to_tensor_dtype(a.dtype)
        self.assertIn(
            f"tensor({TensorProto.DataType.Name(tensor_type).lower()})",
            schema.type_constraints[0].allowed_type_strs,
        )
        node = helper.make_node("Gemm", ["A", "B", "C"], ["Y"], **attrs)
        model = helper.make_model(
            helper.make_graph(
                [node],
                "gemm_bias",
                [
                    helper.make_tensor_value_info(name, tensor_type, list(value.shape))
                    for name, value in zip(("A", "B", "C"), (a, b, c), strict=False)
                ],
                [helper.make_tensor_value_info("Y", tensor_type, [2, 3])],
            ),
            opset_imports=[helper.make_opsetid("", opset)],
        )
        checker.check_model(model, full_check=True)
        originals = [value.copy() for value in (a, b, c)]
        actual = ReferenceEvaluator(model).run(None, {"A": a, "B": b, "C": c})[0]
        self.assertEqual(actual.dtype, a.dtype)
        self.assertEqual(actual.shape, wanted.shape)
        np.testing.assert_array_equal(actual, wanted)
        for value, original in zip((a, b, c), originals, strict=False):
            np.testing.assert_array_equal(value, original)

    def test_legacy_no_broadcast_beta(self):
        c = [[2, 4, 6], [8, 10, 12]]
        fixtures = [
            (1.0, 0.0, [[9, 12, 15], [19, 26, 33]]),
            (0.0, 0.0, [[0, 0, 0], [0, 0, 0]]),
            (0.0, 2.0, [[4, 8, 12], [16, 20, 24]]),
            (0.5, 0.5, [[5.5, 8, 10.5], [13.5, 18, 22.5]]),
            (1.0, 1.0, [[11, 16, 21], [27, 36, 45]]),
        ]
        for dtype in (np.float16, np.float32, np.float64):
            for broadcast in (None, 0):
                for trans_a in (0, 1):
                    for trans_b in (0, 1):
                        for alpha, beta, expected in fixtures:
                            with self.subTest(
                                dtype=dtype,
                                broadcast=broadcast,
                                trans_a=trans_a,
                                trans_b=trans_b,
                                alpha=alpha,
                                beta=beta,
                            ):
                                self._check(
                                    6,
                                    dtype,
                                    c,
                                    expected,
                                    alpha,
                                    beta,
                                    trans_a,
                                    trans_b,
                                    broadcast,
                                )

    def test_supported_bias_broadcast_and_zero_coefficients(self):
        product = [[9, 12, 15], [19, 26, 33]]
        zeros = [[0, 0, 0], [0, 0, 0]]
        biases = [
            ("scalar", 2, [[2, 2, 2], [2, 2, 2]]),
            ("row_vector", [2, 4, 6], [[2, 4, 6], [2, 4, 6]]),
            ("row_matrix", [[2, 4, 6]], [[2, 4, 6], [2, 4, 6]]),
            ("column", [[2], [8]], [[2, 2, 2], [8, 8, 8]]),
        ]
        for opset in (6, 7, 9, 11, 13):
            broadcast = 1 if opset == 6 else None
            for dtype in (np.float16, np.float32, np.float64):
                for name, c, expanded in biases:
                    for trans_a in (0, 1):
                        for trans_b in (0, 1):
                            for alpha, beta, expected in (
                                (0.0, 0.0, zeros),
                                (1.0, 0.0, product),
                                (0.0, 1.0, expanded),
                            ):
                                with self.subTest(
                                    opset=opset,
                                    dtype=dtype,
                                    bias=name,
                                    trans_a=trans_a,
                                    trans_b=trans_b,
                                    alpha=alpha,
                                    beta=beta,
                                ):
                                    self._check(
                                        opset,
                                        dtype,
                                        c,
                                        expected,
                                        alpha,
                                        beta,
                                        trans_a,
                                        trans_b,
                                        broadcast,
                                    )


if __name__ == "__main__":
    unittest.main()
