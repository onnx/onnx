# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Value-semantic regression for the Hardmax 12->13 adapter."""

from __future__ import annotations

import unittest

import numpy as np

import onnx
from onnx import TensorProto, helper, version_converter
from onnx.reference import ReferenceEvaluator


class TestHardmaxLegacyConversion(unittest.TestCase):
    def test_12_to_13_preserves_documented_flatten_first_outputs(self) -> None:
        # In opset 12, default axis=1 flattens the last two axes; v13 axis-wise
        # Hardmax would produce more than one 1 in the same 1x2x2 tensor.
        cases = (
            (None, (1, 2, 2), [[0.0, 0.0], [0.0, 1.0]]),
            (1, (1, 2, 2), [[0.0, 0.0], [0.0, 1.0]]),
            (0, (2, 2), [[0.0, 0.0], [0.0, 1.0]]),
        )
        for axis, shape, expected_last_rows in cases:
            with self.subTest(axis=axis, shape=shape):
                values = np.arange(1, np.prod(shape) + 1, dtype=np.float32).reshape(
                    shape
                )
                attrs = {} if axis is None else {"axis": axis}
                node = helper.make_node("Hardmax", ["X"], ["Y"], **attrs)
                graph = helper.make_graph(
                    [node],
                    "legacy_hardmax",
                    [helper.make_tensor_value_info("X", TensorProto.FLOAT, shape)],
                    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, shape)],
                )
                source = helper.make_model(
                    graph, opset_imports=[helper.make_opsetid("", 12)]
                )
                onnx.checker.check_model(source)
                converted = version_converter.convert_version(source, 13)
                onnx.checker.check_model(converted)
                got = ReferenceEvaluator(converted).run(None, {"X": values})[0]
                expected = np.asarray(expected_last_rows, dtype=np.float32).reshape(
                    shape
                )
                np.testing.assert_array_equal(got, expected)


if __name__ == "__main__":
    unittest.main()
