# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

import numpy as np
from numpy.testing import assert_array_equal

from onnx import checker, helper
from onnx.reference import ReferenceEvaluator


class TestReferenceEvaluatorPadString(unittest.TestCase):
    def _check_pad(
        self, opset, data, pads, expected, constant_value=None, axes=None, mode=None
    ):
        feeds = {"data": data, "pads": np.array(pads, dtype=np.int64)}
        inputs = ["data", "pads"]
        if constant_value is not None:
            feeds["constant_value"] = constant_value
            inputs.append("constant_value")
        elif axes is not None:
            inputs.append("")
        if axes is not None:
            feeds["axes"] = np.array(axes, dtype=np.int64)
            inputs.append("axes")
        attrs = {} if mode is None else {"mode": mode}
        node = helper.make_node("Pad", inputs, ["output"], **attrs)
        infos = [
            helper.make_tensor_value_info(
                name, helper.np_dtype_to_tensor_dtype(value.dtype), value.shape
            )
            for name, value in feeds.items()
        ]
        model = helper.make_model(
            helper.make_graph(
                [node],
                "pad_string",
                infos,
                [
                    helper.make_tensor_value_info(
                        "output",
                        helper.np_dtype_to_tensor_dtype(data.dtype),
                        expected.shape,
                    )
                ],
            ),
            opset_imports=[helper.make_opsetid("", opset)],
        )
        checker.check_model(model)
        actual = ReferenceEvaluator(model).run(None, feeds)[0]
        assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, data.dtype)

    def test_string_default_constant(self):
        for opset in (13, 18, 19, 21):
            for dtype in (np.str_, object):
                for mode in (None, "constant"):
                    with self.subTest(opset=opset, dtype=dtype, mode=mode):
                        data = np.array(["a", "b"], dtype=dtype)
                        self._check_pad(
                            opset,
                            data,
                            [1, 1],
                            np.array(["", "a", "b", ""], dtype=data.dtype),
                            mode=mode,
                        )

    def test_string_default_after_cropping(self):
        for opset in (13, 18, 19, 21):
            for dtype in (np.str_, object):
                for pads, values in (([-1, 2], ["b", "", ""]), ([-3, 4], ["", "", ""])):
                    with self.subTest(opset=opset, dtype=dtype, pads=pads):
                        data = np.array(["a", "b"], dtype=dtype)
                        self._check_pad(
                            opset, data, pads, np.array(values, dtype=data.dtype)
                        )

    def test_string_default_with_axes(self):
        data = np.array([["a", "b"], ["c", "d"]], dtype=object)
        expected = np.array([["", "a", "b", ""], ["", "c", "d", ""]], dtype=object)
        for opset in (18, 19, 21):
            for axes in ([1], [-1]):
                with self.subTest(opset=opset, axes=axes):
                    self._check_pad(opset, data, [1, 1], expected, axes=axes)

    def test_explicit_string_and_numeric_defaults(self):
        for opset in (13, 18, 19, 21):
            for dtype in (np.str_, object):
                with self.subTest(opset=opset, explicit_string=dtype):
                    data = np.array(["a", "b"], dtype=dtype)
                    self._check_pad(
                        opset,
                        data,
                        [1, 1],
                        np.array(["#", "a", "b", "#"], dtype=data.dtype),
                        constant_value=np.array("#", dtype=data.dtype),
                    )
            for dtype in (np.float32, np.int64, np.bool_):
                with self.subTest(opset=opset, numeric_dtype=dtype):
                    self._check_pad(
                        opset,
                        np.array([1, 0], dtype=dtype),
                        [1, 1],
                        np.array([0, 1, 0, 0], dtype=dtype),
                    )


if __name__ == "__main__":
    unittest.main()
