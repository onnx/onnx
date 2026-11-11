# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pytest

from onnx import parser
from onnx.reference import ReferenceEvaluator


class TestReferenceEvaluatorLoop:
    @pytest.mark.parametrize("shape", [(), (2,), (2, 3), (2, 3, 4)])
    def test_scan_output_stacks_iterations_on_new_axis(
        self, shape: tuple[int, ...]
    ) -> None:
        # Each iteration emits a tensor of `shape`. The scan output must have
        # the iterations stacked along a new leading axis: (trip_count, *shape).
        type_str = "float" + (str(list(shape)) if shape else "")
        model = parser.parse_model(f"""
            <ir_version: 10, opset_import: ["": 18]>
            g (int64 M, bool C, {type_str} X) => ({type_str} Y, float S) {{
                Y, S = Loop<body = b (int64 i, bool c, {type_str} a)
                        => (bool c2, {type_str} a2, {type_str} s) {{
                    c2 = Identity(c)
                    a2 = Add(a, a)
                    s = Identity(a2)
                }}>(M, C, X)
            }}
        """)
        x = np.ones(shape, dtype=np.float32)
        feeds = {"M": np.array(3, dtype=np.int64), "C": np.array(True), "X": x}
        y, s = ReferenceEvaluator(model).run(None, feeds)
        np.testing.assert_array_equal(y, x * 8)
        np.testing.assert_array_equal(s, np.stack([x * 2, x * 4, x * 8]))
