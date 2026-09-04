# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx import helper
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class SplitPRNG(Base):
    @staticmethod
    def export() -> None:
        node = helper.make_node(
            "SplitPRNG", inputs=["state"], outputs=["state0", "state1"]
        )
        state = np.array([0, 0], dtype=np.int64)
        state0 = np.array([1797259609, 2579123966], dtype=np.int64)
        state1 = np.array([928981903, 3453687069], dtype=np.int64)
        expect(
            node,
            inputs=[state],
            outputs=[state0, state1],
            name="test_split_prng",
        )

    @staticmethod
    def export_with_data() -> None:
        node = helper.make_node(
            "SplitPRNG",
            inputs=["state", "data"],
            outputs=["state0", "state1"],
        )
        state = np.array([0, 0], dtype=np.int64)
        data = np.array([42, -1], dtype=np.int64)
        state0 = np.array([2814562516, 111458285], dtype=np.int64)
        state1 = np.array([145227835, 1976240827], dtype=np.int64)
        expect(
            node,
            inputs=[state, data],
            outputs=[state0, state1],
            name="test_split_prng_with_data",
        )
