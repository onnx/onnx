# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx import helper
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class InitPRNG(Base):
    @staticmethod
    def export() -> None:
        node = helper.make_node("InitPRNG", inputs=["seed"], outputs=["state"])
        seed = np.array(0x123456789ABCDEF, dtype=np.int64)
        state = np.array([0x01234567, 0x89ABCDEF], dtype=np.int64)
        expect(node, inputs=[seed], outputs=[state], name="test_init_prng")

    @staticmethod
    def export_negative_seed() -> None:
        node = helper.make_node("InitPRNG", inputs=["seed"], outputs=["state"])
        seed = np.array(-1, dtype=np.int64)
        state = np.array([0xFFFFFFFF, 0xFFFFFFFF], dtype=np.int64)
        expect(node, inputs=[seed], outputs=[state], name="test_init_prng_negative_seed")
