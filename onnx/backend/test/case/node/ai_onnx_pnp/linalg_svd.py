# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class LinalgSVD(Base):
    @staticmethod
    def export() -> None:
        threshold = 1.0
        node = onnx.helper.make_node(
            "LinalgSVD",
            inputs=["A"],
            outputs=["U", "S", "Vh"],
            threshold=threshold,
            domain="ai.onnx.pnp",
        )

        A = np.array([
            [
                [-1.125840, -1.152360, -0.250579, -0.433879],
                [0.848710, 0.692009, -0.316013, -2.115219],
                [0.468096, -0.157712, 1.443660, 0.266049],
            ],
            [
                [0.166455, 0.874382, -0.143474, -0.111609],
                [0.931827, 1.259009, 2.004981, 0.053737],
                [0.618057, -0.412802, -0.841065, -2.316042]
            ]
        ])
        U, S, Vh = np.linalg.svd(A, full_matrices=True)        
        expect(node, inputs=[A], outputs=[U, S, Vh], name="test_ai_onnx_pnp_linalg_svd")
