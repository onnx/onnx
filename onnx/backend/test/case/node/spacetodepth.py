# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class SpaceToDepth(Base):
    @staticmethod
    def export() -> None:
        b, c, h, w = shape = (2, 2, 6, 6)
        blocksize = 2
        node = onnx.helper.make_node(
            "SpaceToDepth",
            inputs=["x"],
            outputs=["y"],
            blocksize=blocksize,
        )
        x = np.random.random_sample(shape).astype(np.float32)
        tmp = np.reshape(
            x, [b, c, h // blocksize, blocksize, w // blocksize, blocksize]
        )
        tmp = np.transpose(tmp, [0, 3, 5, 1, 2, 4])
        y = np.reshape(tmp, [b, c * (blocksize**2), h // blocksize, w // blocksize])
        expect(node, inputs=[x], outputs=[y], name="test_spacetodepth")

    @staticmethod
    def export_example() -> None:
        node = onnx.helper.make_node(
            "SpaceToDepth",
            inputs=["x"],
            outputs=["y"],
            blocksize=2,
        )

        # (1, 1, 4, 6) input tensor
        x = np.array(
            [
                [
                    [
                        [0, 6, 1, 7, 2, 8],
                        [12, 18, 13, 19, 14, 20],
                        [3, 9, 4, 10, 5, 11],
                        [15, 21, 16, 22, 17, 23],
                    ]
                ]
            ]
        ).astype(np.float32)

        # (1, 4, 2, 3) output tensor
        y = np.array(
            [
                [
                    [[0, 1, 2], [3, 4, 5]],
                    [[6, 7, 8], [9, 10, 11]],
                    [[12, 13, 14], [15, 16, 17]],
                    [[18, 19, 20], [21, 22, 23]],
                ]
            ]
        ).astype(np.float32)
        expect(node, inputs=[x], outputs=[y], name="test_spacetodepth_example")

    @staticmethod
    def export_dcr_mode_example() -> None:
        node = onnx.helper.make_node(
            "SpaceToDepth",
            inputs=["x"],
            outputs=["y"],
            blocksize=2,
            mode="DCR",
        )

        # (1, 2, 4, 6) input tensor
        x = np.array(
            [
                [
                    [
                        [0.0, 18.0, 1.0, 19.0, 2.0, 20.0],
                        [36.0, 54.0, 37.0, 55.0, 38.0, 56.0],
                        [3.0, 21.0, 4.0, 22.0, 5.0, 23.0],
                        [39.0, 57.0, 40.0, 58.0, 41.0, 59.0],
                    ],
                    [
                        [9.0, 27.0, 10.0, 28.0, 11.0, 29.0],
                        [45.0, 63.0, 46.0, 64.0, 47.0, 65.0],
                        [12.0, 30.0, 13.0, 31.0, 14.0, 32.0],
                        [48.0, 66.0, 49.0, 67.0, 50.0, 68.0],
                    ],
                ]
            ]
        ).astype(np.float32)

        # (1, 8, 2, 3) output tensor
        y = np.array(
            [
                [
                    [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
                    [[9.0, 10.0, 11.0], [12.0, 13.0, 14.0]],
                    [[18.0, 19.0, 20.0], [21.0, 22.0, 23.0]],
                    [[27.0, 28.0, 29.0], [30.0, 31.0, 32.0]],
                    [[36.0, 37.0, 38.0], [39.0, 40.0, 41.0]],
                    [[45.0, 46.0, 47.0], [48.0, 49.0, 50.0]],
                    [[54.0, 55.0, 56.0], [57.0, 58.0, 59.0]],
                    [[63.0, 64.0, 65.0], [66.0, 67.0, 68.0]],
                ]
            ]
        ).astype(np.float32)
        expect(node, inputs=[x], outputs=[y], name="test_spacetodepth_dcr_mode_example")

    @staticmethod
    def export_crd_mode_example() -> None:
        node = onnx.helper.make_node(
            "SpaceToDepth",
            inputs=["x"],
            outputs=["y"],
            blocksize=2,
            mode="CRD",
        )

        # (1, 2, 4, 6) input tensor
        x = np.array(
            [
                [
                    [
                        [0.0, 9.0, 1.0, 10.0, 2.0, 11.0],
                        [18.0, 27.0, 19.0, 28.0, 20.0, 29.0],
                        [3.0, 12.0, 4.0, 13.0, 5.0, 14.0],
                        [21.0, 30.0, 22.0, 31.0, 23.0, 32.0],
                    ],
                    [
                        [36.0, 45.0, 37.0, 46.0, 38.0, 47.0],
                        [54.0, 63.0, 55.0, 64.0, 56.0, 65.0],
                        [39.0, 48.0, 40.0, 49.0, 41.0, 50.0],
                        [57.0, 66.0, 58.0, 67.0, 59.0, 68.0],
                    ],
                ]
            ]
        ).astype(np.float32)

        # (1, 8, 2, 3) output tensor
        y = np.array(
            [
                [
                    [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
                    [[9.0, 10.0, 11.0], [12.0, 13.0, 14.0]],
                    [[18.0, 19.0, 20.0], [21.0, 22.0, 23.0]],
                    [[27.0, 28.0, 29.0], [30.0, 31.0, 32.0]],
                    [[36.0, 37.0, 38.0], [39.0, 40.0, 41.0]],
                    [[45.0, 46.0, 47.0], [48.0, 49.0, 50.0]],
                    [[54.0, 55.0, 56.0], [57.0, 58.0, 59.0]],
                    [[63.0, 64.0, 65.0], [66.0, 67.0, 68.0]],
                ]
            ]
        ).astype(np.float32)
        expect(node, inputs=[x], outputs=[y], name="test_spacetodepth_crd_mode_example")
