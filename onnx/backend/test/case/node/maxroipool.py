# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class MaxRoiPool(Base):
    @staticmethod
    def export_maxroipool() -> None:
        node = onnx.helper.make_node(
            "MaxRoiPool",
            inputs=["X", "rois"],
            outputs=["Y"],
            pooled_shape=[2, 2],
            spatial_scale=1.0,
        )

        X = np.array(
            [
                [
                    [
                        [0.0, 1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0, 7.0],
                        [8.0, 9.0, 10.0, 11.0],
                        [12.0, 13.0, 14.0, 15.0],
                    ]
                ]
            ],
            dtype=np.float32,
        )
        rois = np.array([[0.0, 1.0, 1.0, 2.0, 2.0]], dtype=np.float32)
        # (num_rois, C, pooled_shape[0], pooled_shape[1])
        Y = np.array([[[[5.0, 6.0], [9.0, 10.0]]]], dtype=np.float32)

        expect(
            node,
            inputs=[X, rois],
            outputs=[Y],
            name="test_maxroipool",
        )

    @staticmethod
    def export_maxroipool_multi_batch() -> None:
        node = onnx.helper.make_node(
            "MaxRoiPool",
            inputs=["X", "rois"],
            outputs=["Y"],
            pooled_shape=[2, 2],
            spatial_scale=1.0,
        )

        X = np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4)
        # The third ROI is degenerate (x1 == x2, y1 == y2).
        rois = np.array(
            [
                [0.0, 0.0, 0.0, 3.0, 3.0],
                [1.0, 1.0, 1.0, 3.0, 3.0],
                [0.0, 2.0, 2.0, 2.0, 2.0],
            ],
            dtype=np.float32,
        )
        # (num_rois, C, pooled_shape[0], pooled_shape[1])
        Y = np.array(
            [
                [
                    [[5.0, 7.0], [13.0, 15.0]],
                    [[21.0, 23.0], [29.0, 31.0]],
                ],
                [
                    [[42.0, 43.0], [46.0, 47.0]],
                    [[58.0, 59.0], [62.0, 63.0]],
                ],
                [
                    [[10.0, 10.0], [10.0, 10.0]],
                    [[26.0, 26.0], [26.0, 26.0]],
                ],
            ],
            dtype=np.float32,
        )

        expect(
            node,
            inputs=[X, rois],
            outputs=[Y],
            name="test_maxroipool_multi_batch",
        )

    @staticmethod
    def export_maxroipool_spatial_scale() -> None:
        node = onnx.helper.make_node(
            "MaxRoiPool",
            inputs=["X", "rois"],
            outputs=["Y"],
            pooled_shape=[2, 2],
            spatial_scale=0.5,
        )

        X = np.arange(36, dtype=np.float32).reshape(1, 1, 6, 6)
        # ROI coordinates are scaled by 0.5 to [1, 1, 5, 5] after rounding.
        rois = np.array([[0.0, 1.0, 1.0, 9.0, 9.0]], dtype=np.float32)
        # (num_rois, C, pooled_shape[0], pooled_shape[1])
        Y = np.array([[[[21.0, 23.0], [33.0, 35.0]]]], dtype=np.float32)

        expect(
            node,
            inputs=[X, rois],
            outputs=[Y],
            name="test_maxroipool_spatial_scale",
        )

    @staticmethod
    def export_maxroipool_out_of_bounds() -> None:
        node = onnx.helper.make_node(
            "MaxRoiPool",
            inputs=["X", "rois"],
            outputs=["Y"],
            pooled_shape=[2, 2],
            spatial_scale=1.0,
        )

        X = np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3)
        # The first ROI is partially out of bounds, the second one is fully
        # out of bounds, and the third one tests half away from zero rounding
        # of negative coordinates (-2.5 rounds to -3).
        rois = np.array(
            [
                [0.0, -2.0, -2.0, 1.0, 1.0],
                [0.0, 5.0, 5.0, 7.0, 7.0],
                [0.0, -2.5, 0.0, 0.5, 2.0],
            ],
            dtype=np.float32,
        )
        # (num_rois, C, pooled_shape[0], pooled_shape[1])
        Y = np.array(
            [
                [[[0.0, 0.0], [0.0, 4.0]]],
                [[[0.0, 0.0], [0.0, 0.0]]],
                [[[0.0, 4.0], [0.0, 7.0]]],
            ],
            dtype=np.float32,
        )

        expect(
            node,
            inputs=[X, rois],
            outputs=[Y],
            name="test_maxroipool_out_of_bounds",
        )

    @staticmethod
    def export_maxroipool_float64() -> None:
        node = onnx.helper.make_node(
            "MaxRoiPool",
            inputs=["X", "rois"],
            outputs=["Y"],
            pooled_shape=[2, 2],
            spatial_scale=1.0,
        )

        X = np.array(
            [
                [
                    [
                        [0.0, 1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0, 7.0],
                        [8.0, 9.0, 10.0, 11.0],
                        [12.0, 13.0, 14.0, 15.0],
                    ]
                ]
            ],
            dtype=np.float64,
        )
        rois = np.array([[0.0, 1.0, 1.0, 2.0, 2.0]], dtype=np.float64)
        # (num_rois, C, pooled_shape[0], pooled_shape[1])
        Y = np.array([[[[5.0, 6.0], [9.0, 10.0]]]], dtype=np.float64)

        expect(
            node,
            inputs=[X, rois],
            outputs=[Y],
            name="test_maxroipool_float64",
        )
