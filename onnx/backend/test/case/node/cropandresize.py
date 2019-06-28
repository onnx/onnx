from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class CropAndResize(Base):
    @staticmethod
    def export_crop_and_resize():  # type: () -> None
        node = onnx.helper.make_node(
            "CropAndResize",
            inputs=["X", "rois", "batch_indices", "crop_size"],
            outputs=["Y"],
            extrapolation_value=0.0,
        )

        X = np.array(
            [
                [
                    [
                        [1.1, 2.2],
                        [3.3, 4.4],
                    ],
                    [
                        [5.5, 6.6],
                        [7.7, 8.8],
                    ],
                ],
            ],
            dtype=np.float32,
        )
        batch_indices = np.array([0, 0, 0], dtype=np.int64)
        rois = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 0.5, 1.0]], dtype=np.float32)
        crop_size = np.array([2, 2], dtype=np.int64)
        # (num_rois, C, output_height, output_width)
        Y = np.array(
            [
                [
                    [
                        [1.1, 2.2],
                        [3.3, 4.4],
                    ],
                    [
                        [5.5, 6.6],
                        [7.7, 8.8],
                    ],
                ],
                [
                    [
                        [1.1, 1.65],
                        [2.2, 2.75],
                    ],
                    [
                        [5.5, 6.05],
                        [6.6, 7.15],
                    ],
                ],
                [
                    [
                        [1.1, 2.2],
                        [2.2, 3.3],
                    ],
                    [
                        [5.5, 6.6],
                        [6.6, 7.7],
                    ],
                ],
            ],
            dtype=np.float32,
        )

        expect(node, inputs=[X, rois, batch_indices, crop_size], outputs=[Y], name="test_crop_and_resize")
