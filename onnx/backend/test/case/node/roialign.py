# SPDX-License-Identifier: Apache-2.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class RoiAlign(Base):
    @staticmethod
    def export_roialign():  # type: () -> None
        node = onnx.helper.make_node(
            "RoiAlign",
            inputs=["X", "rois", "batch_indices"],
            outputs=["Y"],
            spatial_scale=1.0,
            output_height=5,
            output_width=5,
            sampling_ratio=2,
        )

        X = np.array(
            [
                [
                    [
                        [
                            0.2764,
                            0.7150,
                            0.1958,
                            0.3416,
                            0.4638,
                            0.0259,
                            0.2963,
                            0.6518,
                            0.4856,
                            0.7250,
                        ],
                        [
                            0.9637,
                            0.0895,
                            0.2919,
                            0.6753,
                            0.0234,
                            0.6132,
                            0.8085,
                            0.5324,
                            0.8992,
                            0.4467,
                        ],
                        [
                            0.3265,
                            0.8479,
                            0.9698,
                            0.2471,
                            0.9336,
                            0.1878,
                            0.4766,
                            0.4308,
                            0.3400,
                            0.2162,
                        ],
                        [
                            0.0206,
                            0.1720,
                            0.2155,
                            0.4394,
                            0.0653,
                            0.3406,
                            0.7724,
                            0.3921,
                            0.2541,
                            0.5799,
                        ],
                        [
                            0.4062,
                            0.2194,
                            0.4473,
                            0.4687,
                            0.7109,
                            0.9327,
                            0.9815,
                            0.6320,
                            0.1728,
                            0.6119,
                        ],
                        [
                            0.3097,
                            0.1283,
                            0.4984,
                            0.5068,
                            0.4279,
                            0.0173,
                            0.4388,
                            0.0430,
                            0.4671,
                            0.7119,
                        ],
                        [
                            0.1011,
                            0.8477,
                            0.4726,
                            0.1777,
                            0.9923,
                            0.4042,
                            0.1869,
                            0.7795,
                            0.9946,
                            0.9689,
                        ],
                        [
                            0.1366,
                            0.3671,
                            0.7011,
                            0.6234,
                            0.9867,
                            0.5585,
                            0.6985,
                            0.5609,
                            0.8788,
                            0.9928,
                        ],
                        [
                            0.5697,
                            0.8511,
                            0.6711,
                            0.9406,
                            0.8751,
                            0.7496,
                            0.1650,
                            0.1049,
                            0.1559,
                            0.2514,
                        ],
                        [
                            0.7012,
                            0.4056,
                            0.7879,
                            0.3461,
                            0.0415,
                            0.2998,
                            0.5094,
                            0.3727,
                            0.5482,
                            0.0502,
                        ],
                    ]
                ]
            ],
            dtype=np.float32,
        )
        batch_indices = np.array([0, 0, 0], dtype=np.int64)
        rois = np.array([[0, 0, 9, 9], [0, 5, 4, 9], [5, 5, 9, 9]], dtype=np.float32)
        # (num_rois, C, output_height, output_width)
        Y = np.array(
            [
                [
                    [
                        [0.5178, 0.3434, 0.3229, 0.4474, 0.6344],
                        [0.4031, 0.5366, 0.4428, 0.4861, 0.4023],
                        [0.2512, 0.4002, 0.5155, 0.6954, 0.3465],
                        [0.3350, 0.4601, 0.5881, 0.3439, 0.6849],
                        [0.4932, 0.7141, 0.8217, 0.4719, 0.4039]
                    ]
                ],
                [
                    [
                        [0.3070, 0.2187, 0.3337, 0.4880, 0.4870],
                        [0.1871, 0.4914, 0.5561, 0.4192, 0.3686],
                        [0.1433, 0.4608, 0.5971, 0.5310, 0.4982],
                        [0.2788, 0.4386, 0.6022, 0.7000, 0.7524],
                        [0.5774, 0.7024, 0.7251, 0.7338, 0.8163]
                    ]
                ],
                [
                    [
                        [0.2393, 0.4075, 0.3379, 0.2525, 0.4743],
                        [0.3671, 0.2702, 0.4105, 0.6419, 0.8308],
                        [0.5556, 0.4543, 0.5564, 0.7502, 0.9300],
                        [0.6626, 0.5617, 0.4813, 0.4954, 0.6663],
                        [0.6636, 0.3721, 0.2056, 0.1928, 0.2478]
                    ]
                ]
            ],
            dtype=np.float32,
        )

        expect(node, inputs=[X, rois, batch_indices], outputs=[Y], name="test_roialign")

    @staticmethod
    def export_roialign_identity_region():  # type: () -> None
        node = onnx.helper.make_node(
            "RoiAlign",
            inputs=["X", "rois", "batch_indices"],
            outputs=["Y"],
            spatial_scale=1.0,
            output_height=2,
            output_width=2,
            sampling_ratio=1,
        )

        X = np.array(
            [
                [
                    [
                        [0, 1, 2, 3],
                        [4, 5, 6, 7],
                        [8, 9, 10, 11],
                        [12, 13, 14, 15],
                        [16, 17, 18, 19],
                    ]
                ]
            ],
            dtype=np.float32,
        )
        batch_indices = np.array([0], dtype=np.int64)
        rois = np.array([[1, 2, 3, 4]], dtype=np.float32)
        # (num_rois, C, output_height, output_width)
        Y = np.array(
            [
                [
                    [
                        [9, 10],
                        [13, 14],
                    ]
                ]
            ],
            dtype=np.float32,
        )

        expect(node, inputs=[X, rois, batch_indices], outputs=[Y], name="test_roialign_identity_region")
