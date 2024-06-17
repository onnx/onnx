# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class ConvTranspose(Base):
    @staticmethod
    def export() -> None:
        x = np.array(
            [[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]]]  # (1, 1, 3, 3)
        ).astype(np.float32)

        W = np.array(
            [
                [
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],  # (1, 2, 3, 3)
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                ]
            ]
        ).astype(np.float32)

        node = onnx.helper.make_node("ConvTranspose", ["X", "W"], ["Y"])

        y = np.array(
            [
                [
                    [
                        [0.0, 1.0, 3.0, 3.0, 2.0],  # (1, 2, 5, 5)
                        [3.0, 8.0, 15.0, 12.0, 7.0],
                        [9.0, 21.0, 36.0, 27.0, 15.0],
                        [9.0, 20.0, 33.0, 24.0, 13.0],
                        [6.0, 13.0, 21.0, 15.0, 8.0],
                    ],
                    [
                        [0.0, 1.0, 3.0, 3.0, 2.0],
                        [3.0, 8.0, 15.0, 12.0, 7.0],
                        [9.0, 21.0, 36.0, 27.0, 15.0],
                        [9.0, 20.0, 33.0, 24.0, 13.0],
                        [6.0, 13.0, 21.0, 15.0, 8.0],
                    ],
                ]
            ]
        ).astype(np.float32)

        expect(node, inputs=[x, W], outputs=[y], name="test_convtranspose")

    @staticmethod
    def export_convtranspose_1d() -> None:
        x = np.array([[[0.0, 1.0, 2.0]]]).astype(np.float32)  # (1, 1, 3)

        W = np.array([[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]).astype(  # (1, 2, 3)
            np.float32
        )

        node = onnx.helper.make_node("ConvTranspose", ["X", "W"], ["Y"])

        y = np.array(
            [[[0.0, 1.0, 3.0, 3.0, 2.0], [0.0, 1.0, 3.0, 3.0, 2.0]]]  # (1, 2, 5)
        ).astype(np.float32)

        expect(node, inputs=[x, W], outputs=[y], name="test_convtranspose_1d")

    @staticmethod
    def export_convtranspose_3d() -> None:
        x = np.array(
            [
                [
                    [
                        [
                            [0.0, 1.0, 2.0, 3.0, 4.0],  # (1, 1, 3, 4, 5)
                            [5.0, 6.0, 7.0, 8.0, 9.0],
                            [10.0, 11.0, 12.0, 13.0, 14.0],
                            [15.0, 16.0, 17.0, 18.0, 19.0],
                        ],
                        [
                            [20.0, 21.0, 22.0, 23.0, 24.0],
                            [25.0, 26.0, 27.0, 28.0, 29.0],
                            [30.0, 31.0, 32.0, 33.0, 34.0],
                            [35.0, 36.0, 37.0, 38.0, 39.0],
                        ],
                        [
                            [40.0, 41.0, 42.0, 43.0, 44.0],
                            [45.0, 46.0, 47.0, 48.0, 49.0],
                            [50.0, 51.0, 52.0, 53.0, 54.0],
                            [55.0, 56.0, 57.0, 58.0, 59.0],
                        ],
                    ]
                ]
            ]
        ).astype(np.float32)

        W = np.array(
            [
                [
                    [
                        [
                            [1.0, 1.0, 1.0],  # (1, 2, 3, 3, 3)
                            [1.0, 1.0, 1.0],
                            [1.0, 1.0, 1.0],
                        ],
                        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                    ],
                    [
                        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                    ],
                ]
            ]
        ).astype(np.float32)

        node = onnx.helper.make_node("ConvTranspose", ["X", "W"], ["Y"])

        y = np.array(
            [
                [
                    [
                        [
                            [0.0, 1.0, 3.0, 6.0, 9.0, 7.0, 4.0],  # (1, 2, 5, 6, 7)
                            [5.0, 12.0, 21.0, 27.0, 33.0, 24.0, 13.0],
                            [15.0, 33.0, 54.0, 63.0, 72.0, 51.0, 27.0],
                            [30.0, 63.0, 99.0, 108.0, 117.0, 81.0, 42.0],
                            [25.0, 52.0, 81.0, 87.0, 93.0, 64.0, 33.0],
                            [15.0, 31.0, 48.0, 51.0, 54.0, 37.0, 19.0],
                        ],
                        [
                            [20.0, 42.0, 66.0, 72.0, 78.0, 54.0, 28.0],
                            [50.0, 104.0, 162.0, 174.0, 186.0, 128.0, 66.0],
                            [90.0, 186.0, 288.0, 306.0, 324.0, 222.0, 114.0],
                            [120.0, 246.0, 378.0, 396.0, 414.0, 282.0, 144.0],
                            [90.0, 184.0, 282.0, 294.0, 306.0, 208.0, 106.0],
                            [50.0, 102.0, 156.0, 162.0, 168.0, 114.0, 58.0],
                        ],
                        [
                            [60.0, 123.0, 189.0, 198.0, 207.0, 141.0, 72.0],
                            [135.0, 276.0, 423.0, 441.0, 459.0, 312.0, 159.0],
                            [225.0, 459.0, 702.0, 729.0, 756.0, 513.0, 261.0],
                            [270.0, 549.0, 837.0, 864.0, 891.0, 603.0, 306.0],
                            [195.0, 396.0, 603.0, 621.0, 639.0, 432.0, 219.0],
                            [105.0, 213.0, 324.0, 333.0, 342.0, 231.0, 117.0],
                        ],
                        [
                            [60.0, 122.0, 186.0, 192.0, 198.0, 134.0, 68.0],
                            [130.0, 264.0, 402.0, 414.0, 426.0, 288.0, 146.0],
                            [210.0, 426.0, 648.0, 666.0, 684.0, 462.0, 234.0],
                            [240.0, 486.0, 738.0, 756.0, 774.0, 522.0, 264.0],
                            [170.0, 344.0, 522.0, 534.0, 546.0, 368.0, 186.0],
                            [90.0, 182.0, 276.0, 282.0, 288.0, 194.0, 98.0],
                        ],
                        [
                            [40.0, 81.0, 123.0, 126.0, 129.0, 87.0, 44.0],
                            [85.0, 172.0, 261.0, 267.0, 273.0, 184.0, 93.0],
                            [135.0, 273.0, 414.0, 423.0, 432.0, 291.0, 147.0],
                            [150.0, 303.0, 459.0, 468.0, 477.0, 321.0, 162.0],
                            [105.0, 212.0, 321.0, 327.0, 333.0, 224.0, 113.0],
                            [55.0, 111.0, 168.0, 171.0, 174.0, 117.0, 59.0],
                        ],
                    ],
                    [
                        [
                            [0.0, 1.0, 3.0, 6.0, 9.0, 7.0, 4.0],
                            [5.0, 12.0, 21.0, 27.0, 33.0, 24.0, 13.0],
                            [15.0, 33.0, 54.0, 63.0, 72.0, 51.0, 27.0],
                            [30.0, 63.0, 99.0, 108.0, 117.0, 81.0, 42.0],
                            [25.0, 52.0, 81.0, 87.0, 93.0, 64.0, 33.0],
                            [15.0, 31.0, 48.0, 51.0, 54.0, 37.0, 19.0],
                        ],
                        [
                            [20.0, 42.0, 66.0, 72.0, 78.0, 54.0, 28.0],
                            [50.0, 104.0, 162.0, 174.0, 186.0, 128.0, 66.0],
                            [90.0, 186.0, 288.0, 306.0, 324.0, 222.0, 114.0],
                            [120.0, 246.0, 378.0, 396.0, 414.0, 282.0, 144.0],
                            [90.0, 184.0, 282.0, 294.0, 306.0, 208.0, 106.0],
                            [50.0, 102.0, 156.0, 162.0, 168.0, 114.0, 58.0],
                        ],
                        [
                            [60.0, 123.0, 189.0, 198.0, 207.0, 141.0, 72.0],
                            [135.0, 276.0, 423.0, 441.0, 459.0, 312.0, 159.0],
                            [225.0, 459.0, 702.0, 729.0, 756.0, 513.0, 261.0],
                            [270.0, 549.0, 837.0, 864.0, 891.0, 603.0, 306.0],
                            [195.0, 396.0, 603.0, 621.0, 639.0, 432.0, 219.0],
                            [105.0, 213.0, 324.0, 333.0, 342.0, 231.0, 117.0],
                        ],
                        [
                            [60.0, 122.0, 186.0, 192.0, 198.0, 134.0, 68.0],
                            [130.0, 264.0, 402.0, 414.0, 426.0, 288.0, 146.0],
                            [210.0, 426.0, 648.0, 666.0, 684.0, 462.0, 234.0],
                            [240.0, 486.0, 738.0, 756.0, 774.0, 522.0, 264.0],
                            [170.0, 344.0, 522.0, 534.0, 546.0, 368.0, 186.0],
                            [90.0, 182.0, 276.0, 282.0, 288.0, 194.0, 98.0],
                        ],
                        [
                            [40.0, 81.0, 123.0, 126.0, 129.0, 87.0, 44.0],
                            [85.0, 172.0, 261.0, 267.0, 273.0, 184.0, 93.0],
                            [135.0, 273.0, 414.0, 423.0, 432.0, 291.0, 147.0],
                            [150.0, 303.0, 459.0, 468.0, 477.0, 321.0, 162.0],
                            [105.0, 212.0, 321.0, 327.0, 333.0, 224.0, 113.0],
                            [55.0, 111.0, 168.0, 171.0, 174.0, 117.0, 59.0],
                        ],
                    ],
                ]
            ]
        ).astype(np.float32)

        expect(node, inputs=[x, W], outputs=[y], name="test_convtranspose_3d")

    @staticmethod
    def export_convtranspose_attributes() -> None:
        x = np.array(
            [[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]]]  # (1, 1, 3, 3)
        ).astype(np.float32)

        W = np.array(
            [
                [
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],  # (1, 2, 3, 3)
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                ]
            ]
        ).astype(np.float32)

        y = np.array(
            [
                [
                    [
                        [0.0, 0.0, 1.0, 1.0, 3.0, 2.0, 2.0, 0.0],  # (1, 2, 10, 8)
                        [0.0, 0.0, 1.0, 1.0, 3.0, 2.0, 2.0, 0.0],
                        [0.0, 0.0, 1.0, 1.0, 3.0, 2.0, 2.0, 0.0],
                        [3.0, 3.0, 7.0, 4.0, 9.0, 5.0, 5.0, 0.0],
                        [3.0, 3.0, 7.0, 4.0, 9.0, 5.0, 5.0, 0.0],
                        [3.0, 3.0, 7.0, 4.0, 9.0, 5.0, 5.0, 0.0],
                        [6.0, 6.0, 13.0, 7.0, 15.0, 8.0, 8.0, 0.0],
                        [6.0, 6.0, 13.0, 7.0, 15.0, 8.0, 8.0, 0.0],
                        [6.0, 6.0, 13.0, 7.0, 15.0, 8.0, 8.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 1.0, 1.0, 3.0, 2.0, 2.0, 0.0],
                        [0.0, 0.0, 1.0, 1.0, 3.0, 2.0, 2.0, 0.0],
                        [0.0, 0.0, 1.0, 1.0, 3.0, 2.0, 2.0, 0.0],
                        [3.0, 3.0, 7.0, 4.0, 9.0, 5.0, 5.0, 0.0],
                        [3.0, 3.0, 7.0, 4.0, 9.0, 5.0, 5.0, 0.0],
                        [3.0, 3.0, 7.0, 4.0, 9.0, 5.0, 5.0, 0.0],
                        [6.0, 6.0, 13.0, 7.0, 15.0, 8.0, 8.0, 0.0],
                        [6.0, 6.0, 13.0, 7.0, 15.0, 8.0, 8.0, 0.0],
                        [6.0, 6.0, 13.0, 7.0, 15.0, 8.0, 8.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                ]
            ]
        ).astype(np.float32)

        node = onnx.helper.make_node(
            "ConvTranspose", ["X", "W"], ["Y"], strides=[3, 2], output_shape=[10, 8]
        )
        expect(node, inputs=[x, W], outputs=[y], name="test_convtranspose_output_shape")

        node = onnx.helper.make_node(
            "ConvTranspose", ["X", "W"], ["Y"], strides=[3, 2], output_padding=[1, 1]
        )
        expect(node, inputs=[x, W], outputs=[y], name="test_convtranspose_pad")

        node = onnx.helper.make_node(
            "ConvTranspose",
            ["X", "W"],
            ["Y"],
            name="test",
            strides=[3, 2],
            output_shape=[10, 8],
            kernel_shape=[3, 3],
            output_padding=[1, 1],
        )
        expect(node, inputs=[x, W], outputs=[y], name="test_convtranspose_kernel_shape")

    @staticmethod
    def export_convtranspose_pads() -> None:
        x = np.array(
            [[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]]]  # (1, 1, 3, 3)
        ).astype(np.float32)

        W = np.array(
            [
                [
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],  # (1, 2, 3, 3)
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                ]
            ]
        ).astype(np.float32)

        node = onnx.helper.make_node(
            "ConvTranspose", ["X", "W"], ["Y"], strides=[3, 2], pads=[1, 2, 1, 2]
        )

        y = np.array(
            [
                [
                    [
                        [1.0, 1.0, 3.0],  # (1, 2, 7, 3)
                        [1.0, 1.0, 3.0],
                        [7.0, 4.0, 9.0],
                        [7.0, 4.0, 9.0],
                        [7.0, 4.0, 9.0],
                        [13.0, 7.0, 15.0],
                        [13.0, 7.0, 15.0],
                    ],
                    [
                        [1.0, 1.0, 3.0],
                        [1.0, 1.0, 3.0],
                        [7.0, 4.0, 9.0],
                        [7.0, 4.0, 9.0],
                        [7.0, 4.0, 9.0],
                        [13.0, 7.0, 15.0],
                        [13.0, 7.0, 15.0],
                    ],
                ]
            ]
        ).astype(np.float32)

        expect(node, inputs=[x, W], outputs=[y], name="test_convtranspose_pads")

    @staticmethod
    def export_convtranspose_dilations() -> None:
        x = np.array(
            [[[[3.0, 8.0, 1.0], [9.0, 5.0, 7.0], [3.0, 2.0, 6.0]]]]  # (1, 1, 3, 3)
        ).astype(np.float32)
        W = np.array([[[[7.0, 2.0], [1.0, 9.0]]]]).astype(np.float32)  # (1, 1, 2, 2)

        node = onnx.helper.make_node(
            "ConvTranspose", ["X", "W"], ["Y"], dilations=[2, 2]
        )

        y = np.array(
            [
                [
                    [
                        [21.0, 56.0, 13.0, 16.0, 2.0],  # [1, 1, 5, 5]
                        [63.0, 35.0, 67.0, 10.0, 14.0],
                        [24.0, 22.0, 76.0, 76.0, 21.0],
                        [9.0, 5.0, 88.0, 45.0, 63.0],
                        [3.0, 2.0, 33.0, 18.0, 54.0],
                    ]
                ]
            ]
        ).astype(np.float32)

        expect(node, inputs=[x, W], outputs=[y], name="test_convtranspose_dilations")

    @staticmethod
    def export_convtranspose_autopad_same() -> None:
        x = np.array(
            [[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]]]  # (1, 1, 3, 3)
        ).astype(np.float32)

        W = np.array(
            [
                [
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],  # (1, 2, 3, 3)
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                ]
            ]
        ).astype(np.float32)

        node = onnx.helper.make_node(
            "ConvTranspose", ["X", "W"], ["Y"], auto_pad="SAME_UPPER", strides=[2, 2]
        )

        y = np.array(
            [
                [
                    [
                        [0.0, 0.0, 1.0, 1.0, 3.0, 2.0],
                        [0.0, 0.0, 1.0, 1.0, 3.0, 2.0],
                        [3.0, 3.0, 8.0, 5.0, 12.0, 7.0],
                        [3.0, 3.0, 7.0, 4.0, 9.0, 5.0],
                        [9.0, 9.0, 20.0, 11.0, 24.0, 13.0],
                        [6.0, 6.0, 13.0, 7.0, 15.0, 8.0],
                    ],
                    [
                        [0.0, 0.0, 1.0, 1.0, 3.0, 2.0],
                        [0.0, 0.0, 1.0, 1.0, 3.0, 2.0],
                        [3.0, 3.0, 8.0, 5.0, 12.0, 7.0],
                        [3.0, 3.0, 7.0, 4.0, 9.0, 5.0],
                        [9.0, 9.0, 20.0, 11.0, 24.0, 13.0],
                        [6.0, 6.0, 13.0, 7.0, 15.0, 8.0],
                    ],
                ]
            ]
        ).astype(np.float32)

        expect(node, inputs=[x, W], outputs=[y], name="test_convtranspose_autopad_same")

    @staticmethod
    def export_convtranspose_group_2() -> None:
        x = np.array(
            [
                [
                    [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                    [[9.0, 10.0, 11.0], [12.0, 13.0, 14.0], [15.0, 16.0, 17.0]],
                ]
            ]
        ).astype(np.float32)
        W = np.array(
            [
                [
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                ],
                [
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                ],
            ]
        ).astype(np.float32)

        node = onnx.helper.make_node("ConvTranspose", ["X", "W"], ["Y"], group=2)

        y = np.array(
            [
                [
                    [
                        [0.0, 1.0, 3.0, 3.0, 2.0],
                        [3.0, 8.0, 15.0, 12.0, 7.0],
                        [9.0, 21.0, 36.0, 27.0, 15.0],
                        [9.0, 20.0, 33.0, 24.0, 13.0],
                        [6.0, 13.0, 21.0, 15.0, 8.0],
                    ],
                    [
                        [9.0, 19.0, 30.0, 21.0, 11.0],
                        [21.0, 44.0, 69.0, 48.0, 25.0],
                        [36.0, 75.0, 117.0, 81.0, 42.0],
                        [27.0, 56.0, 87.0, 60.0, 31.0],
                        [15.0, 31.0, 48.0, 33.0, 17.0],
                    ],
                ]
            ]
        ).astype(np.float32)

        expect(node, inputs=[x, W], outputs=[y], name="test_convtranspose_group_2")

    @staticmethod
    def export_convtranspose_group_2_image_3() -> None:
        x = np.array(
            [
                [
                    [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                    [[9.0, 10.0, 11.0], [12.0, 13.0, 14.0], [15.0, 16.0, 17.0]],
                ],
                [
                    [[18.0, 19.0, 20.0], [21.0, 22.0, 23.0], [24.0, 25.0, 26.0]],
                    [[9.0, 10.0, 11.0], [12.0, 13.0, 14.0], [15.0, 16.0, 17.0]],
                ],
                [
                    [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                    [[9.0, 10.0, 11.0], [12.0, 13.0, 14.0], [15.0, 16.0, 17.0]],
                ],
            ]
        ).astype(np.float32)
        W = np.array(
            [
                [
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                ],
                [
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                ],
            ]
        ).astype(np.float32)

        node = onnx.helper.make_node("ConvTranspose", ["X", "W"], ["Y"], group=2)

        y = np.array(
            [
                [
                    [
                        [0.0, 1.0, 3.0, 3.0, 2.0],
                        [3.0, 8.0, 15.0, 12.0, 7.0],
                        [9.0, 21.0, 36.0, 27.0, 15.0],
                        [9.0, 20.0, 33.0, 24.0, 13.0],
                        [6.0, 13.0, 21.0, 15.0, 8.0],
                    ],
                    [
                        [9.0, 19.0, 30.0, 21.0, 11.0],
                        [21.0, 44.0, 69.0, 48.0, 25.0],
                        [36.0, 75.0, 117.0, 81.0, 42.0],
                        [27.0, 56.0, 87.0, 60.0, 31.0],
                        [15.0, 31.0, 48.0, 33.0, 17.0],
                    ],
                ],
                [
                    [
                        [18.0, 37.0, 57.0, 39.0, 20.0],
                        [39.0, 80.0, 123.0, 84.0, 43.0],
                        [63.0, 129.0, 198.0, 135.0, 69.0],
                        [45.0, 92.0, 141.0, 96.0, 49.0],
                        [24.0, 49.0, 75.0, 51.0, 26.0],
                    ],
                    [
                        [9.0, 19.0, 30.0, 21.0, 11.0],
                        [21.0, 44.0, 69.0, 48.0, 25.0],
                        [36.0, 75.0, 117.0, 81.0, 42.0],
                        [27.0, 56.0, 87.0, 60.0, 31.0],
                        [15.0, 31.0, 48.0, 33.0, 17.0],
                    ],
                ],
                [
                    [
                        [0.0, 1.0, 3.0, 3.0, 2.0],
                        [3.0, 8.0, 15.0, 12.0, 7.0],
                        [9.0, 21.0, 36.0, 27.0, 15.0],
                        [9.0, 20.0, 33.0, 24.0, 13.0],
                        [6.0, 13.0, 21.0, 15.0, 8.0],
                    ],
                    [
                        [9.0, 19.0, 30.0, 21.0, 11.0],
                        [21.0, 44.0, 69.0, 48.0, 25.0],
                        [36.0, 75.0, 117.0, 81.0, 42.0],
                        [27.0, 56.0, 87.0, 60.0, 31.0],
                        [15.0, 31.0, 48.0, 33.0, 17.0],
                    ],
                ],
            ]
        ).astype(np.float32)

        expect(
            node, inputs=[x, W], outputs=[y], name="test_convtranspose_group_2_image_3"
        )
