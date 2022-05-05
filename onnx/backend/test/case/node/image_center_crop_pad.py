# SPDX-License-Identifier: Apache-2.0

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class CenterCropPad(Base):

    @staticmethod
    def export_center_crop_pad_crop_hwc_crop() -> None:
        node = onnx.helper.make_node(
            'CenterCropPad',
            domain='ai.onnx.image',
            inputs=['x', 'shape'],
            outputs=['y'],
        )

        x = np.random.randn(20, 10, 3).astype(np.float32)
        shape = np.array([10, 8], dtype=np.int64)
        y = x[5:15, 1:9, :]

        expect(node, inputs=[x, shape], outputs=[y],
               name='test_image_center_crop_pad_crop_hwc_crop')

    @staticmethod
    def export_center_crop_pad_crop_hwc_crop_uneven() -> None:
        node = onnx.helper.make_node(
            'CenterCropPad',
            domain='ai.onnx.image',
            inputs=['x', 'shape'],
            outputs=['y'],
        )

        x = np.random.randn(20, 10, 3).astype(np.float32)
        shape = np.array([10, 7], dtype=np.int64)
        y = x[5:15, 1:8, :]

        expect(node, inputs=[x, shape], outputs=[y],
               name='test_image_center_crop_pad_crop_hwc_crop')

    @staticmethod
    def export_center_crop_pad_crop_chw_crop() -> None:
        node = onnx.helper.make_node(
            'CenterCropPad',
            domain='ai.onnx.image',
            inputs=['x', 'shape'],
            outputs=['y'],
            channel_first=1,
        )

        x = np.random.randn(3, 20, 10).astype(np.float32)
        shape = np.array([10, 8], dtype=np.int64)
        y = x[:, 5:15, 1:9]

        expect(node, inputs=[x, shape], outputs=[y],
               name='test_image_center_crop_pad_crop_chw_crop')

    @staticmethod
    def export_center_crop_pad_crop_hwc_pad() -> None:
        node = onnx.helper.make_node(
            'CenterCropPad',
            domain='ai.onnx.image',
            inputs=['x', 'shape'],
            outputs=['y'],
        )

        x = np.random.randn(10, 8, 3).astype(np.float32)
        shape = np.array([20, 10], dtype=np.int64)
        y = np.zeros([20, 10, 3], dtype=np.float32)
        y[5:15, 1:9, :] = x

        expect(node, inputs=[x, shape], outputs=[y],
               name='test_image_center_crop_pad_crop_hwc_pad')

    @staticmethod
    def export_center_crop_pad_crop_chw_pad() -> None:
        node = onnx.helper.make_node(
            'CenterCropPad',
            domain='ai.onnx.image',
            inputs=['x', 'shape'],
            outputs=['y'],
            channel_first=1,
        )

        x = np.random.randn(3, 10, 8).astype(np.float32)
        shape = np.array([20, 10], dtype=np.int64)
        y = np.zeros([3, 20, 10], dtype=np.float32)
        y[:, 5:15, 1:9] = x

        expect(node, inputs=[x, shape], outputs=[y],
               name='test_image_center_crop_pad_crop_chw_pad')

    @staticmethod
    def export_center_crop_pad_crop_hwc_crop_and_pad() -> None:
        node = onnx.helper.make_node(
            'CenterCropPad',
            domain='ai.onnx.image',
            inputs=['x', 'shape'],
            outputs=['y'],
        )

        x = np.random.randn(20, 8, 3).astype(np.float32)
        shape = np.array([10, 10], dtype=np.int64)
        y = np.zeros([10, 10, 3], dtype=np.float32)
        y[:, 1:9, :] = x[5:15, :, :]

        expect(node, inputs=[x, shape], outputs=[y],
               name='test_image_center_crop_pad_crop_hwc_crop_and_pad')

    @staticmethod
    def export_center_crop_pad_crop_chw_crop_and_pad() -> None:
        node = onnx.helper.make_node(
            'CenterCropPad',
            domain='ai.onnx.image',
            inputs=['x', 'shape'],
            outputs=['y'],
            channel_first=1,
        )

        x = np.random.randn(3, 20, 8).astype(np.float32)
        shape = np.array([10, 10], dtype=np.int64)
        y = np.zeros([3, 10, 10], dtype=np.float32)
        y[:, :, 1:9] = x[:, 5:15, :]

        expect(node, inputs=[x, shape], outputs=[y],
               name='test_image_center_crop_pad_crop_chw_crop_and_pad')
