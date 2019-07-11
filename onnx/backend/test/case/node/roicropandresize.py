from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect

def compute_roi_crop_and_resize(x, rois, batch_indices, crop_size, mode, extrapolation_value):
    num_rois = rois.shape[0]
    channels = x.shape[1]
    height = x.shape[2]
    width = x.shape[3]
    y = np.ones((num_rois, channels, crop_size[0], crop_size[1]), dtype=np.float32) * extrapolation_value
    for roi_ in range(num_rois):
        x1 = rois[roi_][1]
        y1 = rois[roi_][0]
        x2 = rois[roi_][3]
        y2 = rois[roi_][2]
        height_scale = (y2 - y1) * (height - 1) / (crop_size[0] - 1) if crop_size[0] > 1 else 0
        width_scale = (x2 - x1) * (width - 1) / (crop_size[1] - 1) if crop_size[1] > 1 else 0
        batch_idx = np.round(batch_indices[roi_])

        for ch_ in range(crop_size[0]):
            in_y = y1 * (height - 1) + ch_ * height_scale if crop_size[0] > 1 else 0.5 * (y1 + y2) * (height - 1)
            if ch_ == crop_size[0] - 1:
                in_y = y2 * (height - 1) if crop_size[0] > 1 else 0.5 * (y1 + y2) * (height - 1)
            if ch_ == 0:
                in_y = y1 * (height - 1) if crop_size[0] > 1 else 0.5 * (y1 + y2) * (height - 1)
            if in_y < 0 or in_y > height - 1:
                continue
            top_y = np.floor(in_y).astype(np.int32)
            bottom_y = np.ceil(in_y).astype(np.int32)
            y_lerp = in_y - top_y

            for cw_ in range(crop_size[1]):
                in_x = x1 * (width - 1) + cw_ * width_scale if crop_size[1] > 1 else 0.5 * (x1 + x2) * (width - 1)
                if cw_ == crop_size[0] - 1:
                    in_y = x2 * (width - 1) if crop_size[1] > 1 else 0.5 * (x1 + x2) * (width - 1)
                if cw_ == 0:
                    in_y = x1 * (width - 1) if crop_size[1] > 1 else 0.5 * (x1 + x2) * (width - 1)
                if in_x < 0 or in_x > width - 1:
                    continue

                if mode == 'bilinear':
                    left_x = np.floor(in_x).astype(np.int32)
                    right_x = np.ceil(in_x).astype(np.int32)
                    x_lerp = in_x - left_x
                    for c_ in range(channels):
                        top_left = x[batch_idx][c_][top_y][left_x]
                        top_right = x[batch_idx][c_][top_y][right_x]
                        bottom_left = x[batch_idx][c_][bottom_y][left_x]
                        bottom_right = x[batch_idx][c_][bottom_y][right_x]
                        top = top_left + (top_right - top_left) * x_lerp
                        bottom = bottom_left + (bottom_right - bottom_left) * x_lerp
                        y[roi_][c_][ch_][cw_] = top + (bottom - top) * y_lerp
                else:
                    closest_y = np.round(in_y).astype(np.int32)
                    closest_x = np.round(in_x).astype(np.int32)
                    for c_ in range(channels):
                        y[roi_][c_][ch_][cw_] = x[batch_idx][c_][closest_y][closest_x]
    return y

class RoiCropAndResize(Base):

    @staticmethod
    def export_roi_crop_and_resize_0():  # type: () -> None
        extrapolation_value = 0.0
        node = onnx.helper.make_node(
            "RoiCropAndResize",
            inputs=["X", "rois", "batch_indices", "crop_size"],
            outputs=["Y"],
            extrapolation_value=extrapolation_value,
        )

        X = np.array(
            [[[[1.1, 2.2], [3.3, 4.4],], [[5.5, 6.6], [7.7, 8.8],],],],
            dtype=np.float32,
        )
        batch_indices = np.array([0, 0, 0], dtype=np.int64)
        rois = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 0.5, 1.0]], dtype=np.float32)
        crop_size = np.array([2, 2], dtype=np.int64)
        mode = 'bilinear'
        Y = compute_roi_crop_and_resize(X, rois, batch_indices, crop_size, mode, extrapolation_value)
        expect(node, inputs=[X, rois, batch_indices, crop_size], outputs=[Y], name="test_roi_crop_and_resize_0")

    @staticmethod
    def export_roi_crop_and_resize_extrapolation_value():  # type: () -> None
        extrapolation_value = 1.0
        node = onnx.helper.make_node(
            "RoiCropAndResize",
            inputs=["X", "rois", "batch_indices", "crop_size"],
            outputs=["Y"],
            extrapolation_value=extrapolation_value,
        )

        X = np.array(
            [[[[1.1, 2.2], [3.3, 4.4],], [[5.5, 6.6], [7.7, 8.8],],],],
            dtype=np.float32,
        )
        batch_indices = np.array([0, 0, 0], dtype=np.int64)
        rois = np.array([[0.0, 0.0, 3.0, 3.0], [0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 0.5, 1.0]], dtype=np.float32)
        crop_size = np.array([2, 2], dtype=np.int64)
        mode = 'bilinear'
        Y = compute_roi_crop_and_resize(X, rois, batch_indices, crop_size, mode, extrapolation_value)
        expect(node, inputs=[X, rois, batch_indices, crop_size], outputs=[Y], name="test_roi_crop_and_resize_extrapolation_value")

    @staticmethod
    def export_roi_crop_and_resize_nearest():  # type: () -> None
        extrapolation_value = 0.0
        node = onnx.helper.make_node(
            "RoiCropAndResize",
            inputs=["X", "rois", "batch_indices", "crop_size"],
            outputs=["Y"],
            extrapolation_value=extrapolation_value,
        )

        X = np.array(
            [[[[1.1, 2.2], [3.3, 4.4],], [[5.5, 6.6], [7.7, 8.8],],],],
            dtype=np.float32,
        )
        batch_indices = np.array([0, 0, 0], dtype=np.int64)
        rois = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 0.5, 1.0]], dtype=np.float32)
        crop_size = np.array([2, 2], dtype=np.int64)
        mode = 'nearest'
        Y = compute_roi_crop_and_resize(X, rois, batch_indices, crop_size, mode, extrapolation_value)
        expect(node, inputs=[X, rois, batch_indices, crop_size], outputs=[Y], name="test_roi_crop_and_resize_nearest")

    @staticmethod
    def export_roi_crop_and_resize_crop_size():  # type: () -> None
        extrapolation_value = 0.0
        node = onnx.helper.make_node(
            "RoiCropAndResize",
            inputs=["X", "rois", "batch_indices", "crop_size"],
            outputs=["Y"],
            extrapolation_value=extrapolation_value,
        )

        X = np.array(
            [[[[1.1, 2.2], [3.3, 4.4],], [[5.5, 6.6], [7.7, 8.8],],],],
            dtype=np.float32,
        )
        batch_indices = np.array([0, 0, 0], dtype=np.int64)
        rois = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 0.5, 1.0]], dtype=np.float32)
        crop_size = np.array([1, 2], dtype=np.int64)
        mode = 'nearest'
        Y = compute_roi_crop_and_resize(X, rois, batch_indices, crop_size, mode, extrapolation_value)
        expect(node, inputs=[X, rois, batch_indices, crop_size], outputs=[Y], name="test_roi_crop_and_resize_crop_size")

if __name__ == '__main__':
    acgan = CropAndResize()
    acgan.export_crop_and_resize_0()