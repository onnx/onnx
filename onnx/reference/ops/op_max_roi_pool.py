# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math

import numpy as np

from onnx.reference.op_run import OpRun

_HALF = 0.5


def _round_half_away_from_zero(value: float) -> int:
    """Round as C++ std::round does, half away from zero.

    Python round() and numpy.round() implement banker's rounding, so they
    cannot be used here, and floor(value + 0.5) is not equivalent for
    negative coordinates.
    """
    magnitude = abs(float(value))
    integer_part = math.floor(magnitude)
    rounded = integer_part + int(magnitude - integer_part >= _HALF)
    return -rounded if value < 0 else rounded


class MaxRoiPool(OpRun):
    def _run(self, X, rois, pooled_shape=None, spatial_scale=None):
        if pooled_shape is None:
            pooled_shape = self.pooled_shape
        if spatial_scale is None:
            spatial_scale = self.spatial_scale

        num_rois = rois.shape[0]
        channels = X.shape[1]
        height, width = X.shape[2], X.shape[3]
        pooled_height, pooled_width = pooled_shape
        calculation_dtype = np.float64 if X.dtype == np.float64 else np.float32
        spatial_scale = calculation_dtype(spatial_scale)

        Y = np.empty((num_rois, channels, pooled_height, pooled_width), dtype=X.dtype)
        for n in range(num_rois):
            roi = rois[n].astype(calculation_dtype)
            roi_batch_ind = int(roi[0])
            if not 0 <= roi_batch_ind < X.shape[0]:
                raise ValueError(
                    f"ROI batch index {roi_batch_ind} is out of range for batch size {X.shape[0]}"
                )
            roi_start_w = _round_half_away_from_zero(roi[1] * spatial_scale)
            roi_start_h = _round_half_away_from_zero(roi[2] * spatial_scale)
            roi_end_w = _round_half_away_from_zero(roi[3] * spatial_scale)
            roi_end_h = _round_half_away_from_zero(roi[4] * spatial_scale)

            # Force malformed ROIs to be non-empty.
            roi_width = max(roi_end_w - roi_start_w + 1, 1)
            roi_height = max(roi_end_h - roi_start_h + 1, 1)

            bin_size_h = calculation_dtype(roi_height) / calculation_dtype(
                pooled_height
            )
            bin_size_w = calculation_dtype(roi_width) / calculation_dtype(pooled_width)

            # Bin bounds do not depend on the channel, so compute them once per
            # ROI; the ROI offset is added in float64, which rounds only above
            # 2**53. The window max stays per bin, since the windows have
            # different shapes, but it covers all channels in one reduction.
            edges_h = np.arange(pooled_height + 1, dtype=calculation_dtype)
            edges_w = np.arange(pooled_width + 1, dtype=calculation_dtype)
            hstart = np.clip(
                np.floor(edges_h[:-1] * bin_size_h).astype(np.float64) + roi_start_h,
                0,
                height,
            ).astype(np.int64)
            hend = np.clip(
                np.ceil(edges_h[1:] * bin_size_h).astype(np.float64) + roi_start_h,
                0,
                height,
            ).astype(np.int64)
            wstart = np.clip(
                np.floor(edges_w[:-1] * bin_size_w).astype(np.float64) + roi_start_w,
                0,
                width,
            ).astype(np.int64)
            wend = np.clip(
                np.ceil(edges_w[1:] * bin_size_w).astype(np.float64) + roi_start_w,
                0,
                width,
            ).astype(np.int64)

            for ph in range(pooled_height):
                for pw in range(pooled_width):
                    if hend[ph] <= hstart[ph] or wend[pw] <= wstart[pw]:
                        Y[n, :, ph, pw] = 0
                    else:
                        Y[n, :, ph, pw] = X[
                            roi_batch_ind,
                            :,
                            hstart[ph] : hend[ph],
                            wstart[pw] : wend[pw],
                        ].max(axis=(1, 2))
        return (Y.astype(X.dtype),)
