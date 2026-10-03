# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math

import numpy as np

from onnx.reference.op_run import OpRun


def _round_half_away_from_zero(value: float) -> int:
    """Round as C++ std::round does, half away from zero.

    Python round() and numpy.round() implement banker's rounding, so they
    cannot be used here, and floor(value + 0.5) is not equivalent for
    negative coordinates.
    """
    return int(math.copysign(math.floor(abs(value) + 0.5), value))


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

        Y = np.empty((num_rois, channels, pooled_height, pooled_width), dtype=X.dtype)
        for n in range(num_rois):
            roi = rois[n].astype(np.float64)
            roi_batch_ind = int(roi[0])
            roi_start_w = _round_half_away_from_zero(roi[1] * spatial_scale)
            roi_start_h = _round_half_away_from_zero(roi[2] * spatial_scale)
            roi_end_w = _round_half_away_from_zero(roi[3] * spatial_scale)
            roi_end_h = _round_half_away_from_zero(roi[4] * spatial_scale)

            # Force malformed ROIs to be non-empty.
            roi_width = max(roi_end_w - roi_start_w + 1, 1)
            roi_height = max(roi_end_h - roi_start_h + 1, 1)

            bin_size_h = roi_height / pooled_height
            bin_size_w = roi_width / pooled_width

            for c in range(channels):
                for ph in range(pooled_height):
                    for pw in range(pooled_width):
                        hstart = math.floor(ph * bin_size_h) + roi_start_h
                        wstart = math.floor(pw * bin_size_w) + roi_start_w
                        hend = math.ceil((ph + 1) * bin_size_h) + roi_start_h
                        wend = math.ceil((pw + 1) * bin_size_w) + roi_start_w

                        hstart = min(max(hstart, 0), height)
                        hend = min(max(hend, 0), height)
                        wstart = min(max(wstart, 0), width)
                        wend = min(max(wend, 0), width)

                        if hend <= hstart or wend <= wstart:
                            Y[n, c, ph, pw] = 0
                        else:
                            Y[n, c, ph, pw] = X[
                                roi_batch_ind, c, hstart:hend, wstart:wend
                            ].max()
        return (Y.astype(X.dtype),)
