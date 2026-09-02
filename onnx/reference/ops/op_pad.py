# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun

_MIN_REFLECT_AXIS_LENGTH = 2


def _pad_impl(data, raw_pads, mode, constant_values=0.0, axes=None):
    input_rank = data.ndim
    if axes is None:
        axes = list(range(input_rank))
    else:
        axes = [axis if axis >= 0 else axis + input_rank for axis in axes]
    num_axes = len(axes)
    if num_axes * 2 != len(raw_pads):
        raise RuntimeError(
            "The number of elements in raw_pads should be 2 times the number of axes"
        )

    pad_width = [(0, 0)] * input_rank
    for i, axis in enumerate(axes):
        pad_begin = raw_pads[i]
        pad_end = raw_pads[num_axes + i]
        pad_width[axis] = (pad_begin, pad_end)

    output_shape = tuple(
        dim + int(begin) + int(end)
        for dim, (begin, end) in zip(data.shape, pad_width, strict=True)
    )
    if any(dim < 0 for dim in output_shape):
        raise ValueError(
            f"Padding results in a negative output dimension: {output_shape}"
        )

    # Negative pads crop; numpy.pad rejects them, so slice the crop out first.
    overcropped = any(
        max(-int(begin), 0) + max(-int(end), 0) > dim
        for dim, (begin, end) in zip(data.shape, pad_width, strict=True)
    )
    data = data[
        tuple(
            slice(-begin if begin < 0 else None, end if end < 0 else None)
            for begin, end in pad_width
        )
    ]
    pad_width = [(max(begin, 0), max(end, 0)) for begin, end in pad_width]

    if overcropped:
        if mode != "constant":
            raise ValueError(f"Mode {mode!r} cannot pad an overcropped input")
        return np.full(output_shape, constant_values, dtype=data.dtype)

    if mode == "reflect":
        for axis, (begin, end) in enumerate(pad_width):
            if (begin or end) and data.shape[axis] < _MIN_REFLECT_AXIS_LENGTH:
                raise ValueError(
                    "Reflect padding requires an axis length of at least 2 "
                    f"after cropping, but axis {axis} has length {data.shape[axis]}"
                )
            max_pad = data.shape[axis] - 1
            if begin > max_pad or end > max_pad:
                raise ValueError(
                    "Reflect padding cannot exceed the cropped axis length minus 1: "
                    f"axis {axis} has length {data.shape[axis]} and pads {(begin, end)}"
                )

    if mode == "constant":
        return np.pad(
            data, pad_width=pad_width, mode=mode, constant_values=constant_values
        ).astype(data.dtype)
    return np.pad(data, pad_width=pad_width, mode=mode).astype(data.dtype)


class Pad_1(OpRun):
    def _run(self, data, paddings=None, mode=None, value=None):
        if value is None:
            value = 0
        return (_pad_impl(data, paddings, mode=mode, constant_values=value),)


class Pad_2(OpRun):
    def _run(self, data, pads=None, mode=None, value=None):
        if value is None:
            value = 0
        return (_pad_impl(data, pads, mode=mode, constant_values=value),)


class Pad_11(OpRun):
    def _run(self, data, pads, constant_value=None, mode=None):
        if constant_value is None:
            constant_value = 0
        return (
            _pad_impl(data, pads, mode=mode, constant_values=constant_value, axes=None),
        )


class Pad_18(OpRun):
    def _run(self, data, pads, constant_value=None, axes=None, mode=None):
        if constant_value is None:
            constant_value = 0
        return (
            _pad_impl(data, pads, mode=mode, constant_values=constant_value, axes=axes),
        )
