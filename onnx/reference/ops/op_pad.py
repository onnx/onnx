# SPDX-License-Identifier: Apache-2.0
# pylint: disable=R0913,W0221

import numpy as np

from onnx.reference.op_run import OpRun


def _pad_impl(data, raw_pads, mode, constant_values=0.0, axes=None):  # type: ignore
    input_rank = data.ndim
    pad_width = np.zeros((input_rank, 2), dtype=np.int64)

    if axes is None:
        axes = list(range(input_rank))
    if len(axes) * 2 != len(raw_pads):
        raise RuntimeError(
            f"The number of elements ({raw_pads.size}) "
            f"in raw_pads should be 2 * len(axes) ({len(axes)})."
        )

    for i in range(len(axes)):  # pylint: disable=consider-using-enumerate
        axis = axes[i]
        pads = [raw_pads[i], raw_pads[i + len(axes)]]
        pad_width[axis, :] = pads

    pad_width = tuple(tuple(row) for row in pad_width)

    if mode == "constant":
        return np.pad(
            data, pad_width=pad_width, mode=mode, constant_values=constant_values
        ).astype(data.dtype)
    return np.pad(data, pad_width=pad_width, mode=mode).astype(data.dtype)


class Pad_1(OpRun):
    def _run(self, data, paddings=None, mode=None, value=None):  # type: ignore
        if value is None:
            value = 0
        return (_pad_impl(data, paddings, mode=mode, constant_values=value),)


class Pad_2(OpRun):
    def _run(self, data, pads=None, mode=None, value=None):  # type: ignore
        if value is None:
            value = 0
        return (_pad_impl(data, pads, mode=mode, constant_values=value),)


class Pad_11(OpRun):
    def _run(self, data, pads, constant_value=None, mode=None):  # type: ignore
        if constant_value is None:
            constant_value = 0
        return (
            _pad_impl(data, pads, mode=mode, constant_values=constant_value, axes=None),
        )


class Pad_18(OpRun):
    def _run(self, data, pads, constant_value=None, axes=None, mode=None):  # type: ignore
        if constant_value is None:
            constant_value = 0
        return (
            _pad_impl(data, pads, mode=mode, constant_values=constant_value, axes=axes),
        )
