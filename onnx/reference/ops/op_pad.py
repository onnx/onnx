# SPDX-License-Identifier: Apache-2.0
# pylint: disable=R0913,W0221

import numpy as np

from onnx.defs import onnx_opset_version
from onnx.reference.op_run import OpRun


def _pad_impl(data, raw_pads, mode, constant_values=0.0, axes=None):  # type: ignore
    if raw_pads is not None:
        old_raw_pads = raw_pads
        raw_pads = []
        pos = 0
        for i in range(len(data.shape)):
            if axes is None or i in axes:
                raw_pads.extend(old_raw_pads[pos : pos + 2])
                pos += 2
            else:
                raw_pads.extend([0, 0])
        raw_pads = np.array(raw_pads)

    input_rank = data.ndim
    if input_rank * 2 != raw_pads.size:
        raise RuntimeError("The number of elements in raw_pads should be 2 * data_rank")

    half = raw_pads.shape[0] // 2
    pad_width = tuple((raw_pads[i], raw_pads[i + half]) for i in range(0, half))

    if mode == "constant":
        return np.pad(
            data, pad_width=pad_width, mode=mode, constant_values=constant_values
        )
    return np.pad(data, pad_width=pad_width, mode=mode)


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


if onnx_opset_version() >= 18:
    Pad = Pad_18
elif onnx_opset_version() >= 11:
    Pad = Pad_11  # type: ignore
elif onnx_opset_version() >= 2:
    Pad = Pad_2  # type: ignore
else:
    Pad = Pad_1  # type: ignore
