# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ...defs import onnx_opset_version
from ..op_run import OpRun


def _pad_impl(data, raw_pads, mode, constant_values=0.0, axes=None):  # type: ignore
    if axes is not None:
        old_raw_pads = raw_pads
        raw_pads = []
        pos = 0
        for i in range(len(data.shape)):
            if i in axes:
                raw_pads.extend(old_raw_pads[pos : pos + 2])
                pos += 2
            else:
                raw_pads.extend([0, 0])
        raw_pads = numpy.array(raw_pads)

    input_rank = data.ndim
    if input_rank * 2 != raw_pads.size:
        raise RuntimeError("The number of elements in raw_pads should be 2 * data_rank")

    half = raw_pads.shape[0] // 2
    pad_width = tuple((raw_pads[i], raw_pads[i + half]) for i in range(0, half))

    if mode == "constant":
        return numpy.pad(
            data, pad_width=pad_width, mode=mode, constant_values=constant_values
        )
    return numpy.pad(data, pad_width=pad_width, mode=mode)


class Pad_1(OpRun):
    def _run(self, data, pads, constant_value=None):  # type: ignore
        # TODO: support overridden attributes.
        if constant_value is None:
            constant_value = 0
        return (_pad_impl(data, pads, mode=self.mode, constant_values=constant_value),)  # type: ignore


class Pad_18(OpRun):
    def _run(self, data, pads, constant_value=None, axes=None):  # type: ignore
        # TODO: support overridden attributes.
        if constant_value is None:
            constant_value = 0
        return (
            _pad_impl(
                data, pads, mode=self.mode, constant_values=constant_value, axes=axes  # type: ignore
            ),
        )


if onnx_opset_version() >= 18:
    Pad = Pad_18
else:
    Pad = Pad_1  # type: ignore
