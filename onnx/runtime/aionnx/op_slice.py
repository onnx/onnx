# SPDX-License-Identifier: Apache-2.0
# pylint: disable=R0913,W0221

from typing import Optional

import numpy  # type: ignore

from ...defs import onnx_opset_version
from ._op import OpRun


def _slice(
    data: numpy.ndarray,
    starts: numpy.ndarray,
    ends: numpy.ndarray,
    axes: Optional[numpy.ndarray] = None,
    steps: Optional[numpy.ndarray] = None,
) -> numpy.ndarray:
    if len(starts.shape) == 0:
        starts = numpy.array([starts])
    if len(ends.shape) == 0:
        ends = numpy.array([ends])
    if axes is None:
        if steps is None:
            slices = [slice(s, e) for s, e in zip(starts, ends)]
        else:
            slices = [slice(s, e, d) for s, e, d in zip(starts, ends, steps)]
    else:
        if steps is None:
            slices = [slice(0, a) for a in data.shape]
            for s, e, a in zip(starts, ends, axes):
                slices[a] = slice(s, e)
        else:
            slices = [slice(0, a) for a in data.shape]
            for s, e, a, d in zip(starts, ends, axes, steps):
                slices[a] = slice(s, e, d)
    try:
        return data[tuple(slices)]
    except TypeError as e:  # pragma: no cover
        raise TypeError(
            f"Unable to extract slice {slices!r} for shape {data.shape!r}."
        ) from e


class SliceCommon(OpRun):
    def _run(self, data, starts, ends, axes=None, steps=None):  # type: ignore
        res = _slice(data, starts, ends, axes, steps)
        return (res,)


class Slice_10(SliceCommon):
    def __init__(self, onnx_node, run_params):  # type: ignore
        SliceCommon.__init__(self, onnx_node, run_params)


class Slice_1(SliceCommon):
    def __init__(self, onnx_node, run_params):  # type: ignore
        SliceCommon.__init__(self, onnx_node, run_params)
        for f in ["starts", "ends", "steps", "axes"]:
            if not hasattr(self, f):
                continue
            if getattr(self, f) is not None and len(getattr(self, f)) == 0:
                setattr(self, f, None)

    def _run(self, data):  # type: ignore
        return SliceCommon._run(self, data, self.starts, self.ends, self.axes)  # type: ignore


if onnx_opset_version() >= 10:
    Slice = Slice_10
else:
    Slice = Slice_1  # type: ignore
