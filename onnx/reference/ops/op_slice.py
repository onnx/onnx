# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

from onnx.reference.ops._op import OpRun


def _slice(
    data: Any,
    starts: Any,
    ends: Any,
    axes: Any | None = None,
    steps: Any | None = None,
) -> Any:
    if isinstance(starts, list):
        starts = np.array(starts)
    if isinstance(ends, list):
        ends = np.array(ends)
    if isinstance(axes, list):
        axes = np.array(axes)
    if isinstance(steps, list):
        steps = np.array(steps)
    if len(starts.shape) == 0:
        starts = np.array([starts])
    if len(ends.shape) == 0:
        ends = np.array([ends])
    if axes is None:
        if steps is None:
            slices = [slice(s, e) for s, e in zip(starts, ends, strict=False)]
        else:
            slices = [
                slice(s, e, d) for s, e, d in zip(starts, ends, steps, strict=False)
            ]
    else:  # noqa: PLR5501
        if steps is None:
            slices = [slice(0, a) for a in data.shape]
            for s, e, a in zip(starts, ends, axes, strict=False):
                slices[a] = slice(s, e)
        else:
            slices = [slice(0, a) for a in data.shape]
            for s, e, a, d in zip(starts, ends, axes, steps, strict=False):
                slices[a] = slice(s, e, d)
    try:
        return data[tuple(slices)]
    except TypeError as e:  # pragma: no cover
        raise TypeError(
            f"Unable to extract slice {slices!r} for shape {data.shape!r}."
        ) from e


class SliceCommon(OpRun):
    def _run(self, data, starts, ends, axes=None, steps=None):
        self._get_array_api_namespace(data)
        res = _slice(data, starts, ends, axes, steps)
        return (res,)


class Slice_10(SliceCommon):
    def __init__(self, onnx_node, run_params):
        SliceCommon.__init__(self, onnx_node, run_params)


class Slice_1(SliceCommon):
    def __init__(self, onnx_node, run_params):
        SliceCommon.__init__(self, onnx_node, run_params)
        for f in ["starts", "ends", "steps", "axes"]:
            if not hasattr(self, f):
                continue
            if getattr(self, f) is not None and len(getattr(self, f)) == 0:
                setattr(self, f, None)

    def _run(self, data, axes=None, ends=None, starts=None):
        self._get_array_api_namespace(data)
        return SliceCommon._run(self, data, starts, ends, axes)
