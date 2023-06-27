# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
# pylint: disable=R0912,R0913,W0221

import numpy as np

from onnx.reference.op_run import OpRun

_acceptable_str_dtypes = ("U", "O")


def _convert_to_nested_list(array: np.ndarray) -> list:
    if array.ndim == 1:
        return [np.array(splits, dtype=object) for splits in array.tolist()]
    else:
        return [_convert_to_nested_list(subarray) for subarray in array]


class StringSplit(OpRun):
    def _run(self, x, delimiter=None, maxsplit=None):
        if delimiter is None:
            delimiter = " "

        if (
            x.dtype.kind not in _acceptable_str_dtypes
        ):
            raise TypeError(
                f"Inputs must be string tensors, received dtype {x.dtype}"
            )
        # we want to return a (potentially nested) list of string arrays, preserving input shape
        split_result = np.char.split(x.astype(np.str_), sep=delimiter, maxsplit=maxsplit)
        split_result = _convert_to_nested_list(split_result)
        return (split_result,)
