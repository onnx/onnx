# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
# pylint: disable=R0912,R0913,W0221

import numpy as np

from onnx.reference.op_run import OpRun

_acceptable_str_dtypes = ("U", "O")


class RegexFullMatch(OpRun):
    try:
        import re2
    except ImportError:
        raise ImportError(
            "re2 must be installed to use the RegexFullMatch operator reference implementation"
        )

    def _run(self, x, pattern=None):
        # As per onnx/mapping.py, object numpy dtype corresponds to TensorProto.STRING
        if x.dtype.kind not in _acceptable_str_dtypes:
            raise TypeError(f"Input must be string tensor, received dtype {x.dtype}")
        regex = self.re2.compile(pattern)
        fullmatch_func = np.vectorize(
            lambda x: regex.fullmatch(x) is not None, otypes=[np.bool_]
        )
        return (fullmatch_func(x),)
