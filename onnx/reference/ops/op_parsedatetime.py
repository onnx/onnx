# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221
from datetime import datetime

import numpy as np

from onnx.reference.op_run import OpRun


class ParseDateTime(OpRun):
    def _run(self, x, format, unit, default=None):  # type: ignore
        def parse(el):
            try:
                return datetime.strptime(el, format).timestamp()
            except ValueError:
                return np.nan
        out = np.array([parse(el) for el in x])
        out[np.isnan(out)] = default
        out = out.astype(default.dtype) if default is not None else out.astype(np.int64)
        return (out,)
