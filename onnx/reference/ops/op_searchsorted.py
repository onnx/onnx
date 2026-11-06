# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt

from onnx.reference.op_run import OpRun


class SearchSorted(OpRun):
    def _run(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        sorter: npt.NDArray[np.int64] | None = None,
        *,
        side: Literal["left", "right"] = "left",
    ):
        out = np.searchsorted(x1, x2, side=side, sorter=sorter)
        return (out.astype(np.int64),)
