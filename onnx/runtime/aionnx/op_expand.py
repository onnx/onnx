# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np  # type: ignore

from ..op_run import OpRun


def common_reference_implementation(
    data: np.ndarray, shape: np.ndarray
) -> np.ndarray:
    ones = np.ones(shape, dtype=data.dtype)
    return data * ones


class Expand(OpRun):
    def _run(self, data, shape):  # type: ignore
        return (common_reference_implementation(data, shape),)
