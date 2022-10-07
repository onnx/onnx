# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

from typing import Any, List, Optional, Union

import numpy as np  # type: ignore

from ..op_run import OpRun


class SequenceErase(OpRun):
    def _run(self, S, I=None):  # type: ignore
        if I is None:
            I = -1
        else:
            I = int(I)
        S2 = S.copy()
        del S2[I]
        return (S2,)
