# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ...mapping import TENSOR_TYPE_TO_NP_TYPE
from ..op_run import OpRun


class Optional(OpRun):
    def _run(self, x=None):  # type: ignore
        if x is not None and hasattr(self, "type_proto"):
            tp = self.type_proto  # type: ignore
            dt = TENSOR_TYPE_TO_NP_TYPE[tp]
            if dt != x.dtype:
                raise TypeError(
                    f"Input dtype {x.dtype} ({dt}) and parameter type_proto {tp} disagree"
                )
        return ([x],)
