# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

from typing import Any, List

import numpy as np  # type: ignore

from ..op_run import OpRun


class SequenceMap(OpRun):
    def _run(self, input_sequence, *additional_inputs):  # type: ignore
        body = self.body  # type: ignore
        if len(additional_inputs) == 1 and isinstance(additional_inputs[0], list):
            res = []
            for obj1, obj2 in zip(input_sequence, additional_inputs[0]):
                feeds = {body.input_names[0]: obj1, body.input_names[1]: obj2}
                r = body.run(None, feeds)
                res.extend(r)
            return (res,)
        else:
            feeds = dict(zip(body.input_names[1:], additional_inputs))
            res = []
            for obj in input_sequence:
                feeds[body.input_names[0]] = obj
                r = body.run(None, feeds)
                res.extend(r)
            return (res,)
