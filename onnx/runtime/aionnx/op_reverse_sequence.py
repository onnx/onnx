# SPDX-License-Identifier: Apache-2.0
# pylint: disable=C0123,R0912,R0913,R0914,W0221

from ..op_run import OpRun


class ReverseSequence(OpRun):
    def _run(self, data, sequence_lens):  # type: ignore
        # TODO: support overridden attributes.
        index = [slice(0, s) for s in data.shape]
        index_data = [slice(0, s) for s in data.shape]
        result = data.copy()
        for i, sl in enumerate(sequence_lens):
            index[self.batch_axis] = i  # type: ignore
            index[self.time_axis] = slice(0, sl)  # type: ignore
            index_data[self.batch_axis] = i  # type: ignore
            index_data[self.time_axis] = slice(sl - 1, None, -1)  # type: ignore
            result[tuple(index)] = data[tuple(index_data)]
        return (result,)
