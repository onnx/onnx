# SPDX-License-Identifier: Apache-2.0
# pylint: disable=R0913,R0914,W0221

import numpy as np

from ._op_run_aionnxml import OpRunAiOnnxMl


class DictVectorizer(OpRunAiOnnxMl):
    def _run(self, x, int64_vocabulary=None, string_vocabulary=None):  # type: ignore

        if isinstance(x, (np.ndarray, list)):

            dict_labels = {}
            if int64_vocabulary:
                for i, v in enumerate(int64_vocabulary):
                    dict_labels[v] = i
            else:
                for i, v in enumerate(string_vocabulary):
                    dict_labels[v] = i
            if len(dict_labels) == 0:
                raise RuntimeError(
                    "int64_vocabulary and string_vocabulary cannot be both empty."
                )

            values = []
            rows = []
            cols = []
            for i, row in enumerate(x):
                for k, v in row.items():
                    values.append(v)
                    rows.append(i)
                    cols.append(dict_labels[k])
            values = np.array(values)
            rows = np.array(rows)
            cols = np.array(cols)
            return (
                coo_matrix(
                    (values, (rows, cols)), shape=(len(x), len(dict_labels))
                ).todense(),
            )

        if isinstance(x, dict):
            keys = int64_vocabulary or string_vocabulary
            res = []
            for k in keys:
                res.append(x.get(k, 0))
            return (np.array(res),)

        raise TypeError(f"x must be iterable not {type(x)}.")
