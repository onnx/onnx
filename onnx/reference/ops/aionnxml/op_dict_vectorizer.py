# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops.aionnxml._op_run_aionnxml import OpRunAiOnnxMl


class DictVectorizer(OpRunAiOnnxMl):
    def _run(self, x, int64_vocabulary=None, string_vocabulary=None):  # type: ignore
        if isinstance(x, (np.ndarray, list)):
            if int64_vocabulary is None and string_vocabulary is None:
                raise ValueError(
                    "int64_vocabulary or string_vocabulary must be provided."
                )
            if int64_vocabulary is not None:
                dict_labels = {v: i for i, v in enumerate(int64_vocabulary)}
            else:
                assert string_vocabulary is not None
                dict_labels = {v: i for i, v in enumerate(string_vocabulary)}

            values_list = []
            rows_list = []
            cols_list = []
            for i, row in enumerate(x):
                for k, v in row.items():
                    values_list.append(v)
                    rows_list.append(i)
                    cols_list.append(dict_labels[k])
            values = np.array(values_list)
            rows = np.array(rows_list)
            cols = np.array(cols_list)

            res = np.zeros((len(x), len(dict_labels)), dtype=values.dtype)  # type: ignore
            for r, c, v in zip(rows, cols, values):
                res[r, c] = v
            return (res,)

        if isinstance(x, dict):
            keys = int64_vocabulary or string_vocabulary
            assert keys is not None
            result = [x.get(k, 0) for k in keys]
            return (np.array(result),)

        raise TypeError(f"x must be iterable not {type(x)}.")
