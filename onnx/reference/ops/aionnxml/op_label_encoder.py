# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops.aionnxml._op_run_aionnxml import OpRunAiOnnxMl


class LabelEncoder(OpRunAiOnnxMl):
    def _run(
        self,
        x,
        default_float=None,
        default_int64=None,
        default_string=None,
        default_tensor=None,
        keys_floats=None,
        keys_int64s=None,
        keys_strings=None,
        values_floats=None,
        values_int64s=None,
        values_strings=None,
        keys_tensor=None,
        values_tensor=None,
    ):
        keys = keys_floats or keys_int64s or keys_strings or keys_tensor
        values = values_floats or values_int64s or values_strings or values_tensor
        classes = {}
        has_nan_key = False
        nan_key_value = None
        for key, value in zip(keys, values, strict=False):
            if isinstance(key, (float, np.floating)) and np.isnan(key):
                # A NaN key matches any NaN input, the last one takes precedence.
                has_nan_key = True
                nan_key_value = value
            else:
                classes[key] = value

        if values is values_tensor:
            if default_tensor is None:
                # Defaults defined by the specification when default_tensor is unset.
                if values_tensor.dtype.kind in "iu":
                    defval = -1
                elif values_tensor.dtype.kind == "f":
                    defval = -0.0
                else:
                    defval = "_Unused"
                otype = values_tensor.dtype
            else:
                defval = default_tensor.item()
                otype = default_tensor.dtype
        elif values is values_floats:
            defval = default_float
            otype = np.float32
        elif values is values_int64s:
            defval = default_int64
            otype = np.int64
        elif values is values_strings:
            defval = default_string
            otype = np.str_
            if not isinstance(defval, str):
                defval = ""

        def lookup(x):
            if has_nan_key and isinstance(x, (float, np.floating)) and np.isnan(x):
                return nan_key_value
            return classes.get(x, defval)

        lookup_func = np.vectorize(lookup, otypes=[otype])
        output = lookup_func(x)
        if output.dtype == object:
            output = output.astype(np.str_)
        return (output,)
