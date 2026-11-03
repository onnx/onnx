# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops.aionnxml._op_run_aionnxml import OpRunAiOnnxMl


def _float_bits(value) -> int:
    return int(np.float32(value).view(np.uint32))


class _LabelEncoder(OpRunAiOnnxMl):
    # LabelEncoder-4: a NaN key matches any NaN input. LabelEncoder-2 compares
    # keys bit-wise, so a NaN key only matches a NaN input with the same bits.
    nan_key_matches_any_nan = True

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
        # NaN keys never compare equal in a dict, so they are looked up by bit
        # pattern. If a key is repeated, the last one takes precedence.
        nan_keys = {}
        last_nan_value = None
        for key, value in zip(keys, values, strict=False):
            if isinstance(key, (float, np.floating)) and np.isnan(key):
                nan_keys[_float_bits(key)] = value
                last_nan_value = value
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
            if nan_keys and isinstance(x, (float, np.floating)) and np.isnan(x):
                if self.nan_key_matches_any_nan:
                    return last_nan_value
                return nan_keys.get(_float_bits(x), defval)
            return classes.get(x, defval)

        lookup_func = np.vectorize(lookup, otypes=[otype])
        output = lookup_func(x)
        if output.dtype == object:
            output = output.astype(np.str_)
        return (output,)


class LabelEncoder_2(_LabelEncoder):
    # Opsets 2 and 3: LabelEncoder-2 specifies bit-wise key comparison.
    nan_key_matches_any_nan = False


class LabelEncoder_4(_LabelEncoder):
    nan_key_matches_any_nan = True


LabelEncoder = LabelEncoder_4
