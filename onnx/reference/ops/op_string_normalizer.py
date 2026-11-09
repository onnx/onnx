# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import locale as pylocale
import warnings

import numpy as np

from onnx.reference.op_run import OpRun, RuntimeTypeError


class StringNormalizer(OpRun):
    """Filters complete string elements before changing their case.

    Locale selection retains the process-wide behavior of the reference
    implementation and is not thread-safe.
    """

    def _run(
        self,
        x,
        case_change_action=None,
        is_case_sensitive=None,
        locale=None,
        stopwords=None,
    ):
        if x.ndim != 1 and (x.ndim != 2 or x.shape[0] != 1):
            raise RuntimeTypeError("x must have shape [C] or [1, C].")
        if case_change_action not in {"NONE", "LOWER", "UPPER"}:
            raise RuntimeError(
                f"Unknown option for case_change_action: {case_change_action!r}."
            )
        if pylocale.getlocale() != locale:
            try:
                pylocale.setlocale(pylocale.LC_ALL, locale)
            except pylocale.Error as e:
                warnings.warn(
                    f"Unknown local setting {locale!r} (current: {pylocale.getlocale()!r}) - {e!r}.",
                    stacklevel=1,
                )

        stops = set(stopwords or ())
        if is_case_sensitive:
            retained = [value for value in x.flat if value not in stops]
        else:
            stops = {value.lower() for value in stops}
            retained = [value for value in x.flat if value.lower() not in stops]

        if case_change_action == "LOWER":
            retained = [value.lower() for value in retained]
        elif case_change_action == "UPPER":
            retained = [value.upper() for value in retained]

        # Empty strings are valid retained elements, not removal markers.
        # Object storage also permits case conversion to expand Unicode strings.
        result = np.array(retained or [""], dtype=object)
        if x.ndim == 2:
            result = result.reshape(1, -1)
        return (result,)
