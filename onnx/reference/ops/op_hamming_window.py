# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops._op_common_window import _CommonWindow


class HammingWindow_17(_CommonWindow):
    r"""Legacy window with :math:`\alpha=25/46` and :math:`\beta=21/46`.

    The phase is ``2*pi*n/D``, with ``D=N`` for periodic windows and
    ``D=N-1`` for symmetric windows.
    """

    _alpha = 25.0 / 46.0
    _beta = 1 - _alpha

    def _run(self, size, output_datatype=None, periodic=None):
        ni, N_1 = self._begin(size, periodic, output_datatype)
        res = self._alpha - np.cos(ni * np.pi * 2 / N_1) * self._beta
        return self._end(size, res, output_datatype)


class HammingWindow_29(HammingWindow_17):
    """Hamming window from Harris (1978), equation (30b)."""

    _alpha = 0.54
    _beta = 0.46
