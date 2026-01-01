# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference.ops._op_common_window import _CommonWindow


class HannWindow(_CommonWindow):
    r"""Returns :math:`\\omega_n = \\sin^2\\left( \\frac{\\pi n}{N-1} \\right)` where *N* is the window length.

    See `hann_window <https://pytorch.org/docs/stable/generated/torch.hann_window.html>`_
    """

    def _run(self, size, output_datatype=None, periodic=None):
        xp = self._get_array_api_namespace(size)
        ni, N_1 = self._begin(size, periodic, output_datatype)
        res = xp.sin(ni * np.pi / N_1) ** 2
        return self._end(size, res, output_datatype)
