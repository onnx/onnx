# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np  # type: ignore

from ._op_window_common import _CommonWindow


class HammingWindow(_CommonWindow):
    """
    Returns
    :math:`\\omega_n = \\alpha - \\beta \\cos \\left( \\frac{\\pi n}{N-1} \\right)`
    where *N* is the window length.
    See `hamming_window
    <https://pytorch.org/docs/stable/generated/torch.hamming_window.html>`_.
    `alpha=0.54, beta=0.46`
    """

    def _run(self, size):  # type: ignore
        # TODO: support overridden attributes.
        ni, N_1 = self._begin(size)
        alpha = 25.0 / 46.0
        beta = 1 - alpha
        res = alpha - np.cos(ni * 3.1415 * 2 / N_1) * beta
        return self._end(size, res)
