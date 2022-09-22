# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np  # type: ignore

from ._op_window_common import _CommonWindow


class BlackmanWindow(_CommonWindow):
    """
    Returns
    :math:`\\omega_n = 0.42 - 0.5 \\cos \\left( \\frac{2\\pi n}{N-1} \\right) +
    0.08 \\cos \\left( \\frac{4\\pi n}{N-1} \\right)`
    where *N* is the window length.
    See `blackman_window
    <https://pytorch.org/docs/stable/generated/torch.blackman_window.html>`_
    """

    def _run(self, size):  # type: ignore
        # TODO: support overridden attributes.
        # ni, N_1 = self._begin(size)
        ni, N_1 = np.arange(size, dtype=self.dtype), size
        if self.periodic == 0:  # type: ignore
            N_1 = N_1 - 1
        alpha = 0.42
        beta = 0.08
        pi = 3.1415
        y = np.cos((ni * (pi * 2)) / N_1) * (-0.5)
        y += np.cos((ni * (pi * 4)) / N_1) * beta
        y += alpha
        return self._end(size, y)
