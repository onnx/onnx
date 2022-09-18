# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ._op_window_common import _CommonWindow


class HannWindow(_CommonWindow):
    """
    Returns
    :math:`\\omega_n = \\sin^2\\left( \\frac{\\pi n}{N-1} \\right)`
    where *N* is the window length.
    See `hann_window
    <https://pytorch.org/docs/stable/generated/torch.hann_window.html>`_
    """

    def _run(self, size):  # type: ignore
        # TODO: support overridden attributes.
        ni, N_1 = self._begin(size)
        res = numpy.sin(ni * 3.1415 / N_1) ** 2
        return self._end(size, res)
