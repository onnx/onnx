# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ..op_run import OpRun


def _apply_momentum(r, t, x, g, v, norm_coefficient, alpha, beta):  # type: ignore
    # Add gradient of regularization term.
    g_regularized = norm_coefficient * x + g
    # Coefficient of gradient should be 1 at the first iteration.
    beta_adjusted = beta if t > 0 else 1
    # Update momentum.
    v_new = alpha * v + beta_adjusted * g_regularized
    # Apply SG with momentum update rule.
    x_new = x - r * v_new
    return x_new, v_new


class Momentum(OpRun):
    def _run(self, *data):  # type: ignore
        if len(data) == 5:
            return self._run1(*data)
        n = (len(data) - 2) // 3
        xs = []
        vs = []
        for i in range(0, n):
            a, b = self._run1(  # type: ignore
                *data[:2], data[2 + i], data[2 + n + i], data[2 + n * 2 + i]
            )
            xs.append(a)
            vs.append(b)
        return tuple(xs + vs)

    def _run1(self, r, t, x, g, v):  # type: ignore
        x_new, v_new = _apply_momentum(
            r, t, x, g, v, self.norm_coefficient, self.alpha, self.beta  # type: ignore
        )
        return x_new, v_new
