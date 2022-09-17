# SPDX-License-Identifier: Apache-2.0
# pylint: disable=R0913,R0914,W0221

import numpy  # type: ignore

from ._op_run_training import OpRunTraining


def _apply_adam(  # type: ignore
    r, t, x, g, v, h, norm_coefficient, norm_coefficient_post, alpha, beta, epsilon
):  # type: ignore
    # Add gradient of regularization term.
    g_regularized = norm_coefficient * x + g
    # Update momentum.
    v_new = alpha * v + (1 - alpha) * g_regularized
    # Update second-order momentum.
    h_new = beta * h + (1 - beta) * (g_regularized * g_regularized)
    # Compute element-wise square root.
    h_sqrt = numpy.sqrt(h_new) + epsilon
    # Adjust learning rate.
    r_adjusted = None
    if t > 0:
        # Consider bias correction on momentums.
        r_adjusted = r * numpy.sqrt(1 - beta**t) / (1 - alpha**t)
    else:
        # No bias correction on momentums.
        r_adjusted = r
    # Apply Adam update rule.
    x_new = x - r_adjusted * (v_new / h_sqrt)
    # It's possible to apply regularization in the end.
    x_final = (1 - norm_coefficient_post) * x_new
    return x_final, v_new, h_new


class Adam(OpRunTraining):
    def _run(self, *data):  # type: ignore
        if len(data) == 6:
            return self._run1(*data)
        n = (len(data) - 2) // 4
        xs = []
        vs = []
        hs = []
        for i in range(0, n):
            a, b, c = self._run1(  # type: ignore
                *data[:2],
                data[2 + i],
                data[2 + n + i],
                data[2 + n * 2 + i],
                data[2 + n * 3 + i],
            )
            xs.append(a.astype(numpy.float32))
            vs.append(b.astype(numpy.float32))
            hs.append(c.astype(numpy.float32))
        return tuple(xs + vs + hs)

    def _run1(self, r, t, x, g, v, h):  # type: ignore
        x_new, v_new, h_new = _apply_adam(
            r,
            t,
            x,
            g,
            v,
            h,
            self.norm_coefficient,  # type: ignore
            self.norm_coefficient_post,  # type: ignore
            self.alpha,  # type: ignore
            self.beta,  # type: ignore
            self.epsilon,  # type: ignore
        )
        return x_new, v_new, h_new
