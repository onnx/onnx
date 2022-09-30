# SPDX-License-Identifier: Apache-2.0
# pylint: disable=R0913,W0221

from ._op_run_training import OpRunTraining


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


class Momentum(OpRunTraining):
    def _run(self, *data, alpha=None, beta=None, mode=None, norm_coefficient=None):  # type: ignore
        if mode != "standard":
            raise NotImplementedError(f"Momentum not implemented for mode={mode!r}.")
        if len(data) == 5:
            return self._run1(
                *data, norm_coefficient=norm_coefficient, alpha=alpha, beta=beta
            )
        n = (len(data) - 2) // 3
        xs = []
        vs = []
        for i in range(0, n):
            a, b = self._run1(  # type: ignore
                *data[:2],
                data[2 + i],
                data[2 + n + i],
                data[2 + n * 2 + i],
                norm_coefficient=norm_coefficient,
                alpha=alpha,
                beta=beta,
            )
            xs.append(a)
            vs.append(b)
        return tuple(xs + vs)

    def _run1(self, r, t, x, g, v, norm_coefficient=None, alpha=None, beta=None):  # type: ignore
        x_new, v_new = _apply_momentum(r, t, x, g, v, norm_coefficient, alpha, beta)
        return x_new, v_new
