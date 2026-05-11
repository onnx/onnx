# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun


class CommonGRU(OpRun):
    def __init__(self, onnx_node, run_params):
        OpRun.__init__(self, onnx_node, run_params)
        self.n_outputs = len(onnx_node.output)
        self.number_of_gates = 3

    def f(self, x):
        return 1 / (1 + np.exp(-x))

    def g(self, x):
        return np.tanh(x)

    def _run_forward(self, X, R, B, W, H_0):
        """Run a single forward pass of the GRU.

        Assumes that the num_directions axis has been squeezed out of the
        inputs. (And returns Y, Yh without it.)
        """
        h_list = []

        [w_z, w_r, w_h] = np.split(W, 3)
        [r_z, r_r, r_h] = np.split(R, 3)
        [w_bz, w_br, w_bh, r_bz, r_br, r_bh] = np.split(B, 6)

        gates_w = np.transpose(np.concatenate((w_z, w_r)))
        gates_r = np.transpose(np.concatenate((r_z, r_r)))
        gates_b = np.add(np.concatenate((w_bz, w_br)), np.concatenate((r_bz, r_br)))

        H_t = H_0
        for x in X:
            gates = np.dot(x, gates_w) + np.dot(H_t, gates_r) + gates_b
            z, r = np.split(gates, 2, -1)
            z = self.f(z)
            r = self.f(r)
            h_default = self.g(
                np.dot(x, np.transpose(w_h))
                + np.dot(r * H_t, np.transpose(r_h))
                + w_bh
                + r_bh
            )
            h_linear = self.g(
                np.dot(x, np.transpose(w_h))
                + r * (np.dot(H_t, np.transpose(r_h)) + r_bh)
                + w_bh
            )
            h = h_linear if self.linear_before_reset else h_default
            H = (1 - z) * h + z * H_t
            h_list.append(H)
            H_t = H

        Y = np.stack(h_list, axis=0)
        Y_h = H_t
        return Y, Y_h

    def _step(self, X, R, B, W, H_0, num_directions):
        if self.direction == "forward":
            assert num_directions == 1
            Y, Y_h = self._run_forward(
                X,
                R[0],
                B[0],
                W[0],
                H_0[0],
            )
            Y = np.expand_dims(Y, 1)
            Y_h = np.expand_dims(Y_h, 0)
        elif self.direction == "reverse":
            assert num_directions == 1
            Y, Y_h = self._run_forward(
                np.flip(X, axis=0),
                R[0],
                B[0],
                W[0],
                H_0[0],
            )
            Y = np.flip(Y, axis=0)
            Y = np.expand_dims(Y, 1)
            Y_h = np.expand_dims(Y_h, 0)
        else:
            assert self.direction == "bidirectional"
            assert num_directions == 2
            Yf, Yf_h = self._run_forward(
                X,
                R[0],
                B[0],
                W[0],
                H_0[0],
            )
            Yb, Yb_h = self._run_forward(
                np.flip(X, axis=0),
                R[1],
                B[1],
                W[1],
                H_0[1],
            )
            Yb = np.flip(Yb, axis=0)
            Y = np.stack([Yf, Yb], axis=1)
            Y_h = np.stack([Yf_h, Yb_h], axis=0)

        if self.layout:
            Y = np.transpose(Y, [2, 0, 1, 3])
            Y_h = np.transpose(Y_h, [1, 0, 2])

        return Y, Y_h

    def _run(
        self,
        X,
        W,
        R,
        B=None,
        sequence_lens=None,
        initial_h=None,
        activation_alpha=None,  # noqa: ARG002
        activation_beta=None,  # noqa: ARG002
        activations=None,  # noqa: ARG002
        clip=None,  # noqa: ARG002
        direction=None,  # noqa: ARG002
        hidden_size=None,
        layout=None,
        linear_before_reset=None,  # noqa: ARG002
    ):
        # TODO: support overridden attributes.
        num_directions = W.shape[0]

        hidden_size = R.shape[-1]

        X = X if layout == 0 else np.swapaxes(X, 0, 1)
        batch_size = X.shape[1]
        b = (
            B
            if B is not None
            else np.zeros((num_directions, 2 * self.number_of_gates * hidden_size), dtype=X.dtype)
        )
        h_0 = (
            initial_h
            if initial_h is not None
            else np.zeros((num_directions, batch_size, hidden_size), dtype=X.dtype)
        )

        B = b
        H_0 = h_0

        Y, Y_h = self._step(X, R, B, W, H_0, num_directions=num_directions)
        Y = Y.astype(X.dtype)
        return (Y,) if self.n_outputs == 1 else (Y, Y_h.astype(X.dtype))


class GRU(CommonGRU):
    def __init__(self, onnx_node, run_params):
        CommonGRU.__init__(self, onnx_node, run_params)
