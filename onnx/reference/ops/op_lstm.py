# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun


class CommonLSTM(OpRun):
    def __init__(self, onnx_node, run_params):
        OpRun.__init__(self, onnx_node, run_params)
        self.n_outputs = len(onnx_node.output)
        self.n_gates = 3

    def f(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def g(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def h(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def _run_forward(
        self,
        X: np.ndarray,
        W: np.ndarray,
        R: np.ndarray,
        B: np.ndarray,
        P: np.ndarray,
        H_0: np.ndarray,
        C_0: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run a forward pass of the LSTM.

        Assumes that the num_directions axis has been squeezed out of the
        inputs. (And returns Y, Yh, Yc without it.)
        ``mask`` (sequence_length, batch_size) marks the steps that are within
        ``sequence_lens``. The other steps keep the previous state and output zeros.
        """
        h_list = []

        [p_i, p_o, p_f] = np.split(P, 3)
        H_t = H_0
        C_t = C_0
        for t, x in enumerate(X):
            gates = (
                np.dot(x, np.transpose(W))
                + np.dot(H_t, np.transpose(R))
                + np.add(*np.split(B, 2))
            )
            i, o, f, c = np.split(gates, 4, -1)
            i = self.f(i + p_i * C_t)
            f = self.f(f + p_f * C_t)
            c = self.g(c)
            C = f * C_t + i * c
            o = self.f(o + p_o * C)
            H = o * self.h(C)
            if mask is not None:
                H = np.where(mask[t][:, np.newaxis], H, H_t)
                C = np.where(mask[t][:, np.newaxis], C, C_t)
            h_list.append(H)
            H_t = H
            C_t = C

        Y = np.stack(h_list, axis=0)
        if mask is not None:
            Y = np.where(mask[:, :, np.newaxis], Y, 0)
        Y_h = H_t
        Y_c = C_t
        return Y, Y_h, Y_c

    def _step(
        self,
        X: np.ndarray,
        R: np.ndarray,
        B: np.ndarray,
        W: np.ndarray,
        H_0: np.ndarray,
        C_0: np.ndarray,
        P: np.ndarray,
        num_directions: int,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.direction not in {"forward", "reverse", "bidirectional"}:
            raise RuntimeError(f"Unknown direction {self.direction!r}.")
        expected_num_directions = 2 if self.direction == "bidirectional" else 1
        if num_directions != expected_num_directions:
            raise RuntimeError(
                f"direction={self.direction!r} requires num_directions={expected_num_directions} "
                f"but got {num_directions}."
            )

        mask_reversed = None if mask is None else np.flip(mask, axis=0)
        if self.direction == "forward":
            Y, Y_h, Y_c = self._run_forward(
                X,
                W[0],
                R[0],
                B[0],
                P[0],
                H_0[0],
                C_0[0],
                mask,
            )
            # Add num_directions axis to outputs
            Y = np.expand_dims(Y, 1)
            Y_h = np.expand_dims(Y_h, 0)
            Y_c = np.expand_dims(Y_c, 0)
        elif self.direction == "reverse":
            Y, Y_h, Y_c = self._run_forward(
                np.flip(X, axis=0),
                W[0],
                R[0],
                B[0],
                P[0],
                H_0[0],
                C_0[0],
                mask_reversed,
            )
            Y = np.flip(Y, axis=0)
            Y = np.expand_dims(Y, 1)
            Y_h = np.expand_dims(Y_h, 0)
            Y_c = np.expand_dims(Y_c, 0)
        else:
            Yf, Yf_h, Yf_c = self._run_forward(
                X,
                W[0],
                R[0],
                B[0],
                P[0],
                H_0[0],
                C_0[0],
                mask,
            )
            Yb, Yb_h, Yb_c = self._run_forward(
                np.flip(X, axis=0),
                W[1],
                R[1],
                B[1],
                P[1],
                H_0[1],
                C_0[1],
                mask_reversed,
            )
            Yb = np.flip(Yb, axis=0)
            Y = np.stack([Yf, Yb], axis=1)
            Y_h = np.stack([Yf_h, Yb_h], axis=0)
            Y_c = np.stack([Yf_c, Yb_c], axis=0)

        if self.layout:
            Y = np.transpose(Y, [2, 0, 1, 3])
            Y_h = np.transpose(Y_h, [1, 0, 2])
            Y_c = np.transpose(Y_c, [1, 0, 2])

        return Y, Y_h, Y_c

    def _run(
        self,
        X,
        W,
        R,
        B=None,
        sequence_lens=None,
        initial_h=None,
        initial_c=None,
        P=None,
        activation_alpha=None,  # noqa: ARG002
        activation_beta=None,  # noqa: ARG002
        activations=None,  # noqa: ARG002
        clip=None,  # noqa: ARG002
        direction=None,  # noqa: ARG002
        hidden_size=None,
        input_forget=None,  # noqa: ARG002
        layout=None,  # noqa: ARG002
    ):
        # TODO: support overridden attributes.
        n_gates = 4
        number_of_peepholes = 3

        num_directions = W.shape[0]

        hidden_size = R.shape[-1]

        if self.layout != 0:
            X = np.swapaxes(X, 0, 1)
            if initial_h is not None:
                initial_h = np.swapaxes(initial_h, 0, 1)
            if initial_c is not None:
                initial_c = np.swapaxes(initial_c, 0, 1)

        batch_size = X.shape[1]
        if B is None:
            B = np.zeros((num_directions, 2 * n_gates * hidden_size), dtype=X.dtype)
        if P is None:
            P = np.zeros(
                (num_directions, number_of_peepholes * hidden_size), dtype=X.dtype
            )
        if initial_h is None:
            initial_h = np.zeros(
                (num_directions, batch_size, hidden_size), dtype=X.dtype
            )
        if initial_c is None:
            initial_c = np.zeros(
                (num_directions, batch_size, hidden_size), dtype=X.dtype
            )

        mask = None
        if sequence_lens is not None:
            mask = np.arange(X.shape[0])[:, np.newaxis] < sequence_lens[np.newaxis, :]

        Y, Y_h, Y_c = self._step(
            X,
            R,
            B,
            W,
            initial_h,
            initial_c,
            P,
            num_directions=num_directions,
            mask=mask,
        )
        Y = Y.astype(X.dtype)

        if self.n_outputs == 1:
            return (Y,)
        if self.n_outputs == 2:
            return (Y, Y_h.astype(X.dtype))
        assert self.n_outputs == 3, f"Invalid # outputs: {self.n_outputs}"
        return (Y, Y_h.astype(X.dtype), Y_c.astype(X.dtype))


class LSTM(CommonLSTM):
    def __init__(self, onnx_node, run_params):
        CommonLSTM.__init__(self, onnx_node, run_params)
