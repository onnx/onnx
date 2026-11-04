# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun


class CommonRNN(OpRun):
    def __init__(self, onnx_node, run_params):
        OpRun.__init__(self, onnx_node, run_params)

        if self.direction in ("forward", "reverse"):
            self.num_directions = 1
        elif self.direction == "bidirectional":
            self.num_directions = 2
        else:
            raise RuntimeError(f"Unknown direction {self.direction!r}.")

        if (
            self.activation_alpha is not None
            and len(self.activation_alpha) != self.num_directions
        ):
            raise RuntimeError(
                f"activation_alpha must have the same size as num_directions={self.num_directions}."
            )
        if (
            self.activation_beta is not None
            and len(self.activation_beta) != self.num_directions
        ):
            raise RuntimeError(
                f"activation_beta must have the same size as num_directions={self.num_directions}."
            )

        self.f1 = self.choose_act(
            self.activations[0],
            (
                self.activation_alpha[0]
                if self.activation_alpha is not None and len(self.activation_alpha) > 0
                else None
            ),
            (
                self.activation_beta[0]
                if self.activation_beta is not None and len(self.activation_beta) > 0
                else None
            ),
        )
        if len(self.activations) > 1:
            self.f2 = self.choose_act(
                self.activations[1],
                (
                    self.activation_alpha[1]
                    if self.activation_alpha is not None
                    and len(self.activation_alpha) > 1
                    else None
                ),
                (
                    self.activation_beta[1]
                    if self.activation_beta is not None
                    and len(self.activation_beta) > 1
                    else None
                ),
            )
        self.n_outputs = len(onnx_node.output)

    def choose_act(self, name, alpha, beta):
        if name in ("Tanh", "tanh"):
            return self._f_tanh
        if name in ("Affine", "affine"):
            return lambda x: x * alpha + beta
        raise RuntimeError(f"Unknown activation function {name!r}.")

    def _f_tanh(self, x):
        return np.tanh(x)

    def _run_forward(
        self,
        X: np.ndarray,
        R: np.ndarray,
        B: np.ndarray,
        W: np.ndarray,
        H_0: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run a forward pass of the RNN.

        Assumes that the num_directions axis has been squeezed out of the
        inputs. (And returns Y, Yh without it.)
        """
        h_list = []
        H_t = H_0
        for x in X:
            H = self.f1(
                np.dot(x, np.transpose(W))
                + np.dot(H_t, np.transpose(R))
                + np.add(*np.split(B, 2))
            )
            h_list.append(H)
            H_t = H
        output = np.stack(h_list, axis=0)
        return output, h_list[-1]

    def _run(
        self,
        X,
        W,
        R,
        B=None,
        sequence_lens=None,  # noqa: ARG002
        initial_h=None,
        activation_alpha=None,  # noqa: ARG002
        activation_beta=None,  # noqa: ARG002
        activations=None,  # noqa: ARG002
        clip=None,  # noqa: ARG002
        direction=None,  # noqa: ARG002
        hidden_size=None,
        layout=None,
    ):
        # TODO: support overridden attributes.
        self.num_directions = W.shape[0]

        hidden_size = R.shape[-1]

        if layout == 1:
            X = np.swapaxes(X, 0, 1)
            if initial_h is not None:
                initial_h = np.swapaxes(initial_h, 0, 1)

        batch_size = X.shape[1]
        if B is None:
            B = np.zeros((self.num_directions, 2 * hidden_size), dtype=X.dtype)
        H_0 = (
            initial_h
            if initial_h is not None
            else np.zeros((self.num_directions, batch_size, hidden_size), dtype=X.dtype)
        )

        if self.direction not in {"forward", "reverse", "bidirectional"}:
            raise RuntimeError(f"Unknown direction {self.direction!r}.")
        expected_num_directions = 2 if self.direction == "bidirectional" else 1
        if self.num_directions != expected_num_directions:
            raise RuntimeError(
                f"direction={self.direction!r} requires num_directions={expected_num_directions} "
                f"but got {self.num_directions}."
            )

        if self.direction == "forward":
            Y, Y_h = self._run_forward(
                X,
                R[0],
                B[0],
                W[0],
                H_0[0],
            )
            # Singleton num_directions axis
            Y = np.expand_dims(Y, 1)
            Y_h = np.expand_dims(Y_h, 0)
        elif self.direction == "reverse":
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

        if layout == 1:
            Y = np.transpose(Y, [2, 0, 1, 3])
            Y_h = np.transpose(Y_h, [1, 0, 2])

        Y = Y.astype(X.dtype)
        return (Y,) if self.n_outputs == 1 else (Y, Y_h)


class RNN_7(CommonRNN):
    pass


class RNN_14(CommonRNN):
    pass
