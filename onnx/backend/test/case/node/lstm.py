# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class LSTMHelper:
    def __init__(self, **params: Any) -> None:
        # LSTM Input Names
        X = "X"
        W = "W"
        R = "R"
        B = "B"
        H_0 = "initial_h"
        C_0 = "initial_c"
        P = "P"
        DIRECTION = "direction"
        LAYOUT = "layout"
        number_of_gates = 4
        number_of_peepholes = 3

        required_inputs = [X, W, R]
        for i in required_inputs:
            assert i in params, f"Missing Required Input: {i}"

        self.num_directions = params[W].shape[0]
        self.direction = params.get(DIRECTION, "forward")

        hidden_size = params[R].shape[-1]

        layout = params.get(LAYOUT, 0)
        x = params[X]
        x = x if layout == 0 else np.swapaxes(x, 0, 1)
        batch_size = x.shape[1]

        b = params.get(B, np.zeros((self.num_directions, 2 * number_of_gates * hidden_size), dtype=np.float32))
        p = params.get(P, np.zeros((self.num_directions, number_of_peepholes * hidden_size), dtype=np.float32))
        h_0 = params.get(H_0, np.zeros((self.num_directions, batch_size, hidden_size), dtype=np.float32))
        c_0 = params.get(C_0, np.zeros((self.num_directions, batch_size, hidden_size), dtype=np.float32))

        self.X = x
        self.W = params[W]
        self.R = params[R]
        self.B = b
        self.P = p
        self.H_0 = h_0
        self.C_0 = c_0
        self.LAYOUT = layout

    def f(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def g(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def h(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def _run_forward(self, X, W, R, B, P, H_0, C_0):
        """Run a forward pass of the LSTM.

        Assumes that the num_directions axis has been squeezed out of the
        inputs. (And returns Y, Yh without it.)
        """
        h_list = []

        [p_i, p_o, p_f] = np.split(P, 3)
        H_t = H_0
        C_t = C_0
        for x in X:
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
            h_list.append(H)
            H_t = H
            C_t = C

        Y = np.stack(h_list, axis=0)
        Y_h = H_t
        return Y, Y_h

    def step(self) -> tuple[np.ndarray, np.ndarray]:
        if self.direction == "forward":
            Y, Y_h = self._run_forward(
                self.X,
                self.W[0],
                self.R[0],
                self.B[0],
                self.P[0],
                self.H_0[0],
                self.C_0[0],
            )
            Y = np.expand_dims(Y, 1)
            Y_h = np.expand_dims(Y_h, 0)
        elif self.direction == "reverse":
            Y, Y_h = self._run_forward(
                np.flip(self.X, axis=0),
                self.W[0],
                self.R[0],
                self.B[0],
                self.P[0],
                self.H_0[0],
                self.C_0[0],
            )
            Y = np.flip(Y, axis=0)
            Y = np.expand_dims(Y, 1)
            Y_h = np.expand_dims(Y_h, 0)
        else:
            assert self.direction == "bidirectional"
            Yf, Yf_h = self._run_forward(
                self.X,
                self.W[0],
                self.R[0],
                self.B[0],
                self.P[0],
                self.H_0[0],
                self.C_0[0],
            )
            Yb, Yb_h = self._run_forward(
                np.flip(self.X, axis=0),
                self.W[1],
                self.R[1],
                self.B[1],
                self.P[1],
                self.H_0[1],
                self.C_0[1],
            )
            Yb = np.flip(Yb, axis=0)
            Y = np.stack([Yf, Yb], axis=1)
            Y_h = np.stack([Yf_h, Yb_h], axis=0)

        if self.LAYOUT:
            Y = np.transpose(Y, [2, 0, 1, 3])
            Y_h = np.transpose(Y_h, [1, 0, 2])

        return Y, Y_h


class LSTM(Base):
    @staticmethod
    def export_defaults() -> None:
        input = np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]).astype(np.float32)

        input_size = 2
        hidden_size = 3
        weight_scale = 0.1
        number_of_gates = 4

        node = onnx.helper.make_node(
            "LSTM", inputs=["X", "W", "R"], outputs=["", "Y_h"], hidden_size=hidden_size
        )

        W = weight_scale * np.ones(
            (1, number_of_gates * hidden_size, input_size)
        ).astype(np.float32)
        R = weight_scale * np.ones(
            (1, number_of_gates * hidden_size, hidden_size)
        ).astype(np.float32)

        lstm = LSTMHelper(X=input, W=W, R=R)
        _, Y_h = lstm.step()
        expect(
            node,
            inputs=[input, W, R],
            outputs=[Y_h.astype(np.float32)],
            name="test_lstm_defaults",
        )

    @staticmethod
    def export_initial_bias() -> None:
        input = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]).astype(
            np.float32
        )

        input_size = 3
        hidden_size = 4
        weight_scale = 0.1
        custom_bias = 0.1
        number_of_gates = 4

        node = onnx.helper.make_node(
            "LSTM",
            inputs=["X", "W", "R", "B"],
            outputs=["", "Y_h"],
            hidden_size=hidden_size,
        )

        W = weight_scale * np.ones(
            (1, number_of_gates * hidden_size, input_size)
        ).astype(np.float32)
        R = weight_scale * np.ones(
            (1, number_of_gates * hidden_size, hidden_size)
        ).astype(np.float32)

        # Adding custom bias
        W_B = custom_bias * np.ones((1, number_of_gates * hidden_size)).astype(
            np.float32
        )
        R_B = np.zeros((1, number_of_gates * hidden_size)).astype(np.float32)
        B = np.concatenate((W_B, R_B), 1)

        lstm = LSTMHelper(X=input, W=W, R=R, B=B)
        _, Y_h = lstm.step()
        expect(
            node,
            inputs=[input, W, R, B],
            outputs=[Y_h.astype(np.float32)],
            name="test_lstm_with_initial_bias",
        )

    @staticmethod
    def export_peepholes() -> None:
        input = np.array([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]).astype(
            np.float32
        )

        input_size = 4
        hidden_size = 3
        weight_scale = 0.1
        number_of_gates = 4
        number_of_peepholes = 3

        node = onnx.helper.make_node(
            "LSTM",
            inputs=["X", "W", "R", "B", "sequence_lens", "initial_h", "initial_c", "P"],
            outputs=["", "Y_h"],
            hidden_size=hidden_size,
        )

        # Initializing Inputs
        W = weight_scale * np.ones(
            (1, number_of_gates * hidden_size, input_size)
        ).astype(np.float32)
        R = weight_scale * np.ones(
            (1, number_of_gates * hidden_size, hidden_size)
        ).astype(np.float32)
        B = np.zeros((1, 2 * number_of_gates * hidden_size)).astype(np.float32)
        seq_lens = np.repeat(input.shape[0], input.shape[1]).astype(np.int32)
        init_h = np.zeros((1, input.shape[1], hidden_size)).astype(np.float32)
        init_c = np.zeros((1, input.shape[1], hidden_size)).astype(np.float32)
        P = weight_scale * np.ones((1, number_of_peepholes * hidden_size)).astype(
            np.float32
        )

        lstm = LSTMHelper(
            X=input, W=W, R=R, B=B, P=P, initial_c=init_c, initial_h=init_h
        )
        _, Y_h = lstm.step()
        expect(
            node,
            inputs=[input, W, R, B, seq_lens, init_h, init_c, P],
            outputs=[Y_h.astype(np.float32)],
            name="test_lstm_with_peepholes",
        )

    @staticmethod
    def export_batchwise() -> None:
        input = np.array([[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]]).astype(np.float32)

        input_size = 2
        hidden_size = 7
        weight_scale = 0.3
        number_of_gates = 4
        layout = 1

        node = onnx.helper.make_node(
            "LSTM",
            inputs=["X", "W", "R"],
            outputs=["Y", "Y_h"],
            hidden_size=hidden_size,
            layout=layout,
        )

        W = weight_scale * np.ones(
            (1, number_of_gates * hidden_size, input_size)
        ).astype(np.float32)
        R = weight_scale * np.ones(
            (1, number_of_gates * hidden_size, hidden_size)
        ).astype(np.float32)

        lstm = LSTMHelper(X=input, W=W, R=R, layout=layout)
        Y, Y_h = lstm.step()
        expect(
            node,
            inputs=[input, W, R],
            outputs=[Y.astype(np.float32), Y_h.astype(np.float32)],
            name="test_lstm_batchwise",
        )
