# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class RNNHelper:
    def __init__(self, **params: Any) -> None:
        # RNN Input Names
        X = "X"
        W = "W"
        R = "R"
        B = "B"
        H_0 = "initial_h"
        DIRECTION = "direction"
        LAYOUT = "layout"

        required_inputs = [X, W, R]
        for i in required_inputs:
            assert i in params, f"Missing Required Input: {i}"

        self.num_directions = params[str(W)].shape[0]
        self.direction = params.get(DIRECTION, "forward")

        hidden_size = params[R].shape[-1]

        layout = params.get(LAYOUT, 0)
        x = params[X]
        x = x if layout == 0 else np.swapaxes(x, 0, 1)
        batch_size = x.shape[1]
        b = params.get(
            B, np.zeros((self.num_directions, 2 * hidden_size), dtype=np.float32)
        )
        h_0 = params.get(
            H_0,
            np.zeros((self.num_directions, batch_size, hidden_size), dtype=np.float32),
        )

        self.X = x
        self.W = params[W]
        self.R = params[R]
        self.B = b
        self.H_0 = h_0
        self.LAYOUT = layout

    def f(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def _run_forward(self, X, R, B, W, H_0):
        """Run a forward pass of the RNN.

        Assumes that the num_directions axis has been squeezed out of the
        inputs. (And returns Y, Yh without it.)
        """
        h_list = []

        H_t = H_0
        for x in X:
            H = self.f(
                np.dot(x, np.transpose(W))
                + np.dot(H_t, np.transpose(R))
                + np.add(*np.split(B, 2))
            )
            h_list.append(H)
            H_t = H

        Y = np.stack(h_list, axis=0)
        Y_h = H_t
        return Y, Y_h

    def step(self) -> tuple[np.ndarray, np.ndarray]:
        if self.direction == "forward":
            Y, Y_h = self._run_forward(
                self.X,
                self.R[0],
                self.B[0],
                self.W[0],
                self.H_0[0],
            )
            # Add num_directions axis to output
            Y = np.expand_dims(Y, 1)
            Y_h = np.expand_dims(Y_h, 0)
        elif self.direction == "reverse":
            Y, Y_h = self._run_forward(
                np.flip(self.X, axis=0),
                self.R[0],
                self.B[0],
                self.W[0],
                self.H_0[0],
            )
            Y = np.flip(Y, axis=0)
            Y = np.expand_dims(Y, 1)
            Y_h = np.expand_dims(Y_h, 0)
        else:
            assert self.direction == "bidirectional"
            Yf, Yf_h = self._run_forward(
                self.X,
                self.R[0],
                self.B[0],
                self.W[0],
                self.H_0[0],
            )
            Yb, Yb_h = self._run_forward(
                np.flip(self.X, axis=0),
                self.R[1],
                self.B[1],
                self.W[1],
                self.H_0[1],
            )
            Yb = np.flip(Yb, axis=0)
            Y = np.stack([Yf, Yb], axis=1)
            Y_h = np.stack([Yf_h, Yb_h], axis=0)

        if self.LAYOUT:
            Y = np.transpose(Y, [2, 0, 1, 3])
            Y_h = np.transpose(Y_h, [1, 0, 2])

        return Y, Y_h


class RNN(Base):
    @staticmethod
    def export_defaults() -> None:
        input = np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]).astype(np.float32)

        input_size = 2
        hidden_size = 4
        weight_scale = 0.1

        node = onnx.helper.make_node(
            "RNN", inputs=["X", "W", "R"], outputs=["", "Y_h"], hidden_size=hidden_size
        )

        W = weight_scale * np.ones((1, hidden_size, input_size)).astype(np.float32)
        R = weight_scale * np.ones((1, hidden_size, hidden_size)).astype(np.float32)

        rnn = RNNHelper(X=input, W=W, R=R)
        _, Y_h = rnn.step()
        expect(
            node,
            inputs=[input, W, R],
            outputs=[Y_h.astype(np.float32)],
            name="test_simple_rnn_defaults",
        )

    @staticmethod
    def export_initial_bias() -> None:
        input = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]).astype(
            np.float32
        )

        input_size = 3
        hidden_size = 5
        custom_bias = 0.1
        weight_scale = 0.1

        node = onnx.helper.make_node(
            "RNN",
            inputs=["X", "W", "R", "B"],
            outputs=["", "Y_h"],
            hidden_size=hidden_size,
        )

        W = weight_scale * np.ones((1, hidden_size, input_size)).astype(np.float32)
        R = weight_scale * np.ones((1, hidden_size, hidden_size)).astype(np.float32)

        # Adding custom bias
        W_B = custom_bias * np.ones((1, hidden_size)).astype(np.float32)
        R_B = np.zeros((1, hidden_size)).astype(np.float32)
        B = np.concatenate((W_B, R_B), axis=1)

        rnn = RNNHelper(X=input, W=W, R=R, B=B)
        _, Y_h = rnn.step()
        expect(
            node,
            inputs=[input, W, R, B],
            outputs=[Y_h.astype(np.float32)],
            name="test_simple_rnn_with_initial_bias",
        )

    @staticmethod
    def export_seq_length() -> None:
        input = np.array(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
            ]
        ).astype(np.float32)

        input_size = 3
        hidden_size = 5

        node = onnx.helper.make_node(
            "RNN",
            inputs=["X", "W", "R", "B"],
            outputs=["", "Y_h"],
            hidden_size=hidden_size,
        )

        W = np.random.randn(1, hidden_size, input_size).astype(np.float32)
        R = np.random.randn(1, hidden_size, hidden_size).astype(np.float32)

        # Adding custom bias
        W_B = np.random.randn(1, hidden_size).astype(np.float32)
        R_B = np.random.randn(1, hidden_size).astype(np.float32)
        B = np.concatenate((W_B, R_B), axis=1)

        rnn = RNNHelper(X=input, W=W, R=R, B=B)
        _, Y_h = rnn.step()
        expect(
            node,
            inputs=[input, W, R, B],
            outputs=[Y_h.astype(np.float32)],
            name="test_rnn_seq_length",
        )

    @staticmethod
    def export_batchwise() -> None:
        input = np.array([[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]]).astype(np.float32)

        input_size = 2
        hidden_size = 4
        weight_scale = 0.5
        layout = 1

        node = onnx.helper.make_node(
            "RNN",
            inputs=["X", "W", "R"],
            outputs=["Y", "Y_h"],
            hidden_size=hidden_size,
            layout=layout,
        )

        W = weight_scale * np.ones((1, hidden_size, input_size)).astype(np.float32)
        R = weight_scale * np.ones((1, hidden_size, hidden_size)).astype(np.float32)

        rnn = RNNHelper(X=input, W=W, R=R, layout=layout)
        Y, Y_h = rnn.step()
        expect(
            node,
            inputs=[input, W, R],
            outputs=[Y.astype(np.float32), Y_h.astype(np.float32)],
            name="test_simple_rnn_batchwise",
        )

    @staticmethod
    def export_reverse() -> None:
        input = np.array([[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]]).astype(np.float32)

        input_size = 2
        hidden_size = 4
        weight_scale = 0.1

        node = onnx.helper.make_node(
            "RNN",
            inputs=["X", "W", "R"],
            outputs=["", "Y_h"],
            hidden_size=hidden_size,
            direction="reverse",
        )

        W = weight_scale * np.ones((1, hidden_size, input_size)).astype(np.float32)
        R = weight_scale * np.ones((1, hidden_size, hidden_size)).astype(np.float32)

        rnn = RNNHelper(X=input, W=W, R=R, direction="reverse")
        _, Y_h = rnn.step()
        expect(
            node,
            inputs=[input, W, R],
            outputs=[Y_h.astype(np.float32)],
            name="test_simple_rnn_reverse",
        )

    @staticmethod
    def export_bidirectional() -> None:
        input = np.array([[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]]).astype(np.float32)

        input_size = 2
        hidden_size = 4
        weight_scales = np.array([[[0.5]], [[2.0]]], dtype=np.float32)

        node = onnx.helper.make_node(
            "RNN",
            inputs=["X", "W", "R"],
            outputs=["", "Y_h"],
            hidden_size=hidden_size,
            direction="bidirectional",
        )

        # Multiply broadcasts in num_directions axis, to have different W & R
        # in each direction
        W = weight_scales * np.ones((1, hidden_size, input_size)).astype(np.float32)
        R = weight_scales * np.ones((1, hidden_size, hidden_size)).astype(np.float32)

        rnn = RNNHelper(X=input, W=W, R=R, direction="bidirectional")
        _, Y_h = rnn.step()
        expect(
            node,
            inputs=[input, W, R],
            outputs=[Y_h.astype(np.float32)],
            name="test_simple_rnn_bidirectional",
        )
