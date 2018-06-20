from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore
from typing import Any, Tuple

import onnx
from ..base import Base
from . import expect


class LSTM_Helper():

    number_of_gates = 4
    number_of_peepholes = 3

    def __init__(self, **params):  # type: (*Any) -> None
        # LSTM Input Names
        X = str('X')
        W = str('W')
        R = str('R')
        B = str('B')
        H_0 = str('initial_h')
        C_0 = str('initial_c')
        P = str('P')

        required_inputs = [X, W, R]
        for i in required_inputs:
            assert i in params, "Missing Required Input: {0}".format(i)

        self.num_directions = params[W].shape[0]

        if self.num_directions == 1:
            for k in params.keys():
                if k != X:
                    params[k] = np.squeeze(params[k], axis=0)

            hidden_size = params[R].shape[-1]
            batch_size = params[X].shape[1]

            b = params[B] if B in params else np.zeros(
                2 * self.number_of_gates * hidden_size, dtype=np.float32)
            p = params[P] if P in params else np.zeros(
                self.number_of_peepholes * hidden_size, dtype=np.float32)
            h_0 = params[H_0] if H_0 in params else np.zeros(
                (batch_size, hidden_size), dtype=np.float32)
            c_0 = params[C_0] if C_0 in params else np.zeros(
                (batch_size, hidden_size), dtype=np.float32)

            self.X = params[X]
            self.W = params[W]
            self.R = params[R]
            self.B = b
            self.P = p
            self.H_0 = h_0
            self.C_0 = c_0
        else:
            raise NotImplementedError()

    def f(self, x):  # type: (np.ndarray) -> np.ndarray
        return 1 / (1 + np.exp(-x))

    def g(self, x):  # type: (np.ndarray) -> np.ndarray
        return np.tanh(x)

    def h(self, x):  # type: (np.ndarray) -> np.ndarray
        return np.tanh(x)

    def step(self):  # type: () -> Tuple[np.ndarray, np.ndarray]
        [p_i, p_o, p_f] = np.split(self.P, 3)
        h_list = []
        H_t = self.H_0
        C_t = self.C_0
        for x in np.split(self.X, self.X.shape[0], axis=0):
            gates = np.dot(x, np.transpose(self.W)) + np.dot(
                H_t, np.transpose(self.R)) + np.add(*np.split(self.B, 2))
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
        concatenated = np.concatenate(h_list)
        if self.num_directions == 1:
            output = np.expand_dims(concatenated, 1)
        return output, h_list[-1]

    @staticmethod
    def get_random(shape, scale=1.0):  # type: ignore
        """
        return float 32 array with values uniformly sampled from [-scale, scale)
        """
        return (scale * (2. * np.random.random_sample(shape) - 1.)).astype(np.float32)


class LSTM(Base):
    @staticmethod
    def export_defaults():  # type: () -> None
        input_size = 2
        batch_size = 4
        hidden_size = 3
        input = LSTM_Helper.get_random([1, batch_size, input_size])

        node = onnx.helper.make_node(
            'LSTM',
            inputs=['X', 'W', 'R'],
            outputs=['', 'Y'],
            hidden_size=hidden_size)

        W = LSTM_Helper.get_random(
            [1, LSTM_Helper.number_of_gates * hidden_size, input_size])
        R = LSTM_Helper.get_random(
            [1, LSTM_Helper.number_of_gates * hidden_size, hidden_size])

        lstm = LSTM_Helper(X=input, W=W, R=R)
        _, Y_h = lstm.step()
        expect(
            node,
            inputs=[input, W, R],
            outputs=[Y_h.astype(np.float32)],
            name='test_lstm_defaults')

    @staticmethod
    def export_initial_bias():  # type: () -> None
        input_size = 3
        batch_size = 3
        hidden_size = 4
        input = LSTM_Helper.get_random([1, batch_size, input_size])

        node = onnx.helper.make_node(
            'LSTM',
            inputs=['X', 'W', 'R', 'B'],
            outputs=['', 'Y'],
            hidden_size=hidden_size)

        W = LSTM_Helper.get_random(
            [1, LSTM_Helper.number_of_gates * hidden_size, input_size])
        R = LSTM_Helper.get_random(
            [1, LSTM_Helper.number_of_gates * hidden_size, hidden_size])

        # Adding custom bias
        W_B = LSTM_Helper.get_random(
            [1, LSTM_Helper.number_of_gates * hidden_size])
        R_B = np.zeros(
            (1, LSTM_Helper.number_of_gates * hidden_size)).astype(np.float32)
        B = np.concatenate((W_B, R_B), 1)

        lstm = LSTM_Helper(X=input, W=W, R=R, B=B)
        _, Y_h = lstm.step()
        expect(
            node,
            inputs=[input, W, R, B],
            outputs=[Y_h.astype(np.float32)],
            name='test_lstm_with_initial_bias')

    @staticmethod
    def export_peepholes():  # type: () -> None
        input_size = 4
        batch_size = 2
        hidden_size = 3
        input = np.random.rand(1, batch_size, input_size).astype(np.float32)

        node = onnx.helper.make_node(
            'LSTM',
            inputs=[
                'X', 'W', 'R', 'B', 'sequence_lens', 'initial_h', 'initial_c',
                'P'
            ],
            outputs=['', 'Y'],
            hidden_size=hidden_size)

        # Initializing Inputs
        W = LSTM_Helper.get_random(
            [1, LSTM_Helper.number_of_gates * hidden_size, input_size])
        R = LSTM_Helper.get_random(
            [1, LSTM_Helper.number_of_gates * hidden_size, hidden_size])
        B = np.zeros(
            (1,
             2 * LSTM_Helper.number_of_gates * hidden_size)).astype(np.float32)
        seq_lens = np.repeat(input.shape[0], input.shape[1]).astype(np.int32)
        init_h = np.zeros((1, input.shape[1], hidden_size)).astype(np.float32)
        init_c = np.zeros((1, input.shape[1], hidden_size)).astype(np.float32)
        P = LSTM_Helper.get_random(
            [1, LSTM_Helper.number_of_peepholes * hidden_size])
        lstm = LSTM_Helper(
            X=input, W=W, R=R, B=B, P=P, initial_c=init_c, initial_h=init_h)
        _, Y_h = lstm.step()
        expect(
            node,
            inputs=[input, W, R, B, seq_lens, init_h, init_c, P],
            outputs=[Y_h.astype(np.float32)],
            name='test_lstm_with_peepholes')
