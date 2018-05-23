from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore
from typing import Any

import onnx
from ..base import Base
from . import expect


class LSTM_Helper():
    def __init__(self, **params):  # type: (*Any) -> None
        #LSTM Input Names
        X = str('X')
        W = str('W')
        R = str('R')
        B = str('B')
        H_0 = str('initial_h')
        C_0 = str('initial_c')
        P = str('P')
        number_of_gates = 4
        number_of_peepholes = 3

        required_inputs = [X, W, R]
        for i in required_inputs:
            assert i in params, "Missing Required Input: {0}".format(i)

        self.num_directions = params[W].shape[0]

        if(self.num_directions == 1):
            for k in params.keys():
                params[k] = np.squeeze(params[k], axis=0)

            self.hidden_size = params[R].shape[-1]
            self.batch_size = params[X].shape[0]

            b = params[B] if B in params else np.zeros(2 * number_of_gates * self.hidden_size)
            p = params[P] if P in params else np.zeros(number_of_peepholes * self.hidden_size)
            h_0 = params[H_0] if H_0 in params else np.zeros((self.batch_size, self.hidden_size))
            c_0 = params[C_0] if C_0 in params else np.zeros((self.batch_size, self.hidden_size))

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

    def step(self):  # type: () -> np.ndarray
        [w_i, w_o, w_f, w_c] = np.split(self.W, 4)
        [r_i, r_o, r_f, r_c] = np.split(self.R, 4)
        [p_i, p_o, p_f] = np.split(self.P, 3)
        [w_bi, w_bo, w_bf, w_bc, r_bi, r_bo, r_bf, r_bc] = np.split(self.B, 8)

        i = self.f(np.dot(self.X, np.transpose(w_i)) + np.dot(self.H_0, r_i) + w_bi + r_bi + p_i * self.C_0)
        f = self.f(np.dot(self.X, np.transpose(w_f)) + np.dot(self.H_0, r_f) + w_bf + r_bf + p_f * self.C_0)
        c = self.g(np.dot(self.X, np.transpose(w_c)) + np.dot(self.H_0, r_c) + w_bc + r_bc)
        C = f * self.C_0 + i * c
        o = self.f(np.dot(self.X, np.transpose(w_o)) + np.dot(self.H_0, r_o) + w_bo + r_bo + p_o * C)
        H = o * self.h(C)
        return np.reshape(H, (self.num_directions, self.batch_size, self.hidden_size)).astype(np.float32)


class LSTM(Base):

    @staticmethod
    def export_defaults():  # type: () -> None
        input = np.array([[[1., 2.], [3., 4.], [5., 6.]]]).astype(np.float32)

        input_size = 2
        hidden_size = 3
        weight_scale = 0.1
        number_of_gates = 4

        node = onnx.helper.make_node(
            'LSTM',
            inputs=['X', 'W', 'R'],
            outputs=['Y'],
            hidden_size=hidden_size
        )

        W = weight_scale * np.ones((1, number_of_gates * hidden_size, input_size)).astype(np.float32)
        R = weight_scale * np.ones((1, number_of_gates * hidden_size, hidden_size)).astype(np.float32)

        lstm = LSTM_Helper(X=input, W=W, R=R)
        output = lstm.step()

        expect(node, inputs=[input, W, R], outputs=[output], name='test_lstm_defaults')

    @staticmethod
    def export_initial_bias():  # type: () -> None
        input = np.array([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]]).astype(np.float32)

        input_size = 3
        hidden_size = 4
        weight_scale = 0.1
        custom_bias = 0.1
        number_of_gates = 4

        node = onnx.helper.make_node(
            'LSTM',
            inputs=['X', 'W', 'R', 'B'],
            outputs=['Y'],
            hidden_size=hidden_size
        )

        W = weight_scale * np.ones((1, number_of_gates * hidden_size, input_size)).astype(np.float32)
        R = weight_scale * np.ones((1, number_of_gates * hidden_size, hidden_size)).astype(np.float32)

        # Adding custom bias
        W_B = custom_bias * np.ones((1, number_of_gates * hidden_size)).astype(np.float32)
        R_B = np.zeros((1, number_of_gates * hidden_size)).astype(np.float32)
        B = np.concatenate((W_B, R_B), 1)

        lstm = LSTM_Helper(X=input, W=W, R=R, B=B)
        output = lstm.step()

        expect(node, inputs=[input, W, R, B], outputs=[output], name='test_lstm_with_initial_bias')

    @staticmethod
    def export_peepholes():  # type: () -> None
            input = np.array([[[1., 2., 3., 4.], [5., 6., 7., 8.]]]).astype(np.float32)

            input_size = 4
            hidden_size = 3
            weight_scale = 0.1
            number_of_gates = 4
            number_of_peepholes = 3

            node = onnx.helper.make_node(
                'LSTM',
                inputs=['X', 'W', 'R', 'B', 'sequence_lens', 'initial_h', 'initial_c', 'P'],
                outputs=['Y'],
                hidden_size=hidden_size
            )

            # Initializing Inputs
            W = weight_scale * np.ones((1, number_of_gates * hidden_size, input_size)).astype(np.float32)
            R = weight_scale * np.ones((1, number_of_gates * hidden_size, hidden_size)).astype(np.float32)
            B = np.zeros((1, 2 * number_of_gates * hidden_size)).astype(np.float32)
            seq_lens = np.repeat(input.shape[0], input.shape[1]).astype(np.int32)
            init_h = np.zeros((1, input.shape[1], hidden_size)).astype(np.float32)
            init_c = np.zeros((1, input.shape[1], hidden_size)).astype(np.float32)
            P = weight_scale * np.ones((1, number_of_peepholes * hidden_size)).astype(np.float32)

            lstm = LSTM_Helper(X=input, W=W, R=R, B=B, P=P, initial_c=init_c, initial_h=init_h)
            output = lstm.step()

            expect(node, inputs=[input, W, R, B, seq_lens, init_h, init_c, P], outputs=[output], name='test_lstm_with_peepholes')
