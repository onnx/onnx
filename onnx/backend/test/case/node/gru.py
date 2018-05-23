from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore
from typing import Any

import onnx
from ..base import Base
from . import expect


class GRU_Helper():
    def __init__(self, **params):  # type: (*Any) -> None
        #GRU Input Names
        X = str('X')
        W = str('W')
        R = str('R')
        B = str('B')
        H_0 = str('initial_h')
        LBR = str('linear_before_reset')
        number_of_gates = 3

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
            h_0 = params[H_0] if H_0 in params else np.zeros((self.batch_size, self.hidden_size))
            lbr = params[LBR] if LBR in params else 0

            self.X = params[X]
            self.W = params[W]
            self.R = params[R]
            self.B = b
            self.H_0 = h_0
            self.LBR = lbr

        else:
            raise NotImplementedError()

    def f(self, x):  # type: (np.ndarray) -> np.ndarray
        return 1 / (1 + np.exp(-x))

    def g(self, x):  # type: (np.ndarray) -> np.ndarray
        return np.tanh(x)

    def step(self):  # type: () -> np.ndarray
        [w_z, w_r, w_h] = np.split(self.W, 3)
        [r_z, r_r, r_h] = np.split(self.R, 3)
        [w_bz, w_br, w_bh, r_bz, r_br, r_bh] = np.split(self.B, 6)

        z = self.f(np.dot(self.X, np.transpose(w_z)) + np.dot(self.H_0, r_z) + w_bz + r_bz)
        r = self.f(np.dot(self.X, np.transpose(w_r)) + np.dot(self.H_0, r_r) + w_br + r_br)
        h_default = self.g(np.dot(self.X, np.transpose(w_h)) + np.dot(r * self.H_0, r_h) + w_bh + r_bh)
        h_linear = self.g(np.dot(self.X, np.transpose(w_h)) + r * (np.dot(self.H_0, r_h) + r_bh) + w_bh)
        h = h_linear if self.LBR else h_default
        H = (1 - z) * h + z * self.H_0
        return np.reshape(H, (self.num_directions, self.batch_size, self.hidden_size))


class GRU(Base):

    @staticmethod
    def export_defaults():  # type: () -> None
            input = np.array([[[1., 2.], [3., 4.], [5., 6.]]]).astype(np.float32)

            input_size = 2
            hidden_size = 5
            weight_scale = 0.1
            number_of_gates = 3

            node = onnx.helper.make_node(
                'GRU',
                inputs=['X', 'W', 'R'],
                outputs=['Y'],
                hidden_size=hidden_size
            )

            W = weight_scale * np.ones((1, number_of_gates * hidden_size, input_size)).astype(np.float32)
            R = weight_scale * np.ones((1, number_of_gates * hidden_size, hidden_size)).astype(np.float32)

            gru = GRU_Helper(X=input, W=W, R=R)
            output = gru.step().astype(np.float32)

            expect(node, inputs=[input, W, R], outputs=[output], name='test_gru_defaults')

    @staticmethod
    def export_initial_bias():  # type: () -> None
            input = np.array([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]]).astype(np.float32)

            input_size = 3
            hidden_size = 3
            weight_scale = 0.1
            custom_bias = 0.1
            number_of_gates = 3

            node = onnx.helper.make_node(
                'GRU',
                inputs=['X', 'W', 'R', 'B'],
                outputs=['Y'],
                hidden_size=hidden_size
            )

            W = weight_scale * np.ones((1, number_of_gates * hidden_size, input_size)).astype(np.float32)
            R = weight_scale * np.ones((1, number_of_gates * hidden_size, hidden_size)).astype(np.float32)

            # Adding custom bias
            W_B = custom_bias * np.ones((1, number_of_gates * hidden_size)).astype(np.float32)
            R_B = np.zeros((1, number_of_gates * hidden_size)).astype(np.float32)
            B = np.concatenate((W_B, R_B), axis=1)

            gru = GRU_Helper(X=input, W=W, R=R, B=B)
            output = gru.step().astype(np.float32)

            expect(node, inputs=[input, W, R, B], outputs=[output], name='test_gru_with_initial_bias')
