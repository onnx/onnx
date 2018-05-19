from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class RNN_Helper():
    def __init__(self, **params):
        #RNN Input Names
        X = 'X'
        W = 'W'
        R = 'R'
        B = 'B'
        H_0 = 'initial_h'

        required_inputs = [X, W, R]
        for i in required_inputs:
            assert i in params, "Missing Required Input: {0}".format(i)

        self.num_directions = params[W].shape[0]

        if(self.num_directions == 1):
            for k in params.keys():
                params[k] = np.squeeze(params[k], axis=0)

            self.hidden_size = params[R].shape[-1]
            self.batch_size = params[X].shape[0]

            b = params[B] if B in params else np.zeros(2 * self.hidden_size)
            h_0 = params[H_0] if H_0 in params else np.zeros((self.batch_size, self.hidden_size))

            self.X = params[X]
            self.W = params[W]
            self.R = params[R]
            self.B = b
            self.H_0 = h_0
        else:
            raise NotImplementedError()

    def f(self, x):
        return np.tanh(x)

    def step(self):
        [w_b, r_b] = np.split(self.B, 2)

        H = self.f(np.dot(self.X, np.transpose(self.W)) + np.dot(self.H_0, self.R) + w_b + r_b)
        return np.reshape(H, (self.num_directions, self.batch_size, self.hidden_size))


class RNN(Base):

    @staticmethod
    def export_defaults():
        input = np.array([[[1., 2.], [3., 4.], [5., 6.]]]).astype(np.float32)

        input_size = 2
        hidden_size = 4
        weight_scale = 0.1

        node = onnx.helper.make_node(
            'RNN',
            inputs=['X', 'W', 'R'],
            outputs=['Y'],
            hidden_size=hidden_size
        )

        W = weight_scale * np.ones((1, hidden_size, input_size)).astype(np.float32)
        R = weight_scale * np.ones((1, hidden_size, hidden_size)).astype(np.float32)

        rnn = RNN_Helper(X=input, W=W, R=R)
        output = rnn.step().astype(np.float32)

        expect(node, inputs=[input, W, R], outputs=[output], name='test_simple_rnn_defaults')

    @staticmethod
    def export_initial_bias():
        input = np.array([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]]).astype(np.float32)

        input_size = 3
        hidden_size = 5
        custom_bias = 0.1
        weight_scale = 0.1

        node = onnx.helper.make_node(
            'RNN',
            inputs=['X', 'W', 'R', 'B'],
            outputs=['Y'],
            hidden_size=hidden_size
        )

        W = weight_scale * np.ones((1, hidden_size, input_size)).astype(np.float32)
        R = weight_scale * np.ones((1, hidden_size, hidden_size)).astype(np.float32)

        # Adding custom bias
        W_B = custom_bias * np.ones((1, hidden_size)).astype(np.float32)
        R_B = np.zeros((1, hidden_size)).astype(np.float32)
        B = np.concatenate((W_B, R_B), axis=1)

        rnn = RNN_Helper(X=input, W=W, R=R, B=B)
        output = rnn.step().astype(np.float32)

        expect(node, inputs=[input, W, R, B], outputs=[output], name='test_simple_rnn_with_initial_bias')
