from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore
from typing import Any, Tuple

import onnx
from ..base import Base
from . import expect


class GRU_Helper():
    def __init__(self, **params):  # type: (*Any) -> None
        # GRU Input Names
        X = str('X')
        W = str('W')
        R = str('R')
        B = str('B')
        H_0 = str('initial_h')
        LBR = str('linear_before_reset')
        TM = str('time_major')
        number_of_gates = 3

        required_inputs = [X, W, R]
        for i in required_inputs:
            assert i in params, "Missing Required Input: {0}".format(i)

        self.num_directions = params[W].shape[-1]

        if self.num_directions == 1:
            for k in params.keys():
                if k != X:
                    params[k] = np.squeeze(params[k], axis=-1)

            hidden_size = params[R].shape[-1]
            batch_size = params[X].shape[1]

            tm = params[TM] if TM in params else 1
            x = params[X]
            x = x if tm == 1 else np.swapaxes(x, 0, 1)
            b = params[B] if B in params else np.zeros(2 * number_of_gates * hidden_size)
            h_0 = params[H_0] if H_0 in params else np.zeros((batch_size, hidden_size))
            lbr = params[LBR] if LBR in params else 0

            self.X = x
            self.W = params[W]
            self.R = params[R]
            self.B = b
            self.H_0 = h_0
            self.LBR = lbr
            self.TM = tm

        else:
            raise NotImplementedError()

    def f(self, x):  # type: (np.ndarray) -> np.ndarray
        return 1 / (1 + np.exp(-x))

    def g(self, x):  # type: (np.ndarray) -> np.ndarray
        return np.tanh(x)

    def step(self):  # type: () -> Tuple[np.ndarray, np.ndarray]
        seq_length = self.X.shape[0]
        hidden_size = self.H_0.shape[-1]
        batch_size = self.X.shape[1]

        Y = np.empty([seq_length, batch_size, hidden_size, self.num_directions])
        h_list = []

        [w_z, w_r, w_h] = np.split(self.W, 3)
        [r_z, r_r, r_h] = np.split(self.R, 3)
        [w_bz, w_br, w_bh, r_bz, r_br, r_bh] = np.split(self.B, 6)
        gates_w = np.transpose(np.concatenate((w_z, w_r)))
        gates_r = np.transpose(np.concatenate((r_z, r_r)))
        gates_b = np.add(np.concatenate((w_bz, w_br)), np.concatenate((r_bz, r_br)))

        H_t = self.H_0
        for x in np.split(self.X, self.X.shape[0], axis=0):
            gates = np.dot(x, gates_w) + np.dot(H_t, gates_r) + gates_b
            z, r = np.split(gates, 2, -1)
            z = self.f(z)
            r = self.f(r)
            h_default = self.g(np.dot(x, np.transpose(w_h)) + np.dot(r * H_t, np.transpose(r_h)) + w_bh + r_bh)
            h_linear = self.g(np.dot(x, np.transpose(w_h)) + r * (np.dot(H_t, np.transpose(r_h)) + r_bh) + w_bh)
            h = h_linear if self.LBR else h_default
            H = (1 - z) * h + z * H_t
            h_list.append(H)
            H_t = H

        concatenated = np.concatenate(h_list)
        if self.num_directions == 1:
            Y[:, :, :, 0] = concatenated

        return Y if self.TM == 1 else np.swapaxes(Y, 0, 1), Y[-1]


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
            outputs=['', 'Y_h'],
            hidden_size=hidden_size
        )

        W = weight_scale * np.ones((number_of_gates * hidden_size, input_size, 1)).astype(np.float32)
        R = weight_scale * np.ones((number_of_gates * hidden_size, hidden_size, 1)).astype(np.float32)

        gru = GRU_Helper(X=input, W=W, R=R)
        _, Y_h = gru.step()
        expect(node, inputs=[input, W, R], outputs=[Y_h.astype(np.float32)], name='test_gru_defaults')

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
            outputs=['', 'Y_h'],
            hidden_size=hidden_size
        )

        W = weight_scale * np.ones((number_of_gates * hidden_size, input_size, 1)).astype(np.float32)
        R = weight_scale * np.ones((number_of_gates * hidden_size, hidden_size, 1)).astype(np.float32)

        # Adding custom bias
        W_B = custom_bias * np.ones((number_of_gates * hidden_size, 1)).astype(np.float32)
        R_B = np.zeros((number_of_gates * hidden_size, 1)).astype(np.float32)
        B = np.concatenate((W_B, R_B), axis=0)

        gru = GRU_Helper(X=input, W=W, R=R, B=B)
        _, Y_h = gru.step()
        expect(node, inputs=[input, W, R, B], outputs=[Y_h.astype(np.float32)], name='test_gru_with_initial_bias')

    @staticmethod
    def export_seq_length():  # type: () -> None
        input = np.array([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
                          [[10., 11., 12.], [13., 14., 15.], [16., 17., 18.]]]).astype(np.float32)

        input_size = 3
        hidden_size = 5
        number_of_gates = 3

        node = onnx.helper.make_node(
            'GRU',
            inputs=['X', 'W', 'R', 'B'],
            outputs=['', 'Y_h'],
            hidden_size=hidden_size
        )

        W = np.random.randn(number_of_gates * hidden_size, input_size, 1).astype(np.float32)
        R = np.random.randn(number_of_gates * hidden_size, hidden_size, 1).astype(np.float32)

        # Adding custom bias
        W_B = np.random.randn(number_of_gates * hidden_size, 1).astype(np.float32)
        R_B = np.random.randn(number_of_gates * hidden_size, 1).astype(np.float32)
        B = np.concatenate((W_B, R_B), axis=0)

        gru = GRU_Helper(X=input, W=W, R=R, B=B)
        _, Y_h = gru.step()
        expect(node, inputs=[input, W, R, B], outputs=[Y_h.astype(np.float32)], name='test_gru_seq_length')

    @staticmethod
    def export_batchwise():  # type: () -> None
        input = np.array([[[1., 2.]], [[3., 4.]], [[5., 6.]]]).astype(np.float32)

        input_size = 2
        hidden_size = 6
        number_of_gates = 3
        weight_scale = 0.2
        time_major = 0

        node = onnx.helper.make_node(
            'GRU',
            inputs=['X', 'W', 'R'],
            outputs=['Y', 'Y_h'],
            hidden_size=hidden_size,
            time_major=time_major
        )

        W = weight_scale * np.ones((number_of_gates * hidden_size, input_size, 1)).astype(np.float32)
        R = weight_scale * np.ones((number_of_gates * hidden_size, hidden_size, 1)).astype(np.float32)

        gru = GRU_Helper(X=input, W=W, R=R, time_major=time_major)
        Y, Y_h = gru.step()
        expect(node, inputs=[input, W, R], outputs=[Y.astype(np.float32), Y_h.astype(np.float32)], name='test_gru_batchwise')
