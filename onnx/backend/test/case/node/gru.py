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

        required_inputs = [X, W, R]
        for i in required_inputs:
            assert i in params, "Missing Required Input: {0}".format(i)

        # RNN Attr Names
        ACTS = str('activations')
        CLIP = str('clip')
        DIR = str('direction')
        H_SIZE = str('hidden_size')
        LBR = str('linear_before_reset')

        required_attr = [H_SIZE]

        for i in required_attr:
            assert i in params, "Missing attribute: {0}".format(i)

        number_of_gates = 3
        self.hidden_size = params[H_SIZE]
        self.direction = params[DIR] if DIR in params else 'forward'
        self.num_directions = 2 if self.direction == 'bidirectional' else 1

        activation_lookup = {'Relu': lambda x: np.maximum(x, 0), 'Tanh': np.tanh, 'Sigmoid': lambda x: 1 / (1 + np.exp(-x))}
        f_activations = params[ACTS] if ACTS in params else ['Sigmoid'] * self.num_directions
        self.f = [activation_lookup[act] for act in f_activations if act in activation_lookup]
        g_activations = params[ACTS] if ACTS in params else ['Tanh'] * self.num_directions
        self.g = [activation_lookup[act] for act in g_activations if act in activation_lookup]

        self.B = params[B] if B in params else np.zeros((self.num_directions, 2 * number_of_gates * self.hidden_size), dtype=np.float32)

        self.clip = params[CLIP] if CLIP in params else np.inf

        self.batch_size = params[X].shape[1]
        h_0 = params[H_0] if H_0 in params else np.zeros((self.num_directions, self.batch_size, self.hidden_size), dtype=np.float32)
        self.H_0 = h_0
        self.LBR = params[LBR] if LBR in params else 0

        self.X = params[X]
        self.W = params[W]
        self.R = params[R]

    def step(self):  # type: () -> Tuple[np.ndarray, np.ndarray]
        seq_length = self.X.shape[0]
        Y = np.empty([seq_length, self.num_directions, self.batch_size, self.hidden_size])

        xs = np.split(self.X, self.X.shape[0], axis=0)

        dirs = {'forward', 'reverse'} if self.direction == 'bidirectional' else {self.direction}
        for i, direction in enumerate(dirs):
            h_list = []
            H_t = self.H_0[i]

            [w_z, w_r, w_h] = np.split(self.W[i], 3)
            [r_z, r_r, r_h] = np.split(self.R[i], 3)
            [w_bz, w_br, w_bh, r_bz, r_br, r_bh] = np.split(self.B[i], 6)

            gates_w = np.transpose(np.concatenate((w_z, w_r)))
            gates_r = np.transpose(np.concatenate((r_z, r_r)))
            gates_b = np.add(np.concatenate((w_bz, w_br)), np.concatenate((r_bz, r_br)))

            if direction == 'reverse':
                xs = xs[::-1]

            for x in xs:
                gates = np.dot(x, gates_w) + np.dot(H_t, gates_r) + gates_b
                z, r = np.split(gates, 2, -1)
                z = self.f[i](np.clip(z, -self.clip, self.clip))
                r = self.f[i](np.clip(r, -self.clip, self.clip))
                h_default = self.g[i](np.clip(np.dot(x, np.transpose(w_h)) + np.dot(r * H_t, np.transpose(r_h)) + w_bh + r_bh, -self.clip, self.clip))
                h_linear = self.g[i](np.clip(np.dot(x, np.transpose(w_h)) + r * (np.dot(H_t, np.transpose(r_h)) + r_bh) + w_bh, -self.clip, self.clip))
                h = h_linear if self.LBR else h_default
                H = (1 - z) * h + z * H_t
                h_list.append(H)
                H_t = H

            if direction == 'reverse':
                h_list = h_list[::-1]

            concatenated = np.concatenate(h_list)
            Y[:, i] = concatenated

        return Y, Y[-1]


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

        W = weight_scale * np.ones((1, number_of_gates * hidden_size, input_size)).astype(np.float32)
        R = weight_scale * np.ones((1, number_of_gates * hidden_size, hidden_size)).astype(np.float32)

        gru = GRU_Helper(X=input, W=W, R=R, hidden_size=hidden_size)
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

        W = weight_scale * np.ones((1, number_of_gates * hidden_size, input_size)).astype(np.float32)
        R = weight_scale * np.ones((1, number_of_gates * hidden_size, hidden_size)).astype(np.float32)

        # Adding custom bias
        W_B = custom_bias * np.ones((1, number_of_gates * hidden_size)).astype(np.float32)
        R_B = np.zeros((1, number_of_gates * hidden_size)).astype(np.float32)
        B = np.concatenate((W_B, R_B), axis=1)

        gru = GRU_Helper(X=input, W=W, R=R, B=B, hidden_size=hidden_size)
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

        W = np.random.randn(1, number_of_gates * hidden_size, input_size).astype(np.float32)
        R = np.random.randn(1, number_of_gates * hidden_size, hidden_size).astype(np.float32)

        # Adding custom bias
        W_B = np.random.randn(1, number_of_gates * hidden_size).astype(np.float32)
        R_B = np.random.randn(1, number_of_gates * hidden_size).astype(np.float32)
        B = np.concatenate((W_B, R_B), axis=1)

        gru = GRU_Helper(X=input, W=W, R=R, B=B, hidden_size=hidden_size)
        _, Y_h = gru.step()
        expect(node, inputs=[input, W, R, B], outputs=[Y_h.astype(np.float32)], name='test_gru_seq_length')

    @staticmethod
    def export_initial_h():  # type: () -> None
        input = np.array([[[1., 2.], [3., 4.], [5., 6.]]]).astype(np.float32)

        input_size = 2
        batch_size = 3
        hidden_size = 4
        weight_scale = 0.1
        number_of_gates = 3

        node = onnx.helper.make_node(
            'GRU',
            inputs=['X', 'W', 'R', '', '', 'initial_h'],
            outputs=['', 'Y_h'],
            hidden_size=hidden_size
        )

        W = weight_scale * np.ones((1, number_of_gates * hidden_size, input_size)).astype(np.float32)
        R = weight_scale * np.ones((1, number_of_gates * hidden_size, hidden_size)).astype(np.float32)
        initial_h = np.ones((1, batch_size, hidden_size)).astype(np.float32)

        rnn = GRU_Helper(X=input, W=W, R=R, initial_h=initial_h, hidden_size=hidden_size)
        _, Y_h = rnn.step()
        expect(node, inputs=[input, W, R, initial_h], outputs=[Y_h.astype(np.float32)],
               name='test_gru_with_initial_h')

    @staticmethod
    def export_intermediate_h():  # type: () -> None
        input = np.array([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
                          [[10., 11., 12.], [13., 14., 15.], [16., 17., 18.]]]).astype(np.float32)

        input_size = 3
        hidden_size = 4
        weight_scale = 0.1
        number_of_gates = 3

        node = onnx.helper.make_node(
            'GRU',
            inputs=['X', 'W', 'R'],
            outputs=['Y', ''],
            hidden_size=hidden_size
        )

        W = weight_scale * np.ones((1, number_of_gates * hidden_size, input_size)).astype(np.float32)
        R = weight_scale * np.ones((1, number_of_gates * hidden_size, hidden_size)).astype(np.float32)

        rnn = GRU_Helper(X=input, W=W, R=R, hidden_size=hidden_size)
        Y, _ = rnn.step()
        expect(node, inputs=[input, W, R], outputs=[Y.astype(np.float32)],
               name='test_gru_intermediate_h')

    @staticmethod
    def export_both_outputs():  # type: () -> None
        input = np.array([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
                          [[10., 11., 12.], [13., 14., 15.], [16., 17., 18.]]]).astype(np.float32)

        input_size = 3
        hidden_size = 4
        weight_scale = 0.1
        number_of_gates = 3

        node = onnx.helper.make_node(
            'GRU',
            inputs=['X', 'W', 'R'],
            outputs=['Y', 'Y_h'],
            hidden_size=hidden_size
        )

        W = weight_scale * np.ones((1, number_of_gates * hidden_size, input_size)).astype(np.float32)
        R = weight_scale * np.ones((1, number_of_gates * hidden_size, hidden_size)).astype(np.float32)

        rnn = GRU_Helper(X=input, W=W, R=R, hidden_size=hidden_size)
        Y, Y_h = rnn.step()
        expect(node, inputs=[input, W, R], outputs=[Y.astype(np.float32), Y_h.astype(np.float32)],
               name='test_gru_both_outputs')

    @staticmethod
    def export_reverse():  # type: () -> None
        input = np.array([[[1., 2.], [3., 4.], [5., 6.]]]).astype(np.float32)

        input_size = 2
        hidden_size = 4
        weight_scale = 0.1
        direction = 'reverse'
        number_of_gates = 3

        node = onnx.helper.make_node(
            'GRU',
            inputs=['X', 'W', 'R'],
            outputs=['', 'Y_h'],
            hidden_size=hidden_size,
            direction=direction
        )

        W = weight_scale * np.ones((1, number_of_gates * hidden_size, input_size)).astype(np.float32)
        R = weight_scale * np.ones((1, number_of_gates * hidden_size, hidden_size)).astype(np.float32)

        rnn = GRU_Helper(X=input, W=W, R=R, hidden_size=hidden_size, direction=direction)
        _, Y_h = rnn.step()
        expect(node, inputs=[input, W, R], outputs=[Y_h.astype(np.float32)],
               name='test_gru_reverse')

    @staticmethod
    def export_bidirectional():  # type: () -> None
        input = np.array([[[1., 2.], [3., 4.], [5., 6.]]]).astype(np.float32)

        input_size = 2
        hidden_size = 4
        weight_scale = 0.1
        direction = 'bidirectional'
        number_of_gates = 3

        node = onnx.helper.make_node(
            'GRU',
            inputs=['X', 'W', 'R'],
            outputs=['', 'Y_h'],
            hidden_size=hidden_size,
            direction=direction
        )

        W = weight_scale * np.ones((2, number_of_gates * hidden_size, input_size)).astype(np.float32)
        R = weight_scale * np.ones((2, number_of_gates * hidden_size, hidden_size)).astype(np.float32)

        rnn = GRU_Helper(X=input, W=W, R=R, hidden_size=hidden_size, direction=direction)
        _, Y_h = rnn.step()
        expect(node, inputs=[input, W, R], outputs=[Y_h.astype(np.float32)],
               name='test_gru_bidirectional')

    @staticmethod
    def export_clip():  # type: () -> None
        input = np.array([[[1., 2.], [3., 4.], [5., 6.]]]).astype(np.float32)

        input_size = 2
        hidden_size = 4
        weight_scale = 0.1
        clip = 0.1
        number_of_gates = 3

        node = onnx.helper.make_node(
            'GRU',
            inputs=['X', 'W', 'R'],
            outputs=['', 'Y_h'],
            hidden_size=hidden_size,
            clip=clip
        )

        W = weight_scale * np.ones((1, number_of_gates * hidden_size, input_size)).astype(np.float32)
        R = weight_scale * np.ones((1, number_of_gates * hidden_size, hidden_size)).astype(np.float32)

        rnn = GRU_Helper(X=input, W=W, R=R, hidden_size=hidden_size, clip=clip)
        _, Y_h = rnn.step()
        expect(node, inputs=[input, W, R], outputs=[Y_h.astype(np.float32)], name='test_gru_clip')

    @staticmethod
    def export_activations():  # type: () -> None
        input = np.array([[[1., 2.], [3., 4.], [5., 6.]]]).astype(np.float32)

        input_size = 2
        hidden_size = 4
        weight_scale = 0.1
        activations = ["Relu"]
        number_of_gates = 3

        node = onnx.helper.make_node(
            'GRU',
            inputs=['X', 'W', 'R'],
            outputs=['', 'Y_h'],
            hidden_size=hidden_size,
            activations=activations
        )

        W = weight_scale * np.ones((1, number_of_gates * hidden_size, input_size)).astype(np.float32)
        R = weight_scale * np.ones((1, number_of_gates * hidden_size, hidden_size)).astype(np.float32)

        rnn = GRU_Helper(X=input, W=W, R=R, hidden_size=hidden_size, activations=activations)
        _, Y_h = rnn.step()
        expect(node, inputs=[input, W, R], outputs=[Y_h.astype(np.float32)], name='test_gru_activations')
