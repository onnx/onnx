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

        # LSTM Attr Names
        ACTS = str('activations')
        CLIP = str('clip')
        DIR = str('direction')
        H_SIZE = str('hidden_size')

        required_attrs = [H_SIZE]

        for i in required_attrs:
            assert i in params, "Missing attribute: {0}".format(i)

        number_of_gates = 4
        number_of_peepholes = 3
        self.hidden_size = params[H_SIZE]
        self.direction = params[DIR] if DIR in params else 'forward'
        self.num_directions = 2 if self.direction == 'bidirectional' else 1

        activation_lookup = {'Relu': lambda x: np.maximum(x, 0), 'Tanh': np.tanh, 'Sigmoid': lambda x: 1 / (1 + np.exp(-x))}
        f_activations = params[ACTS][:self.num_directions] if ACTS in params else ['Sigmoid'] * self.num_directions
        self.f = [activation_lookup[act] for act in f_activations if act in activation_lookup]
        g_activations = params[ACTS][self.num_directions: 2 * self.num_directions] if ACTS in params else ['Tanh'] * self.num_directions
        self.g = [activation_lookup[act] for act in g_activations if act in activation_lookup]
        h_activations = params[ACTS][2 * self.num_directions:] if ACTS in params else ['Tanh'] * self.num_directions
        self.h = [activation_lookup[act] for act in h_activations if act in activation_lookup]

        self.B = params[B] if B in params else np.zeros((self.num_directions, 2 * number_of_gates * self.hidden_size), dtype=np.float32)
        self.P = params[P] if P in params else np.zeros((self.num_directions, number_of_peepholes * self.hidden_size), dtype=np.float32)

        self.clip = params[CLIP] if CLIP in params else np.inf

        self.batch_size = params[X].shape[1]
        self.H_0 = params[H_0] if H_0 in params else np.zeros((self.num_directions, self.batch_size, self.hidden_size), dtype=np.float32)
        self.C_0 = params[C_0] if C_0 in params else np.zeros((self.num_directions, self.batch_size, self.hidden_size), dtype=np.float32)

        self.X = params[X]
        self.W = params[W]
        self.R = params[R]

    def step(self):  # type: () -> Tuple[np.ndarray, np.ndarray]
        seq_length = self.X.shape[0]
        Y = np.empty([seq_length, self.num_directions, self.batch_size, self.hidden_size])
        Y_c = np.empty([self.num_directions, self.batch_size, self.hidden_size])

        xs = np.split(self.X, self.X.shape[0], axis=0)

        dirs = {'forward', 'reverse'} if self.direction == 'bidirectional' else {self.direction}
        for idx, direction in enumerate(dirs):
            [p_i, p_o, p_f] = np.split(self.P[idx], 3)

            h_list = []
            H_t = self.H_0[idx]
            C_t = self.C_0[idx]

            if direction == 'reverse':
                xs = xs[::-1]

            for x in xs:
                gates = np.dot(x, np.transpose(self.W[idx])) + np.dot(H_t, np.transpose(self.R[idx])) + np.add(
                    *np.split(self.B[idx], 2))
                i, o, f, c = np.split(gates, 4, -1)
                i = self.f[idx](np.clip(i + p_i * C_t, -self.clip, self.clip))
                f = self.f[idx](np.clip(f + p_f * C_t, -self.clip, self.clip))
                c = self.g[idx](np.clip(c, -self.clip, self.clip))
                C = f * C_t + i * c
                o = self.f[idx](np.clip(o + p_o * C, -self.clip, self.clip))
                H = o * self.h[idx](np.clip(C, -self.clip, self.clip))

                h_list.append(H)
                H_t = H
                C_t = C

            if direction == 'reverse':
                h_list = h_list[::-1]

            concatenated = np.concatenate(h_list)
            Y[:, idx] = concatenated
            Y_c[idx] = C_t

        return Y, Y[-1], Y_c


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
            outputs=['', 'Y_h'],
            hidden_size=hidden_size
        )

        W = weight_scale * np.ones((1, number_of_gates * hidden_size, input_size)).astype(np.float32)
        R = weight_scale * np.ones((1, number_of_gates * hidden_size, hidden_size)).astype(np.float32)

        lstm = LSTM_Helper(X=input, W=W, R=R, hidden_size=hidden_size)
        _, Y_h, _ = lstm.step()
        expect(node, inputs=[input, W, R], outputs=[Y_h.astype(np.float32)], name='test_lstm_defaults')

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
            outputs=['', 'Y_h'],
            hidden_size=hidden_size
        )

        W = weight_scale * np.ones((1, number_of_gates * hidden_size, input_size)).astype(np.float32)
        R = weight_scale * np.ones((1, number_of_gates * hidden_size, hidden_size)).astype(np.float32)

        # Adding custom bias
        W_B = custom_bias * np.ones((1, number_of_gates * hidden_size)).astype(np.float32)
        R_B = np.zeros((1, number_of_gates * hidden_size)).astype(np.float32)
        B = np.concatenate((W_B, R_B), 1)

        lstm = LSTM_Helper(X=input, W=W, R=R, B=B, hidden_size=hidden_size)
        _, Y_h, _ = lstm.step()
        expect(node, inputs=[input, W, R, B], outputs=[Y_h.astype(np.float32)], name='test_lstm_with_initial_bias')

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
            outputs=['', 'Y_h'],
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

        lstm = LSTM_Helper(X=input, W=W, R=R, B=B, P=P, initial_c=init_c, initial_h=init_h, hidden_size=hidden_size)
        _, Y_h, _ = lstm.step()
        expect(node, inputs=[input, W, R, B, seq_lens, init_h, init_c, P], outputs=[Y_h.astype(np.float32)],
               name='test_lstm_with_peepholes')

    @staticmethod
    def export_seq_length():  # type: () -> None
        input = np.array([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
                          [[10., 11., 12.], [13., 14., 15.], [16., 17., 18.]]]).astype(np.float32)

        input_size = 3
        hidden_size = 5
        number_of_gates = 4

        node = onnx.helper.make_node(
            'LSTM',
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

        lstm = LSTM_Helper(X=input, W=W, R=R, B=B, hidden_size=hidden_size)
        _, Y_h, _ = lstm.step()
        expect(node, inputs=[input, W, R, B], outputs=[Y_h.astype(np.float32)], name='test_lstm_seq_length')

    @staticmethod
    def export_initial_h():  # type: () -> None
        input = np.array([[[1., 2.], [3., 4.], [5., 6.]]]).astype(np.float32)

        input_size = 2
        batch_size = 3
        hidden_size = 4
        weight_scale = 0.1
        number_of_gates = 4

        node = onnx.helper.make_node(
            'LSTM',
            inputs=['X', 'W', 'R', '', '', 'initial_h'],
            outputs=['', 'Y_h'],
            hidden_size=hidden_size
        )

        W = weight_scale * np.ones((1, number_of_gates * hidden_size, input_size)).astype(np.float32)
        R = weight_scale * np.ones((1, number_of_gates * hidden_size, hidden_size)).astype(np.float32)
        initial_h = np.ones((1, batch_size, hidden_size)).astype(np.float32)

        lstm = LSTM_Helper(X=input, W=W, R=R, initial_h=initial_h, hidden_size=hidden_size)
        _, Y_h, _ = lstm.step()
        expect(node, inputs=[input, W, R, initial_h], outputs=[Y_h.astype(np.float32)],
               name='test_lstm_with_initial_h')

    @staticmethod
    def export_initial_c():  # type: () -> None
        input = np.array([[[1., 2.], [3., 4.], [5., 6.]]]).astype(np.float32)

        input_size = 2
        batch_size = 3
        hidden_size = 4
        weight_scale = 0.1
        number_of_gates = 4

        node = onnx.helper.make_node(
            'LSTM',
            inputs=['X', 'W', 'R', '', '', '', 'initial_c'],
            outputs=['', 'Y_h'],
            hidden_size=hidden_size
        )

        W = weight_scale * np.ones((1, number_of_gates * hidden_size, input_size)).astype(np.float32)
        R = weight_scale * np.ones((1, number_of_gates * hidden_size, hidden_size)).astype(np.float32)
        initial_c = np.ones((1, batch_size, hidden_size)).astype(np.float32)

        lstm = LSTM_Helper(X=input, W=W, R=R, initial_c=initial_c, hidden_size=hidden_size)
        _, Y_h, _ = lstm.step()
        expect(node, inputs=[input, W, R, initial_c], outputs=[Y_h.astype(np.float32)],
               name='test_lstm_with_initial_c')

    @staticmethod
    def export_initial_h_and_c():  # type: () -> None
        input = np.array([[[1., 2.], [3., 4.], [5., 6.]]]).astype(np.float32)

        input_size = 2
        batch_size = 3
        hidden_size = 4
        weight_scale = 0.1
        number_of_gates = 4

        node = onnx.helper.make_node(
            'LSTM',
            inputs=['X', 'W', 'R', '', '', 'initial_h', 'initial_c'],
            outputs=['', 'Y_h'],
            hidden_size=hidden_size
        )

        W = weight_scale * np.ones((1, number_of_gates * hidden_size, input_size)).astype(np.float32)
        R = weight_scale * np.ones((1, number_of_gates * hidden_size, hidden_size)).astype(np.float32)
        initial_h = np.ones((1, batch_size, hidden_size)).astype(np.float32)
        initial_c = np.ones((1, batch_size, hidden_size)).astype(np.float32)

        lstm = LSTM_Helper(X=input, W=W, R=R, initial_h=initial_h, initial_c=initial_c, hidden_size=hidden_size)
        _, Y_h, _ = lstm.step()
        expect(node, inputs=[input, W, R, initial_h, initial_c], outputs=[Y_h.astype(np.float32)],
               name='test_lstm_with_initial_h_and_c')

    @staticmethod
    def export_intermediate_h():  # type: () -> None
        input = np.array([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
                          [[10., 11., 12.], [13., 14., 15.], [16., 17., 18.]]]).astype(np.float32)

        input_size = 3
        hidden_size = 4
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

        lstm = LSTM_Helper(X=input, W=W, R=R, hidden_size=hidden_size)
        Y, _, _ = lstm.step()
        expect(node, inputs=[input, W, R], outputs=[Y.astype(np.float32)],
               name='test_lstm_intermediate_h')

    @staticmethod
    def export_all_y_c():  # type: () -> None
        input = np.array([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
                          [[10., 11., 12.], [13., 14., 15.], [16., 17., 18.]]]).astype(np.float32)

        input_size = 3
        hidden_size = 4
        weight_scale = 0.1
        number_of_gates = 4

        node = onnx.helper.make_node(
            'LSTM',
            inputs=['X', 'W', 'R'],
            outputs=['Y_h', '', 'Y_c'],
            hidden_size=hidden_size
        )

        W = weight_scale * np.ones((1, number_of_gates * hidden_size, input_size)).astype(np.float32)
        R = weight_scale * np.ones((1, number_of_gates * hidden_size, hidden_size)).astype(np.float32)

        lstm = LSTM_Helper(X=input, W=W, R=R, hidden_size=hidden_size)
        Y_h, _, Y_c = lstm.step()
        expect(node, inputs=[input, W, R], outputs=[Y_h.astype(np.float32), Y_c.astype(np.float32)], name='test_lstm_y_c')

    @staticmethod
    def export_all_outputs():  # type: () -> None
        input = np.array([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
                          [[10., 11., 12.], [13., 14., 15.], [16., 17., 18.]]]).astype(np.float32)

        input_size = 3
        hidden_size = 4
        weight_scale = 0.1
        number_of_gates = 4

        node = onnx.helper.make_node(
            'LSTM',
            inputs=['X', 'W', 'R'],
            outputs=['Y', 'Y_h', 'Y_c'],
            hidden_size=hidden_size
        )

        W = weight_scale * np.ones((1, number_of_gates * hidden_size, input_size)).astype(np.float32)
        R = weight_scale * np.ones((1, number_of_gates * hidden_size, hidden_size)).astype(np.float32)

        lstm = LSTM_Helper(X=input, W=W, R=R, hidden_size=hidden_size)
        Y, Y_h, Y_c = lstm.step()
        expect(node, inputs=[input, W, R], outputs=[Y.astype(np.float32), Y_h.astype(np.float32),
               Y_c.astype(np.float32)], name='test_lstm_all_outputs')

    @staticmethod
    def export_reverse():  # type: () -> None
        input = np.array([[[1., 2.], [3., 4.], [5., 6.]]]).astype(np.float32)

        input_size = 2
        hidden_size = 4
        weight_scale = 0.1
        direction = 'reverse'
        number_of_gates = 4

        node = onnx.helper.make_node(
            'LSTM',
            inputs=['X', 'W', 'R'],
            outputs=['', 'Y_h'],
            hidden_size=hidden_size,
            direction=direction
        )

        W = weight_scale * np.ones((1, number_of_gates * hidden_size, input_size)).astype(np.float32)
        R = weight_scale * np.ones((1, number_of_gates * hidden_size, hidden_size)).astype(np.float32)

        lstm = LSTM_Helper(X=input, W=W, R=R, hidden_size=hidden_size, direction=direction)
        _, Y_h, _ = lstm.step()
        expect(node, inputs=[input, W, R], outputs=[Y_h.astype(np.float32)],
               name='test_lstm_reverse')

    @staticmethod
    def export_bidirectional():  # type: () -> None
        input = np.array([[[1., 2.], [3., 4.], [5., 6.]]]).astype(np.float32)

        input_size = 2
        hidden_size = 4
        weight_scale = 0.1
        direction = 'bidirectional'
        number_of_gates = 4

        node = onnx.helper.make_node(
            'LSTM',
            inputs=['X', 'W', 'R'],
            outputs=['', 'Y_h'],
            hidden_size=hidden_size,
            direction=direction
        )

        W = weight_scale * np.ones((2, number_of_gates * hidden_size, input_size)).astype(np.float32)
        R = weight_scale * np.ones((2, number_of_gates * hidden_size, hidden_size)).astype(np.float32)

        lstm = LSTM_Helper(X=input, W=W, R=R, hidden_size=hidden_size, direction=direction)
        _, Y_h, _ = lstm.step()
        expect(node, inputs=[input, W, R], outputs=[Y_h.astype(np.float32)],
               name='test_lstm_bidirectional')

    @staticmethod
    def export_clip():  # type: () -> None
        input = np.array([[[1., 2.], [3., 4.], [5., 6.]]]).astype(np.float32)

        input_size = 2
        hidden_size = 4
        weight_scale = 0.1
        clip = 0.1
        number_of_gates = 4

        node = onnx.helper.make_node(
            'LSTM',
            inputs=['X', 'W', 'R'],
            outputs=['', 'Y_h'],
            hidden_size=hidden_size,
            clip=clip
        )

        W = weight_scale * np.ones((1, number_of_gates * hidden_size, input_size)).astype(np.float32)
        R = weight_scale * np.ones((1, number_of_gates * hidden_size, hidden_size)).astype(np.float32)

        lstm = LSTM_Helper(X=input, W=W, R=R, hidden_size=hidden_size, clip=clip)
        _, Y_h, _ = lstm.step()
        expect(node, inputs=[input, W, R], outputs=[Y_h.astype(np.float32)], name='test_lstm_clip')

    @staticmethod
    def export_activations():  # type: () -> None
        input = np.array([[[1., 2.], [3., 4.], [5., 6.]]]).astype(np.float32)

        input_size = 2
        hidden_size = 4
        weight_scale = 0.1
        activations = ["Relu", "Relu", "Relu"]
        number_of_gates = 4

        node = onnx.helper.make_node(
            'LSTM',
            inputs=['X', 'W', 'R'],
            outputs=['', 'Y_h'],
            hidden_size=hidden_size,
            activations=activations
        )

        W = weight_scale * np.ones((1, number_of_gates * hidden_size, input_size)).astype(np.float32)
        R = weight_scale * np.ones((1, number_of_gates * hidden_size, hidden_size)).astype(np.float32)

        lstm = LSTM_Helper(X=input, W=W, R=R, hidden_size=hidden_size, activations=activations)
        _, Y_h, _ = lstm.step()
        expect(node, inputs=[input, W, R], outputs=[Y_h.astype(np.float32)], name='test_lstm_activations')
