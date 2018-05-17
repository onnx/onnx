from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class LSTM_Helper():
    def __init__(self, **params):
        # LSTM Input Names
        X = 'X'
        W = 'W'
        R = 'R'
        B = 'B'
        H_0 = 'initial_h'
        C_0 = 'initial_c'
        P = 'P'
        number_of_gates = 4
        number_of_peepholes = 3

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

            b = params[B] if B in params else np.zeros(2 * number_of_gates * hidden_size, dtype=np.float32)
            p = params[P] if P in params else np.zeros(number_of_peepholes * hidden_size, dtype=np.float32)
            h_0 = params[H_0] if H_0 in params else np.zeros((batch_size, hidden_size), dtype=np.float32)
            c_0 = params[C_0] if C_0 in params else np.zeros((batch_size, hidden_size), dtype=np.float32)

            self.X = params[X]
            self.W = params[W]
            self.R = params[R]
            self.B = b
            self.P = p
            self.H_0 = h_0
            self.C_0 = c_0
        else:
            raise NotImplementedError()

    def f(self, x):
        return 1 / (1 + np.exp(-x))

    def g(self, x):
        return np.tanh(x)

    def h(self, x):
        return np.tanh(x)

    def step(self):
        [p_i, p_o, p_f] = np.split(self.P, 3)
        h_list = []
        H_t = self.H_0
        C_t = self.C_0
        for x in np.split(self.X, self.X.shape[0], axis=0):
            gates = np.dot(x, np.transpose(self.W)) + np.dot(H_t, np.transpose(self.R)) + np.add(
                *np.split(self.B, 2))
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
        return output


class LSTM(Base):

    @staticmethod
    def export_defaults():
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
    def export_initial_bias():
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
    def export_peepholes():
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
        expect(node, inputs=[input, W, R, B, seq_lens, init_h, init_c, P], outputs=[output],
               name='test_lstm_with_peepholes')

    @staticmethod
    def export_precomputed():
        hidden_size = 5
        node = onnx.helper.make_node(
            'LSTM',
            inputs=['X', 'W', 'R', 'B'],
            outputs=['Y'],
            hidden_size=hidden_size
        )

        x = np.array([[[-0.2092975, 1.0150607, 0.5475235],
                       [-0.78951997, -0.3851825, -1.058371],
                       [-0.79346454, 1.440994, 0.11596155],
                       [1.3323077, -1.4334958, 0.67276776]],

                      [[-0.20995837, 0.78509414, 1.1314468],
                       [0.09848047, 0.7878104, -0.8595762],
                       [-0.68734163, 0.09055053, -0.08572508],
                       [-0.5343582, 0.78376126, 1.6711195]]]).astype(np.float32)
        w = np.array([[[-0.10433397, -0.32943147, 0.28882498],
                       [-0.01248327, 0.00561786, -0.15700299],
                       [0.19504583, -0.2958504, 0.217354],
                       [0.25214392, 0.16586196, -0.42395538],
                       [-0.3517861, -0.28040624, -0.13271871],
                       [-0.44714275, 0.26895386, 0.23709434],
                       [-0.23318909, -0.07799888, 0.1595177],
                       [-0.2150556, 0.09592342, 0.36917347],
                       [-0.4135554, 0.3642367, 0.2788254],
                       [0.39635754, -0.30597758, -0.43236837],
                       [-0.05483797, 0.34892327, 0.19764519],
                       [0.3905447, 0.3851717, 0.22790933],
                       [-0.41812345, -0.3711061, 0.09412122],
                       [0.07832491, 0.24423867, -0.06567389],
                       [-0.4092345, 0.02651429, 0.00358161],
                       [-0.287251, -0.20949665, 0.06729645],
                       [0.17051661, -0.4089965, -0.08853745],
                       [-0.02344924, 0.0769614, -0.10114089],
                       [0.08732384, -0.13031968, 0.3693238],
                       [0.09512341, 0.39322734, 0.05026317]]]).astype(np.float32)
        r = np.array([[[-0.06249112, -0.3027284, -0.2714152, 0.38169748, -0.3848986],
                       [-0.3015067, -0.11511135, -0.21178144, 0.36077315, -0.15175757],
                       [-0.3106271, -0.13619286, 0.09167248, -0.00270995, 0.16715324],
                       [0.3096549, 0.06414723, 0.27308917, -0.36815444, -0.25398546],
                       [0.40350074, 0.0977748, 0.16108459, -0.39921242, 0.03195441],
                       [0.32862294, 0.43853426, 0.21549404, 0.43726218, -0.12468177],
                       [0.17379999, -0.17711008, -0.02272931, 0.15821773, -0.2360906],
                       [0.04289997, 0.14505583, 0.09238386, 0.15120435, 0.22830069],
                       [0.03189701, -0.12833786, 0.34228635, 0.17734432, 0.3900627],
                       [-0.15498602, 0.05439121, -0.33910578, 0.02097705, -0.42195255],
                       [0.02405897, -0.29173693, 0.28562587, 0.39553738, 0.3302138],
                       [0.08506113, 0.17034757, 0.39796853, 0.08288926, -0.15668085],
                       [0.28090197, -0.07625446, -0.1972777, -0.18111584, -0.30605704],
                       [0.31646246, -0.17583266, -0.30067247, 0.36228096, 0.2409823],
                       [0.41217238, -0.41253033, 0.4323389, -0.43837905, -0.4471018],
                       [0.33483976, -0.09845337, -0.41000995, -0.02262691, -0.25162753],
                       [-0.38541794, -0.16338113, 0.13963765, 0.2420575, 0.01179534],
                       [0.33970326, 0.16575891, -0.0950945, -0.43333504, -0.12386471],
                       [0.22664595, 0.3502648, -0.3177863, 0.10857916, -0.06070068],
                       [0.03233594, 0.15320832, 0.390822, 0.27357525, 0.05000198]]]).astype(np.float32)
        b_w = np.array([[-0.07438096, 0.0116722, 0.25055438, -0.19189253, -0.31282502,
                         -0.40339673, 0.41707057, -0.10233101, 0.0879212, 0.41836756,
                         -0.36332014, -0.41120952, -0.14621562, -0.17775917, -0.06715041,
                         -0.3470294, -0.2873521, 0.280366, -0.13346279, 0.34218425]]).astype(np.float32)
        b_r = np.array([[-0.03111535, -0.33301952, -0.17032543, -0.05586442, -0.32023185,
                         -0.3442515, 0.44716895, -0.2340269, 0.27279443, -0.32590097,
                         -0.07209095, 0.19784379, 0.27797413, 0.00461715, -0.36984265,
                         -0.07255584, -0.12558547, -0.11320454, -0.35173178, -0.13761121]]).astype(np.float32)
        b = np.concatenate((b_w, b_r), axis=1)

        lstm = LSTM_Helper(X=x, W=w, R=r, B=b)
        output = lstm.step()

        output_precomputed = np.array([[[[-0.09183919, -0.20181324, 0.04475278, -0.12022446,
                                          0.05694127],
                                         [-0.02611334, -0.09452314, 0.04121872, -0.18716583,
                                          -0.02087192],
                                         [-0.08711052, -0.23869304, 0.05338074, -0.19986068,
                                          0.06211524],
                                         [-0.04351437, 0.08689842, -0.01133228, 0.00968763,
                                          -0.04177252]]],

                                       [[[-0.1343966, -0.28931525, 0.05130501, -0.13485222,
                                          0.063078],
                                         [-0.0660032, -0.20069027, 0.06826789, -0.31976634,
                                          0.06613372],
                                         [-0.06817593, -0.23132755, 0.07001564, -0.23856725,
                                          0.06021666],
                                         [-0.1600846, -0.15111914, 0.01314638, 0.01177528,
                                          0.03054012]]]]).astype(np.float32)
        assert np.allclose(output, output_precomputed)

        expect(node, inputs=[x, w, r, b], outputs=[output], name='test_lstm_precomputed')
