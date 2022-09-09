# SPDX-License-Identifier: Apache-2.0
# pylint: disable=R0913,R0914,W0221

import numpy  # type: ignore

from ..op_run import OpRun


class CommonGRU(OpRun):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRun.__init__(self, onnx_node, run_params)
        self.n_outputs = len(onnx_node.output)
        self.number_of_gates = 3

    def f(self, x):  # type: ignore
        return 1 / (1 + numpy.exp(-x))

    def g(self, x):  # type: ignore
        return numpy.tanh(x)

    def _step(self, X, R, B, W, H_0, num_directions):  # type: ignore
        seq_length = X.shape[0]
        hidden_size = H_0.shape[-1]
        batch_size = X.shape[1]

        Y = numpy.empty([seq_length, num_directions, batch_size, hidden_size])
        h_list = []

        [w_z, w_r, w_h] = numpy.split(W, 3)
        [r_z, r_r, r_h] = numpy.split(R, 3)
        [w_bz, w_br, w_bh, r_bz, r_br, r_bh] = numpy.split(B, 6)
        gates_w = numpy.transpose(numpy.concatenate((w_z, w_r)))
        gates_r = numpy.transpose(numpy.concatenate((r_z, r_r)))
        gates_b = numpy.add(
            numpy.concatenate((w_bz, w_br)), numpy.concatenate((r_bz, r_br))
        )

        H_t = H_0
        for x in numpy.split(X, X.shape[0], axis=0):
            gates = numpy.dot(x, gates_w) + numpy.dot(H_t, gates_r) + gates_b
            z, r = numpy.split(gates, 2, -1)  # pylint: disable=W0632
            z = self.f(z)
            r = self.f(r)
            h_default = self.g(
                numpy.dot(x, numpy.transpose(w_h))
                + numpy.dot(r * H_t, numpy.transpose(r_h))
                + w_bh
                + r_bh
            )
            h_linear = self.g(
                numpy.dot(x, numpy.transpose(w_h))
                + r * (numpy.dot(H_t, numpy.transpose(r_h)) + r_bh)
                + w_bh
            )
            h = h_linear if self.linear_before_reset else h_default  # type: ignore
            H = (1 - z) * h + z * H_t
            h_list.append(H)
            H_t = H

        concatenated = numpy.concatenate(h_list)
        if num_directions == 1:
            Y[:, 0, :, :] = concatenated

        if self.layout == 0:  # type: ignore
            Y_h = Y[-1]
        else:
            Y = numpy.transpose(Y, [2, 0, 1, 3])
            Y_h = Y[:, :, -1, :]

        return Y, Y_h

    def _run(self, X, W, R, B=None, sequence_lens=None, initial_h=None):  # type: ignore
        num_directions = W.shape[0]

        if num_directions == 1:
            R = numpy.squeeze(R, axis=0)
            W = numpy.squeeze(W, axis=0)
            if B is not None:
                B = numpy.squeeze(B, axis=0)
            if sequence_lens is not None:
                sequence_lens = numpy.squeeze(sequence_lens, axis=0)
            if initial_h is not None:
                initial_h = numpy.squeeze(initial_h, axis=0)

            hidden_size = R.shape[-1]
            batch_size = X.shape[1]

            b = (
                B
                if B is not None
                else numpy.zeros(2 * self.number_of_gates * hidden_size, dtype=X.dtype)
            )
            h_0 = (
                initial_h
                if initial_h is not None
                else numpy.zeros((batch_size, hidden_size), dtype=X.dtype)
            )

            B = b
            H_0 = h_0
        else:
            raise NotImplementedError(
                f"Unsupported value {num_directions} for num_directions and operator "
                f"{self.__class__.__name__!r}."
            )

        Y, Y_h = self._step(X, R, B, W, H_0, num_directions=num_directions)

        return (Y,) if self.n_outputs == 1 else (Y, Y_h)


class GRU(CommonGRU):
    def __init__(self, onnx_node, run_params):  # type: ignore
        CommonGRU.__init__(self, onnx_node, run_params)  # type: ignore
