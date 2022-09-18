# SPDX-License-Identifier: Apache-2.0
# pylint: disable=R0913,W0221

import numpy  # type: ignore

from ...defs import onnx_opset_version
from ..op_run import OpRun


class CommonRNN(OpRun):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRun.__init__(self, onnx_node, run_params)

        if self.direction in ("forward", "reverse"):  # type: ignore
            self.num_directions = 1  # type: ignore
        elif self.direction == "bidirectional":  # type: ignore
            self.num_directions = 2  # type: ignore
        else:
            raise RuntimeError(f"Unknown direction {self.direction!r}.")  # type: ignore

        if (
            self.activation_alpha is not None  # type: ignore
            and len(self.activation_alpha) != self.num_directions  # type: ignore
        ):
            raise RuntimeError(
                f"activation_alpha must have the same size as num_directions={self.num_directions}."  # type: ignore
            )
        if (
            self.activation_beta is not None  # type: ignore
            and len(self.activation_beta) != self.num_directions  # type: ignore
        ):
            raise RuntimeError(
                f"activation_beta must have the same size as num_directions={self.num_directions}."  # type: ignore
            )

        self.f1 = self.choose_act(
            self.activations[0],  # type: ignore
            self.activation_alpha[0]  # type: ignore
            if self.activation_alpha is not None and len(self.activation_alpha) > 0  # type: ignore
            else None,
            self.activation_beta[0]  # type: ignore
            if self.activation_beta is not None and len(self.activation_beta) > 0  # type: ignore
            else None,
        )
        if len(self.activations) > 1:  # type: ignore
            self.f2 = self.choose_act(
                self.activations[1],  # type: ignore
                self.activation_alpha[1]  # type: ignore
                if self.activation_alpha is not None and len(self.activation_alpha) > 1  # type: ignore
                else None,
                self.activation_beta[1]  # type: ignore
                if self.activation_beta is not None and len(self.activation_beta) > 1  # type: ignore
                else None,
            )
        self.n_outputs = len(onnx_node.output)

    def choose_act(self, name, alpha, beta):  # type: ignore
        if name in ("Tanh", "tanh"):
            return self._f_tanh
        if name in ("Affine", "affine"):
            return lambda x: x * alpha + beta
        raise RuntimeError(f"Unknown activation function {name!r}.")

    def _f_tanh(self, x):  # type: ignore
        return numpy.tanh(x)

    def _step(self, X, R, B, W, H_0):  # type: ignore
        h_list = []
        H_t = H_0
        for x in numpy.split(X, X.shape[0], axis=0):
            H = self.f1(
                numpy.dot(x, numpy.transpose(W))
                + numpy.dot(H_t, numpy.transpose(R))
                + numpy.add(*numpy.split(B, 2))
            )
            h_list.append(H)
            H_t = H
        concatenated = numpy.concatenate(h_list)
        if self.num_directions == 1:
            output = numpy.expand_dims(concatenated, 1)
        return output, h_list[-1]

    def _run(self, X, W, R, B=None, sequence_lens=None, initial_h=None):  # type: ignore
        # TODO: support overridden attributes.
        self.num_directions = W.shape[0]

        if self.num_directions == 1:
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

            b = B if B is not None else numpy.zeros(2 * hidden_size, dtype=X.dtype)
            h_0 = (
                initial_h
                if initial_h is not None
                else numpy.zeros((batch_size, hidden_size), dtype=X.dtype)
            )

            B = b
            H_0 = h_0
        else:
            raise NotImplementedError(
                f"Unsupported value {self.num_directions} for num_directions and operator {self.__class__.__name__!r}."
            )

        Y, Y_h = self._step(X, R, B, W, H_0)
        # if self.layout == 1:
        #    #Y = numpy.transpose(Y, [2, 0, 1, 3])
        #    Y_h = Y[:, :, -1, :]

        return (Y,) if self.n_outputs == 1 else (Y, Y_h)


class RNN_7(CommonRNN):

    pass


class RNN_14(CommonRNN):

    pass


if onnx_opset_version() >= 14:
    RNN = RNN_14
else:
    RNN = RNN_7  # type: ignore
