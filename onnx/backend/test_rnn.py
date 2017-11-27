# Provides reference implementation of CuDNN-style RNNs in numpy

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from collections import defaultdict
from itertools import chain

# TODO(dzhulgakov): these tests are not picked up any more and need to be fixed.

"""
import numpy as np
from six.moves import zip

from .test_util import N


def interleave(x, y):
    return list(chain.from_iterable(zip(x, y)))


class RNNBase(object):
    def param_size(self):
        return sum(p.size for p in self.all_params())

    def all_params(self):
        return self.all_weights() + self.all_biases()

    def concat_params(self):
        return np.concatenate(tuple(x.flatten() for x in self.all_params()))

    def feed_params(self, w):
        assert self.param_size() == w.size, \
            '%d vs %d' % (self.param_size(), w.size)
        offset = 0
        for x in self.all_params():
            x.ravel()[:] = w[offset:offset + x.size]
            offset += x.size
        return self


# TODO: add support for it in C2
def ReluCell(h, c, hh, ih):
    s = hh + ih
    return np.maximum(s[:, 0, :], 0), None


ReluCell.GATES = 1
ReluCell.HAS_CELL = False


def TanhCell(h, c, hh, ih):
    s = hh + ih
    return np.tanh(s[:, 0, :]), None


TanhCell.GATES = 1
TanhCell.HAS_CELL = False


def GRUCell(h, c, hh, ih):
    RESET = 0
    INPUT = 1
    HIDDEN = 2
    s = hh + ih
    s[:, [RESET, INPUT], :] = 1.0 / (1.0 + np.exp(-s[:, [RESET, INPUT], :]))
    q = np.tanh(s[:, RESET, :] * hh[:, HIDDEN, :] + ih[:, HIDDEN, :])
    return (1 - s[:, INPUT, :]) * q + s[:, INPUT, :] * h, None


GRUCell.GATES = 3
GRUCell.HAS_CELL = False


def LSTMCell(h, c, hh, ih):
    INPUT = 0
    FORGET = 1
    CELL = 2
    OUTPUT = 3
    s = hh + ih
    s[:, [INPUT, FORGET, OUTPUT], :] = 1.0 / (
        1.0 + np.exp(-s[:, [INPUT, FORGET, OUTPUT], :]))
    s[:, CELL, :] = np.tanh(s[:, CELL, :])
    c = s[:, FORGET, :] * c + s[:, INPUT, :] * s[:, CELL, :]
    h = s[:, OUTPUT, :] * np.tanh(c)
    return h, c


LSTMCell.GATES = 4
LSTMCell.HAS_CELL = True


class RNNRef(RNNBase):
    IH = 'ih'
    HH = 'hh'

    def __init__(self, cell, hidden_size, dim_in, skip=False):
        if skip:
            assert hidden_size == dim_in
        self.cell = cell
        self.hidden_size = hidden_size
        self.dim_in = dim_in
        self.weights = {}
        self.biases = {}
        if not skip:
            self.weights[self.IH] = np.random.randn(
                cell.GATES, self.hidden_size, self.dim_in).astype(np.float32)
        else:
            self.weights[self.IH] = np.empty(0)
        self.biases[self.IH] = np.random.randn(
            cell.GATES, self.hidden_size).astype(np.float32)
        self.weights[self.HH] = np.random.randn(
            cell.GATES, self.hidden_size, self.hidden_size).astype(np.float32)
        self.biases[self.HH] = np.random.randn(
            cell.GATES, self.hidden_size).astype(np.float32)

    def all_weights(self):
        return [self.weights[self.IH], self.weights[self.HH]]

    def all_biases(self):
        return [self.biases[self.IH], self.biases[self.HH]]

    def forward(self, X, h0=None, c0=None):
        assert self.cell.HAS_CELL or c0 is None
        n, b, input_size = X.shape
        assert input_size == self.dim_in
        if h0 is None:
            h0 = np.zeros((b, self.hidden_size))
        if c0 is None:
            c0 = np.zeros((b, self.hidden_size))

        out = np.zeros((n, b, self.hidden_size))
        h = h0
        c = c0 if self.cell.HAS_CELL else None
        for t in range(n):
            hh = np.dot(h, np.transpose(self.weights[self.HH],
                                        axes=(0, 2, 1))) + self.biases[self.HH]
            ih = self.biases[self.IH]
            if self.weights[self.IH].size:
                ih = ih + np.dot(
                    X[t], np.transpose(self.weights[self.IH], axes=(0, 2, 1)))
            else:
                ih = ih + X[t, :, np.newaxis, :]
            h, c = self.cell(h, c, hh, ih)
            out[t] = h
        if not self.cell.HAS_CELL:
            c = None
        return out, h, c


class RNNRefStack(RNNBase):
    def __init__(self, cell, hidden_size, dim_in, layers=1, skip=False):
        self.cell = cell
        self.hidden_size = hidden_size
        self.layers = [
            RNNRef(cell, hidden_size, dim_in
                   if l == 0 else hidden_size, skip and l == 0)
            for l in range(layers)
        ]

    def all_weights(self):
        return [x for l in self.layers for x in l.all_weights()]

    def all_biases(self):
        return [x for l in self.layers for x in l.all_biases()]

    def forward(self, X, h0=None, c0=None):
        assert self.cell.HAS_CELL or c0 is None
        _, b, _ = X.shape
        if h0 is None:
            h0 = np.zeros((len(self.layers), b, self.hidden_size))
        if c0 is None:
            c0 = np.zeros((len(self.layers), b, self.hidden_size))
        h = np.zeros(h0.shape)
        c = np.zeros(c0.shape)
        if not self.cell.HAS_CELL:
            c0 = c = defaultdict(lambda: None)
        for i, layer in enumerate(self.layers):
            X, h[i], c[i] = layer.forward(X, h0[i], c0[i])
        if not self.cell.HAS_CELL:
            c = None
        return X, h, c


class BiRNNRefStack(RNNBase):
    def __init__(self, cell, hidden_size, dim_in, layers=1, skip=False):
        self.cell = cell
        self.hidden_size = hidden_size
        self.layers = [
            RNNRef(cell, hidden_size, dim_in
                   if l == 0 else hidden_size * 2, skip and l == 0)
            for l in range(layers)
        ]
        self.back_layers = [
            RNNRef(cell, hidden_size, dim_in
                   if l == 0 else hidden_size * 2, skip and l == 0)
            for l in range(layers)
        ]

    def all_weights(self):
        return [
            x
            for l in interleave(self.layers, self.back_layers)
            for x in l.all_weights()
        ]

    def all_biases(self):
        return [
            x
            for l in interleave(self.layers, self.back_layers)
            for x in l.all_biases()
        ]

    def forward(self, X, h0=None, c0=None):
        assert self.cell.HAS_CELL or c0 is None
        _, b, _ = X.shape
        if h0 is None:
            h0 = np.zeros((len(self.layers) * 2, b, self.hidden_size))
        if c0 is None:
            c0 = np.zeros((len(self.layers) * 2, b, self.hidden_size))
        h = np.zeros(h0.shape)
        c = np.zeros(c0.shape)
        if not self.cell.HAS_CELL:
            c0 = c = defaultdict(lambda: None)
        for i, (layer,
                back_layer) in enumerate(zip(self.layers, self.back_layers)):
            f, h[i * 2], c[i * 2] = layer.forward(X, h0[i * 2], c0[i * 2])
            b, h[i * 2 + 1], c[i * 2 + 1] = back_layer.forward(
                X[::-1, :, :], h0[i * 2 + 1], c0[i * 2 + 1])
            X = np.concatenate((f, b[::-1, :, :]), axis=2)
        if not self.cell.HAS_CELL:
            c = None
        return X, h, c


def get_cell_class(cell_type):
    if cell_type == 'relu':
        return ReluCell
    elif cell_type == 'tanh':
        return TanhCell
    elif cell_type == 'lstm':
        return LSTMCell
    elif cell_type == 'gru':
        return GRUCell
    else:
        raise ValueError("Unknown cell_type " + cell_type)


def create_rnn(cell_type,
               dim_in,
               directions=1,
               skip_input_transform=False,
               num_layers=1,
               hidden_size=None):
    assert isinstance(hidden_size, int)
    assert cell_type is not None
    assert num_layers >= 1
    assert directions in (1, 2)
    cell = get_cell_class(cell_type)
    cl = BiRNNRefStack if directions == 2 else RNNRefStack
    return cl(
        cell,
        hidden_size=hidden_size,
        dim_in=dim_in,
        layers=num_layers,
        skip=skip_input_transform)


def _create_test(name,
                 cell_type,
                 hidden_size,
                 dim_in,
                 num_layers=1,
                 directions=1,
                 batch_size=20,
                 timesteps=10,
                 **orig_kwargs):
    kwargs = orig_kwargs.copy()
    kwargs.update({
        'cell_type': cell_type,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'directions': directions,
    })
    r = create_rnn(dim_in=dim_in, **kwargs)
    hdim = directions * num_layers
    num_outputs = 3 if get_cell_class(cell_type).HAS_CELL else 2
    inputs = [(r.param_size(), ), (timesteps, batch_size, dim_in),
              (hdim, batch_size, hidden_size)]
    if (get_cell_class(cell_type).HAS_CELL):
        inputs.append((hdim, batch_size, hidden_size))
    return (
        name,
        N("OptimizedRNN", **kwargs),
        lambda w, *args: r.feed_params(w).forward(*args)[:num_outputs],
        inputs,
        'CUDA', )


# Caffe2 backend implements only CUDA device for now
node_tests = [
    _create_test(
        "test_rnn_lstm_simple", cell_type='lstm', hidden_size=32, dim_in=15),
    _create_test(
        "test_rnn_lstm_multilayer",
        cell_type='lstm',
        hidden_size=32,
        dim_in=15,
        num_layers=3),
    _create_test(
        "test_rnn_lstm_bidirectional",
        cell_type='lstm',
        hidden_size=32,
        dim_in=15,
        num_layers=3,
        directions=2),
    _create_test(
        "test_rnn_lstm_skipinput",
        cell_type='lstm',
        hidden_size=32,
        dim_in=32,
        num_layers=3,
        skip_input_transform=True),
    _create_test(
    "test_rnn_gru_simple", cell_type='gru', hidden_size=32, dim_in=15),
]
"""
