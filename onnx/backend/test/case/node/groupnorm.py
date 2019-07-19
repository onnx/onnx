from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


def GroupNorm4d(x, gamma, beta, G, eps=1e-05):  # type: (np.array, np.array, np.array, int, float) -> np.array
    N, C, H, W = x.shape
    x = x.reshape((N, G, C // G, H, W))
    mean = np.mean(x, axis=(2, 3, 4), keepdims=True)
    var = np.var(x, axis=(2, 3, 4), keepdims=True)
    x = (x - mean) / np.sqrt(var + eps)
    x = x.reshape((N, C, H, W))
    return x * gamma + beta


def GroupNormNd(x, gamma, beta, G, eps=1e-05):  # type: (np.array, np.array, np.array, int, float) -> np.array
    originalShape = x.shape
    N = x.shape[0]
    C = x.shape[1]
    D0_N = x.shape[2:]

    new_shape = (N,) + (G,) + (C // G,) + (D0_N)
    x = x.reshape(new_shape)

    i = 0
    axis = []
    for dim in x.shape:
        if i > 1:
            axis.append(i)
        i = i + 1

    mean = np.mean(x, axis=tuple(axis), keepdims=True)
    var = np.var(x, axis=tuple(axis), keepdims=True)
    x = (x - mean) / np.sqrt(var + eps)
    x = x.reshape(originalShape)
    return x * gamma + beta


class GroupNormalization(Base):

    @staticmethod
    def export():  # type: () -> None

        x = np.array([[[[4., 5.], [0., 6.]],
                       [[6., 6.], [9., 6.]],
                       [[3., 4.], [6., 0.]],
                       [[6., 6.], [2., 1.]]],
                      [[[3., 3.], [5., 4.]],
                       [[9., 1.], [2., 1.]],
                       [[6., 2.], [4., 2.]],
                       [[1., 1.], [6., 1.]]]]).astype(np.float32)

        b = np.array([1, 1.5, -1.0, 1.5]).astype(np.float32)
        s = np.array([1, 1, 0, 1]).astype(np.float32)

        num_groups = 2
        eps = 1e-05

        node = onnx.helper.make_node('GroupNormalization',
                                     inputs=['x', 's', 'b'],
                                     outputs=['y'],
                                     num_groups=num_groups,
                                     epsilon=eps)

        y = GroupNorm4d(x, s.reshape((1, 4, 1, 1)), b.reshape((1, 4, 1, 1)), num_groups, eps)
        expect(node, inputs=[x, s, b], outputs=[y], name='test_groupnorm')

    @staticmethod
    def export_large_eps():  # type: () -> None

        x = np.random.rand(2, 4, 10, 12).astype(np.float32)

        b = np.array([0, 0, 0, 0]).astype(np.float32)
        s = np.array([1, 1, 1, 1]).astype(np.float32)

        num_groups = 2
        eps = 10.0

        node = onnx.helper.make_node('GroupNormalization',
                                     inputs=['x', 's', 'b'],
                                     outputs=['y'],
                                     num_groups=num_groups,
                                     epsilon=eps)

        y = GroupNorm4d(x, s.reshape((1, 4, 1, 1)), b.reshape((1, 4, 1, 1)), num_groups, eps)
        expect(node, inputs=[x, s, b], outputs=[y], name='test_groupnorm_large_eps')

    @staticmethod
    def export_large_num_groups():  # type: () -> None

        x = np.random.rand(2, 18, 10, 12).astype(np.float32)

        b = np.zeros([18]).astype(np.float32)
        s = np.ones([18]).astype(np.float32)

        num_groups = 9

        node = onnx.helper.make_node('GroupNormalization',
                                     inputs=['x', 's', 'b'],
                                     outputs=['y'],
                                     num_groups=num_groups)

        y = GroupNorm4d(x, s.reshape((1, 18, 1, 1)), b.reshape((1, 18, 1, 1)), num_groups)
        expect(node, inputs=[x, s, b], outputs=[y], name='test_groupnorm_large_num_groups')

    @staticmethod
    def export_6_dimensions():  # type: () -> None

        # A 4 dimension
        x = np.random.rand(2, 4, 3, 6, 4, 2).astype(np.float32)

        b = np.array([0, 0, 0, 0]).astype(np.float32)
        s = np.array([1, 1, 1, 1]).astype(np.float32)

        num_groups = 2
        eps = 1e-05

        y = GroupNormNd(x, s.reshape((1, 4, 1, 1, 1, 1)), b.reshape((1, 4, 1, 1, 1, 1)), num_groups, eps)

        node = onnx.helper.make_node('GroupNormalization',
                                     inputs=['x', 's', 'b'],
                                     outputs=['y'],
                                     num_groups=num_groups,
                                     epsilon=eps)
        expect(node, inputs=[x, s, b], outputs=[y], name='test_groupnorm_6D')
