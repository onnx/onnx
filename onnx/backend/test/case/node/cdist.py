from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


def np_cdist(x, y, metric='euclidean', p=2):  # type: (np.ndarray, np.ndarray, Text) -> (np.ndarray)
    if metric == 'sqeuclidean':
        z = np.empty((x.shape[0], y.shape[0]))
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                d = x[i, :] - y[j, :]
                z[i, j] = d @ d
        return z.astype(x.dtype)

    if metric == 'euclidean':
        z = np.empty((x.shape[0], y.shape[0]))
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                d = x[i, :] - y[j, :]
                z[i, j] = (d @ d) ** 0.5
        return z.astype(x.dtype)

    if metric in ('manhattan', 'cityblock'):
        z = np.empty((x.shape[0], y.shape[0]))
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                d = x[i, :] - y[j, :]
                z[i, j] = np.sum(np.abs(d))
        return z.astype(x.dtype)

    if metric in ('minkowski'):
        z = np.empty((x.shape[0], y.shape[0]))
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                d = x[i, :] - y[j, :]
                z[i, j] = np.sum(np.power(d, p)) ** (1. / p)
        return z.astype(x.dtype)

    raise NotImplementedError("Metric '%s' is not supported." % metric)


class CDist(Base):

    @staticmethod
    def export():  # type: () -> None
        for metric in ['sqeuclidean', 'euclidean', 'manhattan', 'minkowski']:
            node = onnx.helper.make_node(
                'CDist',
                inputs=['x', 'y'],
                outputs=['z'],
                metric=metric,
            )
            x = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
            y = np.array([[7, 8, 9]]).astype(np.float32)
            z = np_cdist(x, y, metric)  # expected output [[108.], [27.]] for sqeuclidean
            expect(node, inputs=[x, y], outputs=[z],
                   name='test_cdist_%s_example' % metric)

        for metric in ['minkowski']:
            for p in [3]:
                node = onnx.helper.make_node(
                    'CDist',
                    inputs=['x', 'y'],
                    outputs=['z'],
                    metric=metric,
                    p=p,
                )
                x = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
                y = np.array([[7, 8, 9]]).astype(np.float32)
                z = np_cdist(x, y, metric)
                expect(node, inputs=[x, y], outputs=[z],
                       name='test_cdist_%s_%d_dexample' % (metric, p))

