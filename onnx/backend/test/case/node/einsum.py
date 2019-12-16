
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect
from typing import List

def einsum_reference_implementation(Eqn, Operands):  # type: (np.ndarray, List[np.ndarray]) -> np.ndarray
    Z = np.einsum(Eqn[0], *Operands)
    return Z


class Einsum(Base):

    @staticmethod
    def export_einsum_transpose():  # type: () -> None
        node = onnx.helper.make_node(
            'Einsum',
            inputs=['ij->ji', 'x'],
            outputs=['y'],
        )

        Eqn = np.array([u'ij->ji']).astype(np.object)
        X = np.random.randn(3, 4)
        Y = einsum_reference_implementation(Eqn, (X,))

        expect(node, inputs=[Eqn, X], outputs=[Y],
               name='test_einsum_transpose')

    @staticmethod
    def export_einsum_sum():  # type: () -> None
        node = onnx.helper.make_node(
            'Einsum',
            inputs=['ij->i', 'x'],
            outputs=['y'],
        )

        Eqn = np.array([u'ij->i']).astype(np.object)
        X = np.random.randn(3, 4)
        Z = einsum_reference_implementation(Eqn, (X,))

        expect(node, inputs=[Eqn, X], outputs=[Z],
               name='test_einsum_sum')

    @staticmethod
    def export_einsum_batch_diagonal():  # type: () -> None
        node = onnx.helper.make_node(
            'Einsum',
            inputs=['...ii->...i', 'x'],
            outputs=['y'],
        )

        Eqn = np.array([u'...ii->...i']).astype(np.object)
        X = np.random.randn(3, 5, 5)
        Z = einsum_reference_implementation(Eqn, (X,))

        expect(node, inputs=[Eqn, X], outputs=[Z],
               name='test_einsum_batch_diagonal')

    @staticmethod
    def export_einsum_inner_prod():  # type: () -> None
        node = onnx.helper.make_node(
            'Einsum',
            inputs=['i,i', 'x', 'y'],
            outputs=['z'],
        )

        Eqn = np.array([u'i,i']).astype(np.object)
        X = np.random.randn(5)
        Y = np.random.randn(5)
        Z = einsum_reference_implementation(Eqn, (X, Y))

        expect(node, inputs=[Eqn, X, Y], outputs=[Z],
               name='test_einsum_inner_prod')

    @staticmethod
    def export_einsum_batch_matmul():  # type: () -> None
        node = onnx.helper.make_node(
            'Einsum',
            inputs=['bij,bjk->bik', 'x', 'y'],
            outputs=['z'],
        )

        Eqn = np.array([u'bij,bjk->bik']).astype(np.object)
        X = np.random.randn(5, 2, 3)
        Y = np.random.randn(5, 3, 4)
        Z = einsum_reference_implementation(Eqn, (X, Y))

        expect(node, inputs=[Eqn, X, Y], outputs=[Z],
               name='test_einsum_batch_matmul')
