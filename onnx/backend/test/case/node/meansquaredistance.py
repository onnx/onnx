from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class MeanSquaredDistance(Base):

    @staticmethod
    def export_mean_square_distance_none():  # type: () -> None
        # Define operator attributes.
        reduction = 'none'

        # Create operator.
        node = onnx.helper.make_node('MeanSquaredDistance',
                                     inputs=['R', 'T'],
                                     outputs=['X'],
                                     reduction=reduction
                                     )

        # Define operator inputs
        r = np.array([1.2, 2.5], dtype=np.float32)
        t = np.array([1.1, 2.6], dtype=np.float32)

        # Compute Mean Square Distance
        msd = (np.square(r - t))

        # Check results
        expect(node, inputs=[r, t], outputs=[msd], name='test_mean_square_distance_none')


    @staticmethod
    def export_mean_square_distance_none_weights():  # type: () -> None
        # Define operator attributes.
        reduction = 'none'

        # Create operator.
        node = onnx.helper.make_node('MeanSquaredDistance',
                                     inputs=['R', 'T', 'W'],
                                     outputs=['X'],
                                     reduction=reduction
                                     )

        # Define operator inputs
        r = np.array([1.2, 2.5], dtype=np.float32)
        t = np.array([1.1, 2.6], dtype=np.float32)
        weights = np.array([0.8, 0.9], dtype=np.float32)

        # Compute Mean Square Distance
        l = np.square(r - t)
        msd = np.multiply(weights, l)

        # Check results
        expect(node, inputs=[r, t, weights], outputs=[msd], name='test_mean_square_distance_none_weights')


    @staticmethod
    def export_mean_square_distance_sum():  # type: () -> None
        # Define operator attributes.
        reduction = 'sum'

        # Create operator.
        node = onnx.helper.make_node('MeanSquaredDistance',
                                     inputs=['R', 'T'],
                                     outputs=['X'],
                                     reduction=reduction
                                     )

        # Define operator inputs
        r = np.array([[1.2, 2.5, 3.1], [1.3, 2.3, 3.4]], dtype=np.float32)
        t = np.array([[1.1, 2.6, 3.2], [1.4, 2.2, 3.3]], dtype=np.float32)

        # Compute Mean Square Distance
        l = np.square(r - t)
        msd = np.sum(l)

        # Check results
        expect(node, inputs=[r, t], outputs=[msd], name='test_mean_square_distance_sum')


    @staticmethod
    def export_mean_square_distance_mean():  # type: () -> None
        # Define operator attributes.
        reduction = 'mean'

        # Create operator.
        node = onnx.helper.make_node('MeanSquaredDistance',
                                     inputs=['R', 'T'],
                                     outputs=['X'],
                                     reduction=reduction
                                     )

        # Define operator inputs
        r = np.array([[1.2, 2.5, 3.1], [1.3, 2.3, 3.4]], dtype=np.float32)
        t = np.array([[1.1, 2.6, 3.2], [1.4, 2.2, 3.3]], dtype=np.float32)

        # Compute Mean Square Distance
        l = np.square(r - t)
        msd = np.mean(l)

        # Check results
        expect(node, inputs=[r, t], outputs=[msd], name='test_mean_square_distance_mean')


    @staticmethod
    def export_mean_square_distance_mean_3d():  # type: () -> None
        # Define operator attributes.
        reduction = 'mean'

        # Create operator.
        node = onnx.helper.make_node('MeanSquaredDistance',
                                     inputs=['R', 'T'],
                                     outputs=['X'],
                                     reduction=reduction
                                     )

        # Define operator inputs
        r = np.array([[[1.2, 2.5], [3.1, 1.3]], [[2.3, 3.4], [1.1, 2.2]], [[3.6, 1.7], [2.5, 3.8]]], dtype=np.float32)
        t = np.array([[[1.1, 2.6], [3.2, 1.4]], [[2.2, 3.3], [1.2, 2.1]], [[3.5, 1.6], [2.5, 3.9]]], dtype=np.float32)

        # Compute Mean Square Distance
        l = np.square(r - t)
        msd = np.mean(l)

        # Check results
        expect(node, inputs=[r, t], outputs=[msd], name='test_mean_square_distance_mean_3d')
