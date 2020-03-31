from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


def softmaxcrossentropy(x, target, weight=None, reduction='mean', ignore_index=None):  # type: ignore
    max_x = np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x - max_x)
    p = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    inp = np.log(p)
    input_shape = inp.shape
    if len(input_shape) == 2:
        N, C = input_shape
        neg_gather_element_input = np.zeros((N, ), dtype=np.float32)
        for i in range(N):
            if target[i] != ignore_index:
                neg_gather_element_input[i] = -inp[i][target[i]]
    elif len(input_shape) == 3:
        N, C, D = input_shape
        neg_gather_element_input = np.zeros((N, D), dtype=np.float32)
        for i in range(N):
            for d in range(D):
                if target[i][d] != ignore_index:
                    neg_gather_element_input[i][d] = -inp[i][target[i][d]][d]
    elif len(input_shape) == 4:
        N, C, D, H = input_shape
        neg_gather_element_input = np.zeros((N, D, H), dtype=np.float32)
        for i in range(N):
            for d in range(D):
                for h in range(H):
                    if target[i][d][h] != ignore_index:
                        neg_gather_element_input[i][d][h] = -inp[i][target[i][d][h]][d][h]
    else:
        raise NotImplementedError
    loss = neg_gather_element_input

    if weight is None and ignore_index is not None:
        c = input_shape[1]
        weight = np.ones(c, dtype=np.float32)
        weight[ignore_index] = 0

    if weight is not None:
        gather_weight = np.take(weight, target)

        if ignore_index is not None:
            if len(input_shape) == 2:
                for i in range(input_shape[0]):
                    if target[i] == ignore_index:
                        gather_weight[i] = 0
            elif len(input_shape) == 3:
                for i in range(input_shape[0]):
                    for j in range(input_shape[2]):
                        if target[i][j] == ignore_index:
                            gather_weight[i][j] = 0
            elif len(input_shape) == 4:
                for i in range(input_shape[0]):
                    for j in range(input_shape[2]):
                        for k in range(input_shape[3]):
                            if target[i][j][k] == ignore_index:
                                gather_weight[i][j][k] = 0
            else:
                raise NotImplementedError

        loss = gather_weight * loss
        if reduction == 'mean':
            return loss.sum() / gather_weight.sum()

    if reduction == 'mean':
        loss = np.mean(loss)
    if reduction == 'sum':
        loss = np.sum(loss)

    return loss


class SoftmaxCrossEntropyLoss(Base):

    @staticmethod
    def export_softmaxcrossentropy_none():  # type: () -> None
        # Define operator attributes.
        reduction = 'none'

        # Create operator.
        node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                                     inputs=['x', 'y'],
                                     outputs=['z'],
                                     reduction=reduction)

        # Define operator inputs.
        np.random.seed(0)
        x = np.random.rand(3, 5).astype(np.float32)
        labels = np.random.randint(0, high=5, size=(3, ))

        # Compute SoftmaxCrossEntropyLoss
        sce = softmaxcrossentropy(x, labels, reduction='none')

        # Check results
        expect(node, inputs=[x, labels], outputs=[sce], name='test_softmax_cross_entropy_none')

    @staticmethod
    def export_softmaxcrossentropy_none_weights():  # type: () -> None
        # Define operator attributes.
        reduction = 'none'

        # Create operator.
        node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                                     inputs=['x', 'y', 'w'],
                                     outputs=['z'],
                                     reduction=reduction)

        # Define operator inputs.
        np.random.seed(0)
        x = np.random.rand(3, 5).astype(np.float32)
        labels = np.random.randint(0, high=5, size=(3, ))
        weights = np.array([0.9, 0.7, 0.8, 0.9, 0.9], dtype=np.float32)

        # Compute SoftmaxCrossEntropyLoss
        sce = softmaxcrossentropy(x, labels, weight=weights, reduction='none')

        # Check results
        expect(node, inputs=[x, labels, weights], outputs=[sce], name='test_softmax_cross_entropy_none_weights')

    @staticmethod
    def export_softmaxcrossentropy_sum():  # type: () -> None
        # Define operator attributes.
        reduction = 'sum'

        # Create operator.
        node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                                     inputs=['x', 'y'],
                                     outputs=['z'],
                                     reduction=reduction)

        # Define operator inputs.
        np.random.seed(0)
        x = np.random.rand(3, 5).astype(np.float32)
        labels = np.random.randint(0, high=5, size=(3, ))

        # Compute SoftmaxCrossEntropyLoss
        sce = softmaxcrossentropy(x, labels, reduction='sum')

        # Check results
        expect(node, inputs=[x, labels], outputs=[sce], name='test_softmax_cross_entropy_sum')

    @staticmethod
    def export_softmaxcrossentropy_mean():  # type: () -> None
        # Define operator attributes.
        reduction = 'mean'

        # Create operator.
        node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                                     inputs=['x', 'y'],
                                     outputs=['z'],
                                     reduction=reduction)

        # Define operator inputs.
        np.random.seed(0)
        x = np.random.rand(3, 5).astype(np.float32)
        labels = np.random.randint(0, high=5, size=(3, ))

        # Compute SoftmaxCrossEntropyLoss
        sce = softmaxcrossentropy(x, labels)

        # Check results
        expect(node, inputs=[x, labels], outputs=[sce], name='test_softmax_cross_entropy_mean')

    @staticmethod
    def export_softmaxcrossentropy_mean_3d():  # type: () -> None
        # Define operator attributes.
        reduction = 'mean'

        # Create operator.
        node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                                     inputs=['x', 'y'],
                                     outputs=['z'],
                                     reduction=reduction)

        # Define operator inputs.
        np.random.seed(0)
        x = np.random.rand(3, 5, 2).astype(np.float32)
        y = np.random.randint(0, high=5, size=(3, 2))

        # Compute SoftmaxCrossEntropyLoss
        sce = softmaxcrossentropy(x, y)

        # Check results
        expect(node, inputs=[x, y], outputs=[sce], name='test_softmax_cross_entropy_mean_3d')

    @staticmethod
    def export_softmaxcrossentropy_mean_weights():  # type: () -> None
        # Define operator attributes.
        reduction = 'mean'

        # Create operator.
        node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                                     inputs=['x', 'y', 'w'],
                                     outputs=['z'],
                                     reduction=reduction)

        # Define operator inputs.
        np.random.seed(0)
        x = np.random.rand(3, 5).astype(np.float32)
        labels = np.random.randint(0, high=5, size=(3, ))
        weights = np.array([0.9, 0.7, 0.8, 0.9, 0.9], dtype=np.float32)

        # Compute SoftmaxCrossEntropyLoss
        sce = softmaxcrossentropy(x, labels, weight=weights)

        # Check results
        expect(node, inputs=[x, labels, weights], outputs=[sce], name='test_softmax_cross_entropy_mean_weight')

    @staticmethod
    def export_softmaxcrossentropy_mean_weights_ignore_index():  # type: () -> None
        # Define operator attributes.
        reduction = 'mean'
        ignore_index = np.int64(0)

        # Create operator.
        node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                                     inputs=['x', 'y', 'w'],
                                     outputs=['z'],
                                     reduction=reduction,
                                     ignore_index=ignore_index)

        # Define operator inputs.
        np.random.seed(0)
        x = np.random.rand(3, 5).astype(np.float32)
        labels = np.random.randint(0, high=5, size=(3, ))
        labels[0] = 0
        weights = np.array([0.9, 0.7, 0.8, 0.9, 0.9], dtype=np.float32)

        # Compute SoftmaxCrossEntropyLoss
        sce = softmaxcrossentropy(x, labels, weight=weights, ignore_index=ignore_index)

        # Check results
        expect(node, inputs=[x, labels, weights], outputs=[sce], name='test_softmax_cross_entropy_mean_weight_ignore_index')

    @staticmethod
    def export_softmaxcrossentropy_mean_no_weights_ignore_index():  # type: () -> None
        # Define operator attributes.
        reduction = 'mean'
        ignore_index = np.int64(2)

        # Create operator.
        node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                                    inputs=['x', 'y'],
                                    outputs=['z'],
                                    reduction=reduction,
                                    ignore_index=ignore_index)

        # Define operator inputs.
        np.random.seed(0)
        x = np.random.rand(3, 5).astype(np.float32)
        labels = np.array([1, 3, 2], dtype=np.int64)
        labels[0] = 2

        # Compute SoftmaxCrossEntropyLoss
        sce = softmaxcrossentropy(x, labels, ignore_index=ignore_index)

        # Check results
        expect(node, inputs=[x, labels], outputs=[sce], name='test_softmax_cross_entropy_mean_no_weight_ignore_index')

    @staticmethod
    def export_softmaxcrossentropy_mean_weights_ignore_index_3d():  # type: () -> None
        # Define operator attributes.
        reduction = 'mean'
        ignore_index = np.int64(1)

        # Create operator.
        node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                                    inputs=['x', 'y', 'w'],
                                    outputs=['z'],
                                    reduction=reduction,
                                    ignore_index=ignore_index)

        # Define operator inputs.
        np.random.seed(0)
        x = np.random.rand(3, 5, 2).astype(np.float32)
        labels = np.random.randint(0, high=5, size=(3, 2))
        labels[0][0] = 1
        weights = np.array([0.2, 0.3, 0.6, 0.1, 0.5], dtype=np.float32)

        # Compute SoftmaxCrossEntropyLoss
        sce = softmaxcrossentropy(x, labels, weight=weights, ignore_index=ignore_index)

        # Check results
        expect(node, inputs=[x, labels, weights], outputs=[sce], name='test_softmax_cross_entropy_mean_weight_ignore_index_3d')

    @staticmethod
    def export_softmaxcrossentropy_mean_no_weights_ignore_index_3d():  # type: () -> None
        # Define operator attributes.
        reduction = 'mean'
        ignore_index = np.int64(2)

        # Create operator.
        node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                                    inputs=['x', 'y'],
                                    outputs=['z'],
                                    reduction=reduction,
                                    ignore_index=ignore_index)

        # Define operator inputs.
        np.random.seed(0)
        x = np.random.rand(3, 5, 2).astype(np.float32)
        labels = np.random.randint(0, high=5, size=(3, 2))
        labels[0][0] = 2

        # Compute SoftmaxCrossEntropyLoss
        sce = softmaxcrossentropy(x, labels, ignore_index=ignore_index)

        # Check results
        expect(node, inputs=[x, labels], outputs=[sce], name='test_softmax_cross_entropy_mean_no_weight_ignore_index_3d')

    @staticmethod
    def export_softmaxcrossentropy_mean_weights_ignore_index_4d():  # type: () -> None
        # Define operator attributes.
        reduction = 'mean'
        ignore_index = np.int64(2)

        # Create operator.
        node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                                    inputs=['x', 'y', 'w'],
                                    outputs=['z'],
                                    reduction=reduction,
                                    ignore_index=ignore_index)

        # Define operator inputs.
        np.random.seed(0)
        x = np.random.rand(3, 5, 2, 7).astype(np.float32)
        labels = np.random.randint(0, high=5, size=(3, 2, 7))
        labels[0][0][0] = 2
        weights = np.array([0.2, 0.3, 0.6, 0.1, 0.5], dtype=np.float32)

        # Compute SoftmaxCrossEntropyLoss
        sce = softmaxcrossentropy(x, labels, reduction=reduction, weight=weights, ignore_index=ignore_index)

        # Check results
        expect(node, inputs=[x, labels, weights], outputs=[sce], name='test_softmax_cross_entropy_mean_weight_ignore_index_4d')

    @staticmethod
    def export_softmaxcrossentropy_mean_no_weights_ignore_index_4d():  # type: () -> None
        # Define operator attributes.
        reduction = 'mean'
        ignore_index = np.int64(2)

        # Create operator.
        node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                                    inputs=['x', 'y'],
                                    outputs=['z'],
                                    reduction=reduction,
                                    ignore_index=ignore_index)

        # Define operator inputs.
        np.random.seed(0)
        x = np.random.rand(3, 5, 2, 7).astype(np.float32)
        labels = np.random.randint(0, high=5, size=(3, 2, 7))
        labels[0][0][0] = 2

        # Compute SoftmaxCrossEntropyLoss
        sce = softmaxcrossentropy(x, labels, reduction=reduction, ignore_index=ignore_index)

        # Check results
        expect(node, inputs=[x, labels], outputs=[sce], name='test_softmax_cross_entropy_mean_no_weight_ignore_index_4d')
