from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class NonMaxSuppression(Base):

    @staticmethod
    def export_nonmaxsuppression_suppress_by_IOU():  # type: () -> None
        node = onnx.helper.make_node(
            'NonMaxSuppression',
            inputs=['boxes', 'scores', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold'],
            outputs=['selected_indices']
        )
        boxes = np.array([[
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.1, 1.0, 1.1],
            [0.0, -0.1, 1.0, 0.9],
            [0.0, 10.0, 1.0, 11.0],
            [0.0, 10.1, 1.0, 11.1],
            [0.0, 100.0, 1.0, 101.0]
        ]]).astype(np.float32)
        scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
        max_output_boxes_per_class = np.array([3]).astype(np.int64)
        iou_threshold = np.array([0.5]).astype(np.float32)
        score_threshold = np.array([0.0]).astype(np.float32)
        selected_indices = np.array([[0, 0, 3], [0, 0, 0], [0, 0, 5]]).astype(np.int64)

        expect(node, inputs=[boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold], outputs=[selected_indices], name='test_nonmaxsuppression_suppress_by_IOU')

    @staticmethod
    def export_nonmaxsuppression_suppress_by_IOU_and_scores():  # type: () -> None
        node = onnx.helper.make_node(
            'NonMaxSuppression',
            inputs=['boxes', 'scores', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold'],
            outputs=['selected_indices']
        )
        boxes = np.array([[
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.1, 1.0, 1.1],
            [0.0, -0.1, 1.0, 0.9],
            [0.0, 10.0, 1.0, 11.0],
            [0.0, 10.1, 1.0, 11.1],
            [0.0, 100.0, 1.0, 101.0]
        ]]).astype(np.float32)
        scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
        max_output_boxes_per_class = np.array([3]).astype(np.int64)
        iou_threshold = np.array([0.5]).astype(np.float32)
        score_threshold = np.array([0.4]).astype(np.float32)
        selected_indices = np.array([[0, 0, 3], [0, 0, 0]]).astype(np.int64)

        expect(node, inputs=[boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold], outputs=[selected_indices], name='test_nonmaxsuppression_suppress_by_IOU_and_scores')

    @staticmethod
    def export_nonmaxsuppression_flipped_coordinates():  # type: () -> None
        node = onnx.helper.make_node(
            'NonMaxSuppression',
            inputs=['boxes', 'scores', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold'],
            outputs=['selected_indices']
        )
        boxes = np.array([[
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.1, 1.0, 1.1],
            [0.0, 0.9, 1.0, -0.1],
            [0.0, 10.0, 1.0, 11.0],
            [1.0, 10.1, 0.0, 11.1],
            [1.0, 101.0, 0.0, 100.0]
        ]]).astype(np.float32)
        scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
        max_output_boxes_per_class = np.array([3]).astype(np.int64)
        iou_threshold = np.array([0.5]).astype(np.float32)
        score_threshold = np.array([0.0]).astype(np.float32)
        selected_indices = np.array([[0, 0, 3], [0, 0, 0], [0, 0, 5]]).astype(np.int64)

        expect(node, inputs=[boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold], outputs=[selected_indices], name='test_nonmaxsuppression_flipped_coordinates')

    @staticmethod
    def export_nonmaxsuppression_limit_output_size():  # type: () -> None
        node = onnx.helper.make_node(
            'NonMaxSuppression',
            inputs=['boxes', 'scores', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold'],
            outputs=['selected_indices']
        )
        boxes = np.array([[
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.1, 1.0, 1.1],
            [0.0, -0.1, 1.0, 0.9],
            [0.0, 10.0, 1.0, 11.0],
            [0.0, 10.1, 1.0, 11.1],
            [0.0, 100.0, 1.0, 101.0]
        ]]).astype(np.float32)
        scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
        max_output_boxes_per_class = np.array([2]).astype(np.int64)
        iou_threshold = np.array([0.5]).astype(np.float32)
        score_threshold = np.array([0.0]).astype(np.float32)
        selected_indices = np.array([[0, 0, 3], [0, 0, 0]]).astype(np.int64)

        expect(node, inputs=[boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold], outputs=[selected_indices], name='test_nonmaxsuppression_limit_output_size')

    @staticmethod
    def export_nonmaxsuppression_single_box():  # type: () -> None
        node = onnx.helper.make_node(
            'NonMaxSuppression',
            inputs=['boxes', 'scores', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold'],
            outputs=['selected_indices']
        )
        boxes = np.array([[
            [0.0, 0.0, 1.0, 1.0]
        ]]).astype(np.float32)
        scores = np.array([[[0.9]]]).astype(np.float32)
        max_output_boxes_per_class = np.array([3]).astype(np.int64)
        iou_threshold = np.array([0.5]).astype(np.float32)
        score_threshold = np.array([0.0]).astype(np.float32)
        selected_indices = np.array([[0, 0, 0]]).astype(np.int64)

        expect(node, inputs=[boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold], outputs=[selected_indices], name='test_nonmaxsuppression_single_box')

    @staticmethod
    def export_nonmaxsuppression_identical_boxes():  # type: () -> None
        node = onnx.helper.make_node(
            'NonMaxSuppression',
            inputs=['boxes', 'scores', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold'],
            outputs=['selected_indices']
        )
        boxes = np.array([[
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],

            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0]
        ]]).astype(np.float32)
        scores = np.array([[[0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]]]).astype(np.float32)
        max_output_boxes_per_class = np.array([3]).astype(np.int64)
        iou_threshold = np.array([0.5]).astype(np.float32)
        score_threshold = np.array([0.0]).astype(np.float32)
        selected_indices = np.array([[0, 0, 0]]).astype(np.int64)

        expect(node, inputs=[boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold], outputs=[selected_indices], name='test_nonmaxsuppression_identical_boxes')

    @staticmethod
    def export_nonmaxsuppression_center_point_box_format():  # type: () -> None
        node = onnx.helper.make_node(
            'NonMaxSuppression',
            inputs=['boxes', 'scores', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold'],
            outputs=['selected_indices'],
            center_point_box=1
        )
        boxes = np.array([[
            [0.5, 0.5, 1.0, 1.0],
            [0.5, 0.6, 1.0, 1.0],
            [0.5, 0.4, 1.0, 1.0],
            [0.5, 10.5, 1.0, 1.0],
            [0.5, 10.6, 1.0, 1.0],
            [0.5, 100.5, 1.0, 1.0]
        ]]).astype(np.float32)
        scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
        max_output_boxes_per_class = np.array([3]).astype(np.int64)
        iou_threshold = np.array([0.5]).astype(np.float32)
        score_threshold = np.array([0.0]).astype(np.float32)
        selected_indices = np.array([[0, 0, 3], [0, 0, 0], [0, 0, 5]]).astype(np.int64)

        expect(node, inputs=[boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold], outputs=[selected_indices], name='test_nonmaxsuppression_center_point_box_format')

    @staticmethod
    def export_nonmaxsuppression_two_classes():  # type: () -> None
        node = onnx.helper.make_node(
            'NonMaxSuppression',
            inputs=['boxes', 'scores', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold'],
            outputs=['selected_indices']
        )
        boxes = np.array([[
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.1, 1.0, 1.1],
            [0.0, -0.1, 1.0, 0.9],
            [0.0, 10.0, 1.0, 11.0],
            [0.0, 10.1, 1.0, 11.1],
            [0.0, 100.0, 1.0, 101.0]
        ]]).astype(np.float32)
        scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3],
                            [0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
        max_output_boxes_per_class = np.array([2]).astype(np.int64)
        iou_threshold = np.array([0.5]).astype(np.float32)
        score_threshold = np.array([0.0]).astype(np.float32)
        selected_indices = np.array([[0, 0, 3], [0, 0, 0], [0, 1, 3], [0, 1, 0]]).astype(np.int64)

        expect(node, inputs=[boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold], outputs=[selected_indices], name='test_nonmaxsuppression_two_classes')

    @staticmethod
    def export_nonmaxsuppression_two_batches():  # type: () -> None
        node = onnx.helper.make_node(
            'NonMaxSuppression',
            inputs=['boxes', 'scores', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold'],
            outputs=['selected_indices']
        )
        boxes = np.array([[[0.0, 0.0, 1.0, 1.0],
                           [0.0, 0.1, 1.0, 1.1],
                           [0.0, -0.1, 1.0, 0.9],
                           [0.0, 10.0, 1.0, 11.0],
                           [0.0, 10.1, 1.0, 11.1],
                           [0.0, 100.0, 1.0, 101.0]],
                          [[0.0, 0.0, 1.0, 1.0],
                           [0.0, 0.1, 1.0, 1.1],
                           [0.0, -0.1, 1.0, 0.9],
                           [0.0, 10.0, 1.0, 11.0],
                           [0.0, 10.1, 1.0, 11.1],
                           [0.0, 100.0, 1.0, 101.0]]]).astype(np.float32)
        scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]],
                           [[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
        max_output_boxes_per_class = np.array([2]).astype(np.int64)
        iou_threshold = np.array([0.5]).astype(np.float32)
        score_threshold = np.array([0.0]).astype(np.float32)
        selected_indices = np.array([[0, 0, 3], [0, 0, 0], [1, 0, 3], [1, 0, 0]]).astype(np.int64)

        expect(node, inputs=[boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold], outputs=[selected_indices], name='test_nonmaxsuppression_two_batches')
