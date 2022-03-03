# SPDX-License-Identifier: Apache-2.0
# coding: utf-8


import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect
from typing import Sequence


class NormalizeStrings(Base):

    @staticmethod
    def export() -> None:
        def make_graph(node: onnx.helper.NodeProto, input_shape: Sequence[int], output_shape: Sequence[int]) -> onnx.helper.GraphProto:
            graph = onnx.helper.make_graph(
                nodes=[node],
                name='StringNormalizer',
                inputs=[onnx.helper.make_tensor_value_info('x',
                                                            onnx.TensorProto.STRING,
                                                            input_shape)],
                outputs=[onnx.helper.make_tensor_value_info('y',
                                                             onnx.TensorProto.STRING,
                                                             output_shape)])
            return graph

        #1st model_monday_casesensintive_nochangecase
        stopwords = [u'monday']
        node = onnx.helper.make_node(
            'StringNormalizer',
            inputs=['x'],
            outputs=['y'],
            is_case_sensitive=1,
            stopwords=stopwords
        )

        x = np.array([u'monday', u'tuesday', u'wednesday', u'thursday']).astype(object)
        y = np.array([u'tuesday', u'wednesday', u'thursday']).astype(object)

        graph = make_graph(node, [4], [3])
        model = onnx.helper.make_model(graph, producer_name='backend-test')
        expect(model, inputs=[x], outputs=[y], name="test_strnorm_model_monday_casesensintive_nochangecase")

        #2nd model_nostopwords_nochangecase
        node = onnx.helper.make_node(
            'StringNormalizer',
            inputs=['x'],
            outputs=['y'],
            is_case_sensitive=1
        )

        x = np.array([u'monday', u'tuesday']).astype(object)
        y = x

        graph = make_graph(node, [2], [2])
        model = onnx.helper.make_model(graph, producer_name='backend-test')
        expect(model, inputs=[x], outputs=[y], name="test_strnorm_model_nostopwords_nochangecase")

        # 3rd model_monday_casesensintive_lower
        stopwords = [u'monday']
        node = onnx.helper.make_node(
            'StringNormalizer',
            inputs=['x'],
            outputs=['y'],
            case_change_action='LOWER',
            is_case_sensitive=1,
            stopwords=stopwords
        )

        x = np.array([u'monday', u'tuesday', u'wednesday', u'thursday']).astype(object)
        y = np.array([u'tuesday', u'wednesday', u'thursday']).astype(object)

        graph = make_graph(node, [4], [3])
        model = onnx.helper.make_model(graph, producer_name='backend-test')
        expect(model, inputs=[x], outputs=[y], name="test_strnorm_model_monday_casesensintive_lower")

        #4 model_monday_casesensintive_upper
        stopwords = [u'monday']
        node = onnx.helper.make_node(
            'StringNormalizer',
            inputs=['x'],
            outputs=['y'],
            case_change_action='UPPER',
            is_case_sensitive=1,
            stopwords=stopwords
        )

        x = np.array([u'monday', u'tuesday', u'wednesday', u'thursday']).astype(object)
        y = np.array([u'TUESDAY', u'WEDNESDAY', u'THURSDAY']).astype(object)

        graph = make_graph(node, [4], [3])
        model = onnx.helper.make_model(graph, producer_name='backend-test')
        expect(model, inputs=[x], outputs=[y], name="test_strnorm_model_monday_casesensintive_upper")

        #5 monday_insensintive_upper_twodim
        stopwords = [u'monday']
        node = onnx.helper.make_node(
            'StringNormalizer',
            inputs=['x'],
            outputs=['y'],
            case_change_action='UPPER',
            stopwords=stopwords
        )

        input_shape = [1, 6]
        output_shape = [1, 4]
        x = np.array([u'Monday', u'tuesday', u'wednesday', u'Monday', u'tuesday', u'wednesday']).astype(object).reshape(input_shape)
        y = np.array([u'TUESDAY', u'WEDNESDAY', u'TUESDAY', u'WEDNESDAY']).astype(object).reshape(output_shape)

        graph = make_graph(node, input_shape, output_shape)
        model = onnx.helper.make_model(graph, producer_name='backend-test')
        expect(model, inputs=[x], outputs=[y], name="test_strnorm_model_monday_insensintive_upper_twodim")

        #6 monday_empty_output
        stopwords = [u'monday']
        node = onnx.helper.make_node(
            'StringNormalizer',
            inputs=['x'],
            outputs=['y'],
            case_change_action='UPPER',
            is_case_sensitive=0,
            stopwords=stopwords
        )

        x = np.array([u'monday', u'monday']).astype(object)
        y = np.array([u'']).astype(object)

        graph = make_graph(node, [2], [1])
        model = onnx.helper.make_model(graph, producer_name='backend-test')
        expect(model, inputs=[x], outputs=[y], name="test_strnorm_model_monday_empty_output")
