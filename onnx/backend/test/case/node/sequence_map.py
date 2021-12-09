# SPDX-License-Identifier: Apache-2.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class SequenceMap(Base):
    @staticmethod
    def export_sequence_map_identity_1_sequence():  # type: () -> None
        body = onnx.helper.make_graph(
            [onnx.helper.make_node('Identity', ['in0'], ['out0'])],
            'seq_map_body',
            [onnx.helper.make_tensor_value_info(
                'in0', onnx.TensorProto.FLOAT, ['N'])],
            [onnx.helper.make_tensor_value_info(
                'out0', onnx.TensorProto.FLOAT, ['N'])]
        )

        node = onnx.helper.make_node(
            'SequenceMap',
            inputs=['x'],
            outputs=['y'],
            body=body
        )

        x = [np.random.uniform(0.0, 1.0, 10).astype(np.float32)
             for _ in range(3)]
        y = x
        input_type_protos = [
            onnx.helper.make_sequence_type_proto(
                onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, ['N'])),
        ]
        output_type_protos = [
            onnx.helper.make_sequence_type_proto(
                onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, ['N'])),
        ]
        expect(node, inputs=[x], outputs=[y],
               input_type_protos=input_type_protos,
               output_type_protos=output_type_protos,
               name='test_sequence_map_identity_1_sequence')

    @staticmethod
    def export_sequence_map_identity_2_sequences():  # type: () -> None
        body = onnx.helper.make_graph(
            [onnx.helper.make_node('Identity', ['in0'], ['out0']),
             onnx.helper.make_node('Identity', ['in1'], ['out1'])],
            'seq_map_body',
            [onnx.helper.make_tensor_value_info('in0', onnx.TensorProto.FLOAT, ['N']),
             onnx.helper.make_tensor_value_info('in1', onnx.TensorProto.FLOAT, ['N'])],
            [onnx.helper.make_tensor_value_info('out0', onnx.TensorProto.FLOAT, ['N']),
             onnx.helper.make_tensor_value_info('out1', onnx.TensorProto.FLOAT, ['N'])]
        )

        node = onnx.helper.make_node(
            'SequenceMap',
            inputs=['x0', 'x1'],
            outputs=['y0', 'y1'],
            body=body
        )

        x0 = [np.random.uniform(0.0, 1.0, np.random.randint(
            1, 10)).astype(np.float32) for _ in range(3)]
        x1 = [np.random.uniform(0.0, 1.0, np.random.randint(
            1, 10)).astype(np.float32) for _ in range(3)]
        y0 = x0
        y1 = x1
        input_type_protos = [
            onnx.helper.make_sequence_type_proto(
                onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, ['N'])),
            onnx.helper.make_sequence_type_proto(
                onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, ['N'])),
        ]
        output_type_protos = [
            onnx.helper.make_sequence_type_proto(
                onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, ['N'])),
            onnx.helper.make_sequence_type_proto(
                onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, ['N'])),
        ]
        expect(node, inputs=[x0, x1], outputs=[y0, y1],
               input_type_protos=input_type_protos,
               output_type_protos=output_type_protos,
               name='test_sequence_map_identity_2_sequences')

    @staticmethod
    def export_sequence_map_identity_1_sequence_1_tensor():  # type: () -> None
        body = onnx.helper.make_graph(
            [onnx.helper.make_node('Identity', ['in0'], ['out0']),
             onnx.helper.make_node('Identity', ['in1'], ['out1'])],
            'seq_map_body',
            [onnx.helper.make_tensor_value_info('in0', onnx.TensorProto.FLOAT, ['N']),
             onnx.helper.make_tensor_value_info('in1', onnx.TensorProto.FLOAT, ['N'])],
            [onnx.helper.make_tensor_value_info(
                'out0', onnx.TensorProto.FLOAT, ['N']),
             onnx.helper.make_tensor_value_info(
                'out1', onnx.TensorProto.FLOAT, ['N'])]
        )

        node = onnx.helper.make_node(
            'SequenceMap',
            inputs=['x0', 'x1'],
            outputs=['y0', 'y1'],
            body=body
        )

        x0 = [np.random.uniform(0.0, 1.0, np.random.randint(
            1, 10)).astype(np.float32) for _ in range(3)]
        x1 = np.random.uniform(0.0, 1.0, np.random.randint(
            1, 10)).astype(np.float32)
        y0 = x0
        y1 = [x1 for _ in range(3)]
        input_type_protos = [
            onnx.helper.make_sequence_type_proto(
                onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, ['N'])),
            onnx.helper.make_sequence_type_proto(
                onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, ['N'])),
        ]
        output_type_protos = [
            onnx.helper.make_sequence_type_proto(
                onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, ['N'])),
            onnx.helper.make_sequence_type_proto(
                onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, ['N'])),
        ]
        expect(node, inputs=[x0, x1], outputs=[y0, y1],
               input_type_protos=input_type_protos,
               output_type_protos=output_type_protos,
               name='test_sequence_map_identity_1_sequence_1_tensor')

    @staticmethod
    def export_sequence_map_add_2_sequences():  # type: () -> None
        body = onnx.helper.make_graph(
            [onnx.helper.make_node('Add', ['in0', 'in1'], ['out0'])],
            'seq_map_body',
            [onnx.helper.make_tensor_value_info('in0', onnx.TensorProto.FLOAT, ['N']),
             onnx.helper.make_tensor_value_info('in1', onnx.TensorProto.FLOAT, ['N'])],
            [onnx.helper.make_tensor_value_info(
                'out0', onnx.TensorProto.FLOAT, ['N'])]
        )

        node = onnx.helper.make_node(
            'SequenceMap',
            inputs=['x0', 'x1'],
            outputs=['y0'],
            body=body
        )

        N = [np.random.randint(1, 10) for _ in range(3)]
        x0 = [np.random.uniform(0.0, 1.0, N[k]).astype(np.float32)
              for k in range(3)]
        x1 = [np.random.uniform(0.0, 1.0, N[k]).astype(np.float32)
              for k in range(3)]
        y0 = x0 + x1
        input_type_protos = [
            onnx.helper.make_sequence_type_proto(
                onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, ['N'])),
            onnx.helper.make_sequence_type_proto(
                onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, ['N'])),
        ]
        output_type_protos = [
            onnx.helper.make_sequence_type_proto(
                onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, ['N'])),
        ]
        expect(node, inputs=[x0, x1], outputs=[y0],
               input_type_protos=input_type_protos,
               output_type_protos=output_type_protos,
               name='test_sequence_map_add_2_sequences')

    @staticmethod
    def export_sequence_map_add_1_sequence_1_tensor():  # type: () -> None
        body = onnx.helper.make_graph(
            [onnx.helper.make_node('Add', ['in0', 'in1'], ['out0'])],
            'seq_map_body',
            [onnx.helper.make_tensor_value_info('in0', onnx.TensorProto.FLOAT, ['N']),
             onnx.helper.make_tensor_value_info('in1', onnx.TensorProto.FLOAT, ['N'])],
            [onnx.helper.make_tensor_value_info(
                'out0', onnx.TensorProto.FLOAT, ['N'])]
        )

        node = onnx.helper.make_node(
            'SequenceMap',
            inputs=['x0', 'x1'],
            outputs=['y0'],
            body=body
        )

        x0 = [np.random.uniform(0.0, 1.0, 10).astype(np.float32) for k in range(3)]
        x1 = np.random.uniform(0.0, 1.0, 10).astype(np.float32)
        y0 = [x0[i] + x1 for i in range(3)]
        input_type_protos = [
            onnx.helper.make_sequence_type_proto(
                onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, ['N'])),
            onnx.helper.make_sequence_type_proto(
                onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, ['N'])),
        ]
        output_type_protos = [
            onnx.helper.make_sequence_type_proto(
                onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, ['N'])),
        ]
        expect(node, inputs=[x0, x1], outputs=[y0],
               input_type_protos=input_type_protos,
               output_type_protos=output_type_protos,
               name='test_sequence_map_add_1_sequence_1_tensor')
