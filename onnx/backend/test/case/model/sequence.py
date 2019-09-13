from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
import typing
from ..base import Base
from . import expect
from onnx import TensorProto
from typing import List, Optional, Text, Union


class Sequence(Base):

    @staticmethod
    def export():  # type: () -> None

        def make_graph(
                nodes,  # type: List[onnx.helper.NodeProto]
                input_shapes,  # type: List[Optional[typing.Sequence[Union[Text, int]]]]
                output_shapes,  # type: List[Optional[typing.Sequence[Union[Text, int]]]]
                input_names,  # type: List[Text]
                output_names,  # type: List[Text]
                input_types,  # type: List[TensorProto.DataType]
                output_types,  # type: List[TensorProto.DataType]
                initializers=None  # type: Optional[List[TensorProto]]
        ):  # type: (...) -> onnx.helper.GraphProto
            graph = onnx.helper.make_graph(
                nodes=nodes,
                name='Sequence',
                inputs=[
                    onnx.helper.make_tensor_value_info(
                        name,
                        input_type,
                        input_shape)
                    for name, input_type, input_shape in zip(input_names, input_types, input_shapes)],
                outputs=[
                    onnx.helper.make_tensor_value_info(
                        name,
                        output_type,
                        output_shape)
                    for name, output_type, output_shape in zip(output_names, output_types, output_shapes)],
                initializer=initializers)
            return graph

        #1st testcase - insert and at.
        # 1. SequenceEmpty:         -> []
        # 2. SequenceInsert(x):     -> [x]
        # 3. SequenceInsert(y):     -> [x, y]
        # 4. SequenceInsert(z, 1):  -> [x, z, y]
        # 5. SequenceAt(2):         -> y
        seq_empty_node = onnx.helper.make_node('SequenceEmpty', [], ['Seq_empty'])
        seq_insert_node = onnx.helper.make_node('SequenceInsert', ['Seq_empty', 'X'], ['Seq_1'])
        seq_insert_node2 = onnx.helper.make_node('SequenceInsert', ['Seq_1', 'Y'], ['Seq_2'])
        seq_insert_node3 = onnx.helper.make_node('SequenceInsert', ['Seq_2', 'Z', 'pos'], ['Seq_3'])
        seq_at_node = onnx.helper.make_node('SequenceAt', ['Seq_3', 'pos_at'], ['out'])

        x_shape = [2, 3, 4]
        y_shape = [1, 3, 4]
        z_shape = [3, 3, 4]
        out_shape = [None, 3, 4]

        x = np.ones(x_shape, dtype=np.float32)
        y = np.zeros(y_shape, dtype=np.float32)
        z = np.ones(z_shape, dtype=np.float32) * 2

        pos = onnx.helper.make_tensor('pos', TensorProto.INT64, (), (1, ))
        pos_at = onnx.helper.make_tensor('pos_at', TensorProto.INT64, (), (2, ))

        graph = make_graph(
            [seq_empty_node, seq_insert_node, seq_insert_node2, seq_insert_node3, seq_at_node],
            [x_shape, y_shape, z_shape, [], []],  # type: ignore
            [out_shape],  # type: ignore
            ['X', 'Y', 'Z', 'pos', 'pos_at'],
            ['out'],
            [onnx.TensorProto.FLOAT] * 3 + [onnx.TensorProto.INT64] * 2,  # type: ignore
            [onnx.TensorProto.FLOAT],
            [pos, pos_at])
        model = onnx.helper.make_model(graph, producer_name='backend-test')
        expect(model, inputs=[x, y, z], outputs=[y], name="test_sequence_model1")

        #2nd testcase - erase and at.
        # 1. SequenceConstruct(x, y, z):    -> [x, y, z]
        # 2. SequenceErase(1):              -> [x, z]
        # 3. SequenceAt(1):                 -> z
        seq_construct_node = onnx.helper.make_node('SequenceConstruct', ['X', 'Y', 'Z'], ['seq_1'])
        seq_erase_node = onnx.helper.make_node('SequenceErase', ['seq_1', 'pos_erase'], ['seq_2'])
        seq_at_node = onnx.helper.make_node('SequenceAt', ['seq_2', 'pos_at'], ['out'])

        tensor_shape = [2, 3, 4]

        x = np.ones(tensor_shape, dtype=np.float32)
        y = np.zeros(tensor_shape, dtype=np.float32)
        z = np.ones(tensor_shape, dtype=np.float32) * 2

        pos_erase = onnx.helper.make_tensor('pos_erase', TensorProto.INT64, (), (1, ))
        pos_at = onnx.helper.make_tensor('pos_at', TensorProto.INT64, (), (1, ))

        graph = make_graph(
            [seq_construct_node, seq_erase_node, seq_at_node],
            [tensor_shape, tensor_shape, tensor_shape, [], []],  # type: ignore
            [tensor_shape],  # type: ignore
            ['X', 'Y', 'Z', 'pos_erase', 'pos_at'],
            ['out'],
            [onnx.TensorProto.FLOAT] * 3 + [onnx.TensorProto.INT64] * 2,  # type: ignore
            [onnx.TensorProto.FLOAT],
            [pos_erase, pos_at])
        model = onnx.helper.make_model(graph, producer_name='backend-test')
        expect(model, inputs=[x, y, z], outputs=[z], name="test_sequence_model2")

        #3rd testcase - erase, insert and at, with negative index value.
        # 1. SequenceConstruct(x, y, z):    -> [x, y, z]
        # 2. SequenceErase(-3):             -> [y, z]
        # 3. SequenceInsert(x, -1):         -> [y, x, z]
        # 4. SequenceAt(-1):                -> z
        seq_construct_node = onnx.helper.make_node('SequenceConstruct', ['X', 'Y', 'Z'], ['seq_1'])
        seq_erase_node = onnx.helper.make_node('SequenceErase', ['seq_1', 'pos_erase'], ['seq_2'])
        seq_insert_node = onnx.helper.make_node('SequenceInsert', ['seq_2', 'X', 'pos_insert'], ['seq_3'])
        seq_at_node = onnx.helper.make_node('SequenceAt', ['seq_3', 'pos_at'], ['out'])

        tensor_shape = [2, 3, 4]

        x = np.ones(tensor_shape, dtype=np.float32)
        y = np.zeros(tensor_shape, dtype=np.float32)
        z = np.ones(tensor_shape, dtype=np.float32) * 2

        pos_erase = onnx.helper.make_tensor('pos_erase', TensorProto.INT64, (), (-3, ))
        pos_insert = onnx.helper.make_tensor('pos_insert', TensorProto.INT64, (), (-1, ))
        pos_at = onnx.helper.make_tensor('pos_at', TensorProto.INT64, (), (-1, ))

        graph = make_graph(
            [seq_construct_node, seq_erase_node, seq_insert_node, seq_at_node],
            [tensor_shape, tensor_shape, tensor_shape, [], [], []],  # type: ignore
            [tensor_shape],  # type: ignore
            ['X', 'Y', 'Z', 'pos_erase', 'pos_insert', 'pos_at'],
            ['out'],
            [onnx.TensorProto.FLOAT] * 3 + [onnx.TensorProto.INT64] * 3,  # type: ignore
            [onnx.TensorProto.FLOAT],
            [pos_erase, pos_insert, pos_at])
        model = onnx.helper.make_model(graph, producer_name='backend-test')
        expect(model, inputs=[x, y, z], outputs=[z], name="test_sequence_model3")

        #4th testcase - concat
        seq_construct_node = onnx.helper.make_node('SequenceConstruct', ['X', 'Y', 'Z'], ['seq_1'])
        seq_concat_node = onnx.helper.make_node('ConcatFromSequence', ['seq_1'], ['out'], axis=1)

        tensor_shape = [2, 3, 4]
        concat_out_shape = [2, None, 4]

        x = np.ones(tensor_shape, dtype=np.float32)
        y = np.zeros(tensor_shape, dtype=np.float32)
        z = np.ones(tensor_shape, dtype=np.float32) * 2
        concat_out = np.concatenate((x, y, z), 1)

        graph = make_graph(
            [seq_construct_node, seq_concat_node],
            [tensor_shape] * 3,  # type: ignore
            [concat_out_shape],  # type: ignore
            ['X', 'Y', 'Z'],
            ['out'],
            [onnx.TensorProto.FLOAT] * 3,  # type: ignore
            [onnx.TensorProto.FLOAT])
        model = onnx.helper.make_model(graph, producer_name='backend-test')
        expect(model, inputs=[x, y, z], outputs=[concat_out], name="test_sequence_model4")

        #5th testcase - concat with new_axis = 1
        seq_construct_node = onnx.helper.make_node('SequenceConstruct', ['X', 'Y', 'Z'], ['seq_1'])
        seq_concat_node = onnx.helper.make_node('ConcatFromSequence', ['seq_1'], ['out'], axis=-1, new_axis=1)

        tensor_shape = [2, 3, 4]
        concat_out_shape = [2, 3, 4, 3]

        x = np.ones(tensor_shape, dtype=np.float32)
        y = np.zeros(tensor_shape, dtype=np.float32)
        z = np.ones(tensor_shape, dtype=np.float32) * 2
        concat_out = np.stack((x, y, z), -1)

        graph = make_graph(
            [seq_construct_node, seq_concat_node],
            [tensor_shape] * 3,  # type: ignore
            [concat_out_shape],  # type: ignore
            ['X', 'Y', 'Z'],
            ['out'],
            [onnx.TensorProto.FLOAT] * 3,  # type: ignore
            [onnx.TensorProto.FLOAT],)
        model = onnx.helper.make_model(graph, producer_name='backend-test')
        expect(model, inputs=[x, y, z], outputs=[concat_out], name="test_sequence_model5")

        #6th testcase - split and len
        seq_split_node = onnx.helper.make_node('SplitToSequence', ['X'], ['seq_1'], axis=-1)
        seq_len_node = onnx.helper.make_node('SequenceLength', ['seq_1'], ['len'])

        tensor_shape = [2, 3, 4]
        len_shape = []  # type: ignore

        x = np.ones(tensor_shape, dtype=np.float32)
        out_len = np.int64(4)

        graph = onnx.helper.make_graph(
            nodes=[seq_split_node, seq_len_node],
            name='Sequence',
            inputs=[
                onnx.helper.make_tensor_value_info(
                    'X',
                    onnx.TensorProto.FLOAT,
                    tensor_shape)],
            outputs=[
                onnx.helper.make_tensor_value_info(
                    'len',
                    onnx.TensorProto.INT64,
                    len_shape)])  # type: ignore

        model = onnx.helper.make_model(graph, producer_name='backend-test')
        expect(model, inputs=[x], outputs=[out_len], name="test_sequence_model6")

        #7th testcase - split with keepdims=0, and SequenceAt
        seq_split_node = onnx.helper.make_node('SplitToSequence', ['X'], ['seq_1'], axis=0, keepdims=0)
        seq_at_node = onnx.helper.make_node('SequenceAt', ['seq_1', 'pos_at'], ['out'])

        tensor_shape = [2, 3, 4]
        out_shape = [3, 4]

        x = np.random.rand(*tensor_shape)
        out = list(x)[1]
        pos_at = onnx.helper.make_tensor('pos_at', TensorProto.INT64, (), (1, ))

        graph = make_graph(
            [seq_split_node, seq_at_node],
            [tensor_shape, []],  # type: ignore
            [out_shape],  # type: ignore
            ['X', 'pos_at'],
            ['out'],
            [onnx.TensorProto.DOUBLE, onnx.TensorProto.INT64],
            [onnx.TensorProto.DOUBLE],
            [pos_at])
        model = onnx.helper.make_model(graph, producer_name='backend-test')
        expect(model, inputs=[x], outputs=[out], name="test_sequence_model7")
