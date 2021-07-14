# SPDX-License-Identifier: Apache-2.0

import onnx
from onnx import helper, TensorProto, shape_inference, version_converter, ValueInfoProto
from typing import Text, List, Dict, Any, Union, Callable, Optional, cast
import string
import numpy as np  # type: ignore
import unittest

#####################################################################################
# Every test creates a model containing a single operator from the lowest possible
# opset version, upgrades it to the most recent opset version and then runs checker +
# shape inference on the upgraded model.
####################################################################################

latest_opset = onnx.defs.onnx_opset_version()
tested_ops = []


class TestAutomaticUpgrade(unittest.TestCase):

    def _test_op_upgrade(
        self,
        op,  # type: Text
        from_opset,  # type: int
        input_shapes=[[3, 4, 5]],  # type: List[Union[List[Optional[int]], Text]]
        output_shapes=[[3, 4, 5]],  # type: List[List[Optional[int]]]
        input_types=None,  # type: Union[List[Any], None]
        output_types=None,  # type: Union[List[Any], None]
        initializer=[],  # type: List[Any]
        attrs={},  # type: Dict[Text, Any]
        seq_inputs=[],  # type: List[int]
        seq_outputs=[],  # type: List[int]
        optional_inputs=[],  # type: List[int]
        optional_outputs=[]  # type: List[int]
    ):  # type: (...) -> None
        global tested_ops
        tested_ops.append(op)

        n_inputs = len(input_shapes)
        letters = list(string.ascii_lowercase)[:n_inputs]
        input_names = [
            letter if shape != '' else '' for (letter, shape) in zip(letters, input_shapes)
        ]
        if input_types is None:
            input_types = [TensorProto.FLOAT] * n_inputs
        is_sequence = [0 if id not in seq_inputs else 1 for id in range(n_inputs)]
        is_optional = [0 if id not in optional_inputs else 1 for id in range(n_inputs)]
        # turn empty strings into [0] to ease type analysis, even though those entries
        # will be ignored
        input_shapes_cast = cast(List[List[int]],
                [[0] if isinstance(shape, str) else shape for shape in input_shapes]
        )
        inputs = []  # type: List[ValueInfoProto]
        for (name, ttype, shape, is_seq, is_opt) in \
                zip(input_names, input_types, input_shapes_cast, is_sequence, is_optional):
            if name != '':
                if is_seq:
                    inputs += [helper.make_tensor_sequence_value_info(name, ttype, shape)]
                elif is_opt:
                    type_proto = helper.make_tensor_type_proto(ttype, shape)
                    optional_type_proto = helper.make_optional_type_proto(type_proto)
                    inputs += [helper.make_value_info(name, optional_type_proto)]
                else:
                    inputs += [helper.make_tensor_value_info(name, ttype, shape)]

        n_outputs = len(output_shapes)
        output_names = list(string.ascii_lowercase)[n_inputs:n_inputs + n_outputs]
        if output_types is None:
            output_types = [TensorProto.FLOAT] * n_outputs
        is_sequence = [0 if id not in seq_outputs else 1 for id in range(n_outputs)]
        is_optional = [0 if id not in optional_outputs else 1 for id in range(n_outputs)]
        output_shapes_cast = cast(List[List[int]],
                [[0] if isinstance(shape, str) else shape for shape in output_shapes]
        )
        outputs = []  # type: List[ValueInfoProto]
        for (name, ttype, shape, is_seq, is_opt) in \
                zip(output_names, output_types, output_shapes_cast, is_sequence, is_optional):
            if is_seq:
                outputs += [helper.make_tensor_sequence_value_info(name, ttype, shape)]
            elif is_opt:
                type_proto = helper.make_tensor_type_proto(ttype, shape)
                optional_type_proto = helper.make_optional_type_proto(type_proto)
                outputs += [helper.make_value_info(name, optional_type_proto)]
            else:
                outputs += [helper.make_tensor_value_info(name, ttype, shape)]

        node = helper.make_node(op, input_names, output_names, **attrs)
        graph = helper.make_graph([node], op, inputs, outputs, initializer)
        original = helper.make_model(
            graph,
            producer_name='test',
            opset_imports=[helper.make_opsetid('', from_opset)]
        )
        onnx.checker.check_model(original)
        shape_inference.infer_shapes(original, strict_mode=True)

        converted = version_converter.convert_version(original, latest_opset)
        onnx.checker.check_model(converted)
        shape_inference.infer_shapes(converted, strict_mode=True)

    def test_Abs(self):  # type: () -> None
        self._test_op_upgrade('Abs', 1, attrs={'consumed_inputs': [0]})

    def test_Acosh(self):  # type: () -> None
        self._test_op_upgrade('Acosh', 9)

    def test_Acos(self):  # type: () -> None
        self._test_op_upgrade('Acos', 7)

    def test_And(self):  # type: () -> None
        # 6->7 adapter is missing
        self._test_op_upgrade('And', 7, [[2, 3], [2, 3]], [[2, 3]],
            [TensorProto.BOOL, TensorProto.BOOL], [TensorProto.BOOL]
        )

    def test_Asinh(self):  # type: () -> None
        self._test_op_upgrade('Asinh', 9)

    def test_Atanh(self):  # type: () -> None
        self._test_op_upgrade('Atanh', 9)

    def test_Add_1(self):  # type: () -> None
        self._test_op_upgrade('Add', 1,
            [[3, 4, 5], [3, 4, 5]],
            attrs={'consumed_inputs': [0]}
        )

    def test_Add_2(self):  # type: () -> None
        self._test_op_upgrade('Add', 1, [[3, 4, 5], [5]],
            attrs={'consumed_inputs': [0], 'broadcast': 1}
        )

    def test_Add_3(self):  # type: () -> None
        self._test_op_upgrade('Add', 1, [[3, 4, 5], [3]],
            attrs={'consumed_inputs': [0], 'broadcast': 1, 'axis': 0}
        )

    def test_ArgMax_1(self):  # type: () -> None
        self._test_op_upgrade('ArgMax', 7, [[2, 3, 4]], [[1, 3, 4]],
            output_types=[TensorProto.INT64]
        )

    def test_ArgMax_2(self):  # type: () -> None
        self._test_op_upgrade('ArgMax', 7, [[2, 3, 4]], [[2, 1, 4]],
            output_types=[TensorProto.INT64],
            attrs={'axis': 1}
        )

    def test_ArgMin_1(self):  # type: () -> None
        self._test_op_upgrade('ArgMin', 7, [[2, 3, 4]], [[1, 3, 4]],
            output_types=[TensorProto.INT64]
        )

    def test_ArgMin_2(self):  # type: () -> None
        self._test_op_upgrade('ArgMin', 7, [[2, 3, 4]], [[2, 1, 4]],
            output_types=[TensorProto.INT64],
            attrs={'axis': 1}
        )

    def test_Asin(self):  # type: () -> None
        self._test_op_upgrade('Asin', 7)

    def test_Atan(self):  # type: () -> None
        self._test_op_upgrade('Atan', 7)

    def test_AveragePool(self):  # type: () -> None
        self._test_op_upgrade('AveragePool', 1, [[1, 1, 5, 5]], [[1, 1, 4, 4]],
            attrs={'kernel_shape': [2, 2]}
        )

    def test_Bernoulli(self):  # type: () -> None
        self._test_op_upgrade('Bernoulli', 15)

    def test_BitShift(self):  # type: () -> None
        self._test_op_upgrade('BitShift', 11, [[2, 3], [2, 3]], [[2, 3]],
            [TensorProto.UINT8, TensorProto.UINT8], [TensorProto.UINT8],
            attrs={'direction': 'RIGHT'}
        )

    def test_BatchNormalization_1(self):  # type: () -> None
        self._test_op_upgrade('BatchNormalization', 1, [[1, 3], [3], [3], [3], [3]], [[1, 3]],
            attrs={'consumed_inputs': [1, 1], 'is_test': 1, 'spatial': 1}
        )

    def test_BatchNormalization_2(self):  # type: () -> None
        self._test_op_upgrade('BatchNormalization', 14,
            [[1, 3], [3], [3], [3], [3]], [[1, 3], [3], [3]],
            attrs={'training_mode': 1}
        )

    def test_Cast(self):  # type: () -> None
        # 5->6 adapter is missing
        self._test_op_upgrade('Cast', 6, [[2, 3]], [[2, 3]], [TensorProto.INT64], attrs={'to': 1})

    def test_Ceil(self):  # type: () -> None
        self._test_op_upgrade('Ceil', 1, attrs={'consumed_inputs': [0]})

    def test_Celu(self):  # type: () -> None
        self._test_op_upgrade('Celu', 12)

    def test_Clip_1(self):  # type: () -> None
        self._test_op_upgrade('Clip', 1, attrs={'consumed_inputs': [0]})

    def test_Clip_2(self):  # type: () -> None
        self._test_op_upgrade('Clip', 1, attrs={'consumed_inputs': [0], 'min': -1.4})

    def test_Clip_3(self):  # type: () -> None
        self._test_op_upgrade('Clip', 1, attrs={'consumed_inputs': [0], 'max': 2.6})

    def test_Clip_4(self):  # type: () -> None
        self._test_op_upgrade('Clip', 1, attrs={'consumed_inputs': [0], 'min': -1.4, 'max': 2.6})

    def test_Compress(self):  # type: () -> None
        self._test_op_upgrade('Compress', 9, [[6, 7], [3]], [[3]],
            [TensorProto.FLOAT, TensorProto.BOOL], [TensorProto.FLOAT]
        )

    def test_Concat(self):  # type: () -> None
        self._test_op_upgrade('Concat', 1, [[2, 3], [2, 4]], [[2, 7]])

    def test_constant(self):  # type: () -> None
        value = helper.make_tensor(
            'Value',
            TensorProto.FLOAT,
            dims=[3, 4, 5],
            vals=np.random.rand(3, 4, 5).astype(np.float32).tobytes(),
            raw=True
        )
        self._test_op_upgrade('Constant', 1, [], attrs={'value': value})

    def test_ConstantOfShape(self):  # type: () -> None
        self._test_op_upgrade('ConstantOfShape', 9, [[3]])

    def test_Conv_1(self):  # type: () -> None
        self._test_op_upgrade('Conv', 1, [[1, 3, 5, 5], [4, 3, 2, 2], [4]], [[1, 4, 4, 4]])

    def test_Conv_2(self):  # type: () -> None
        self._test_op_upgrade('Conv', 1, [[1, 3, 5, 5], [4, 3, 2, 2], [4]], [[1, 4, 4, 4]])

    def test_Conv_3(self):  # type: () -> None
        self._test_op_upgrade('Conv', 1, [[1, 3, 5, 5], [4, 1, 2, 2], [4]], [[1, 4, 3, 7]],
            attrs={'dilations': [1, 2], 'group': 3, 'pads': [0, 1, 2, 3], 'strides': [2, 1]})

    def test_Convinteger(self):  # type: () -> None
        self._test_op_upgrade('ConvInteger', 10, [[1, 3, 5, 5], [4, 3, 2, 2], [4]], [[1, 4, 4, 4]],
            [TensorProto.UINT8, TensorProto.UINT8, TensorProto.UINT8], [TensorProto.INT32]
        )

    def test_ConvTranspose(self):  # type: () -> None
        self._test_op_upgrade('ConvTranspose', 1, [[1, 1, 5, 5], [1, 1, 3, 3]], [[1, 1, 7, 7]])

    def test_Cosh(self):  # type: () -> None
        self._test_op_upgrade('Cosh', 9)

    def test_Cos(self):  # type: () -> None
        self._test_op_upgrade('Cos', 7)

    def test_Cumsum(self):  # type: () -> None
        self._test_op_upgrade('CumSum', 11, [[3, 4, 5], []], [[3, 4, 5]],
            [TensorProto.FLOAT, TensorProto.INT64]
        )

    def test_DepthToSpace(self):  # type: () -> None
        self._test_op_upgrade('DepthToSpace', 1, [[1, 8, 3, 3]], [[1, 2, 6, 6]],
            attrs={'blocksize': 2}
        )

    def test_DequantizeLinear(self):  # type: () -> None
        self._test_op_upgrade('DequantizeLinear', 10, [[2, 3], [], []], [[2, 3]],
            [TensorProto.INT8, TensorProto.FLOAT, TensorProto.INT8]
        )

    def test_Det_1(self):  # type: () -> None
        self._test_op_upgrade('Det', 11, [[3, 5, 5]], [[3]])

    def test_Det_2(self):  # type: () -> None
        self._test_op_upgrade('Det', 11, [[5, 5]], [[]])

    def test_DynamicQuantizeLinear(self):  # type: () -> None
        self._test_op_upgrade('DynamicQuantizeLinear', 11, [[3, 4, 5]], [[3, 4, 5], [], []],
            output_types=[TensorProto.UINT8, TensorProto.FLOAT, TensorProto.UINT8]
        )

    def test_Div(self):  # type: () -> None
        self._test_op_upgrade('Div', 1, [[3, 4, 5], [3, 1, 5]], attrs={'consumed_inputs': [0]})

    def test_Dropout(self):  # type: () -> None
        self._test_op_upgrade('Dropout', 1, attrs={'consumed_inputs': [0], 'is_test': 1})

    def test_Einsum_1(self):  # type: () -> None
        self._test_op_upgrade('Einsum', 12, [[3, 4, 5], [3, 5, 6]], [[3, 4, 6]],
            attrs={'equation': 'bij, bjk -> bik'}
        )

    def test_Einsum_2(self):  # type: () -> None
        self._test_op_upgrade('Einsum', 12, [[4, 5]], [[5, 4]],
            attrs={'equation': 'ij->ji'}
        )

    def test_Elu(self):  # type: () -> None
        self._test_op_upgrade('Elu', 1, attrs={'consumed_inputs': [0]})

    def test_Equal(self):  # type: () -> None
        # 6->7 adapter is missing
        self._test_op_upgrade('Equal', 7, [[2, 3], [2, 3]], [[2, 3]], output_types=[TensorProto.BOOL])

    def test_Erf(self):  # type: () -> None
        self._test_op_upgrade('Erf', 9)

    def test_Exp(self):  # type: () -> None
        self._test_op_upgrade('Exp', 1, attrs={'consumed_inputs': [0]})

    def test_Expand(self):  # type: () -> None
        shape = helper.make_tensor(
            'b',
            TensorProto.INT64,
            dims=[4],
            vals=np.array([5, 2, 6, 4])
        )
        self._test_op_upgrade('Expand', 8, [[2, 1, 4], [4]], [[5, 2, 6, 4]],
            [TensorProto.FLOAT, TensorProto.INT64],
            initializer=[shape]
        )

    def test_EyeLike(self):  # type: () -> None
        self._test_op_upgrade('EyeLike', 9, [[4, 5]], [[4, 5]])

    def test_Flatten(self):  # type: () -> None
        self._test_op_upgrade('Flatten', 1, [[3, 4, 5]], [[3, 20]], attrs={'axis': 1})

    def test_Floor(self):  # type: () -> None
        self._test_op_upgrade('Floor', 1, attrs={'consumed_inputs': [0]})

    def test_Gather(self):  # type: () -> None
        self._test_op_upgrade('Gather', 1, [[3, 4, 5], [6, 7]], [[6, 7, 4, 5]],
            [TensorProto.FLOAT, TensorProto.INT64]
        )

    def test_GatherElements(self):  # type: () -> None
        self._test_op_upgrade('GatherElements', 11, [[3, 4, 5], [6, 7]], [[6, 7]],
            [TensorProto.FLOAT, TensorProto.INT64]
        )

    def test_GatherND(self):  # type: () -> None
        self._test_op_upgrade('GatherND', 11, [[1, 2, 3], [1, 2, 3]], [[1, 2]])

    def test_Gemm(self):  # type: () -> None
        self._test_op_upgrade('Gemm', 1, [[5, 4], [4, 3], [3]], [[5, 3]])

    def test_GlobalAveragePool(self):  # type: () -> None
        self._test_op_upgrade('GlobalAveragePool', 1, [[1, 3, 10, 10]], [[1, 3, 1, 1]])

    def test_GlobalMaxPool(self):  # type: () -> None
        self._test_op_upgrade('GlobalMaxPool', 1, [[1, 3, 10, 10]], [[1, 3, 1, 1]])

    def test_GlobalLpPool(self):  # type: () -> None
        # 1->2 adapter is missing
        self._test_op_upgrade('GlobalLpPool', 2, [[1, 3, 10, 10]], [[1, 3, 1, 1]])

    def test_Greater(self):  # type: () -> None
        # 6->7 adapter is missing
        self._test_op_upgrade('Greater', 7, [[2, 3], [2, 3]], [[2, 3]],
            output_types=[TensorProto.BOOL]
        )

    def test_GreaterOrEqual(self):  # type: () -> None
        self._test_op_upgrade('GreaterOrEqual', 12, [[2, 3], [2, 3]], [[2, 3]],
            output_types=[TensorProto.BOOL]
        )

    def test_GRU_1(self):  # type: () -> None
        # 2->3, 6->7 adapters are missing
        self._test_op_upgrade('GRU', 7,
            [[5, 3, 4], [1, 18, 4], [1, 18, 4]], [[5, 1, 3, 6], [1, 3, 6]],
            attrs={'hidden_size': 6}
        )

    def test_GRU_2(self):  # type: () -> None
        # 2->3, 6->7 adapters are missing
        self._test_op_upgrade('GRU', 7,
            [[5, 3, 4], [2, 18, 4], [2, 18, 4]], [[5, 2, 3, 6], [2, 3, 6]],
            attrs={'hidden_size': 6, 'direction': 'bidirectional'}
        )

    def test_GRU_3(self):  # type: () -> None
        # 2->3, 6->7 adapters are missing
        self._test_op_upgrade('GRU', 7,
            [[5, 3, 4], [1, 18, 4], [1, 18, 4], [1, 24], [5], [1, 5, 6]],
            [[5, 1, 3, 6], [1, 3, 6]],
            [TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.INT64, TensorProto.FLOAT],
            attrs={'hidden_size': 6}
        )

    def test_HardSigmoid(self):  # type: () -> None
        self._test_op_upgrade('HardSigmoid', 1, attrs={'consumed_inputs': [0]})

    def test_HardSwish(self):  # type: () -> None
        self._test_op_upgrade('HardSwish', 14)

    def test_Hardmax(self):  # type: () -> None
        self._test_op_upgrade('Hardmax', 1)

    def test_Identity(self):  # type: () -> None
        self._test_op_upgrade('Identity', 1)

    def test_If(self):  # type: () -> None
        sub_output = [helper.make_tensor_value_info('out', TensorProto.FLOAT, [3, 4, 5])]
        then_tensor = helper.make_tensor(
            'Value',
            TensorProto.FLOAT,
            dims=[3, 4, 5],
            vals=np.random.rand(3, 4, 5).astype(np.float32).tobytes(), raw=True
        )
        then_node = helper.make_node('Constant', [], ['out'], value=then_tensor)
        then_graph = helper.make_graph([then_node], 'then_graph', [], sub_output, [])
        else_tensor = helper.make_tensor(
            'Value',
            TensorProto.FLOAT,
            dims=[3, 4, 5],
            vals=np.random.rand(3, 4, 5).astype(np.float32).tobytes(), raw=True
        )
        else_node = helper.make_node('Constant', [], ['out'], value=else_tensor)
        else_graph = helper.make_graph([else_node], 'else_graph', [], sub_output, [])
        self._test_op_upgrade('If', 1, [[0]], [[3, 4, 5]], [TensorProto.BOOL],
            attrs={'then_branch': then_graph, 'else_branch': else_graph}
        )

    def test_InstanceNormalization(self):  # type: () -> None
        self._test_op_upgrade('InstanceNormalization', 1, [[1, 3], [3], [3]], [[1, 3]],
            attrs={'consumed_inputs': [0]}
        )

    def test_IsInf(self):  # type: () -> None
        self._test_op_upgrade('IsInf', 10, [[2, 3]], [[2, 3]], output_types=[TensorProto.BOOL])

    def test_IsNaN(self):  # type: () -> None
        self._test_op_upgrade('IsNaN', 9, [[2, 3]], [[2, 3]], output_types=[TensorProto.BOOL])

    def test_LeakyRelu(self):  # type: () -> None
        self._test_op_upgrade('LeakyRelu', 1, attrs={'consumed_inputs': [0]})

    def test_Less(self):  # type: () -> None
        # 6->7 adapter is missing
        self._test_op_upgrade('Less', 7, [[2, 3], [2, 3]], [[2, 3]], output_types=[TensorProto.BOOL])

    def test_LessOrEqual(self):  # type: () -> None
        self._test_op_upgrade('LessOrEqual', 12, [[2, 3], [2, 3]], [[2, 3]],
            output_types=[TensorProto.BOOL]
        )

    def test_Log(self):  # type: () -> None
        self._test_op_upgrade('Log', 1, attrs={'consumed_inputs': [0]})

    def test_LogSoftmax(self):  # type: () -> None
        self._test_op_upgrade('LogSoftmax', 1)

    def test_Loop_1(self):  # type: () -> None
        iter_count = onnx.helper.make_tensor_value_info('iter_count', onnx.TensorProto.INT64, [])
        cond_in = onnx.helper.make_tensor_value_info('cond_in', onnx.TensorProto.BOOL, [])
        x_in = onnx.helper.make_tensor_value_info('x_in', onnx.TensorProto.FLOAT, [1])
        cond_out = onnx.helper.make_tensor_value_info('cond_out', onnx.TensorProto.BOOL, [])
        x_out = onnx.helper.make_tensor_value_info('x_out', onnx.TensorProto.FLOAT, [1])
        x_scan = onnx.helper.make_tensor_value_info('x_scan', onnx.TensorProto.FLOAT, [1])
        const = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['one'],
            value=onnx.helper.make_tensor(
                name='value',
                data_type=onnx.TensorProto.FLOAT,
                dims=[1],
                vals=np.array([1]).astype(np.float32).astype(float),
            )
        )
        add = onnx.helper.make_node(
            'Add',
            inputs=['x_in', 'one'],
            outputs=['x_out']
        )
        id_1 = onnx.helper.make_node(
            'Identity',
            inputs=['x_out'],
            outputs=['x_scan']
        )
        id_2 = onnx.helper.make_node(
            'Identity',
            inputs=['cond_in'],
            outputs=['cond_out']
        )
        loop_body = onnx.helper.make_graph(
            [const, add, id_1, id_2],
            'loop_body',
            [iter_count, cond_in, x_in],
            [cond_out, x_out, x_scan]
        )
        self._test_op_upgrade('Loop', 1, [[], '', [1]], [[1], [5, 1]],
            [TensorProto.INT64, TensorProto.BOOL, TensorProto.FLOAT],
            attrs={'body': loop_body}
        )

    def test_Loop_2(self):  # type: () -> None
        iter_count = onnx.helper.make_tensor_value_info('iter_count', onnx.TensorProto.INT64, [])
        cond_in = onnx.helper.make_tensor_value_info('cond_in', onnx.TensorProto.BOOL, [])
        x_in = onnx.helper.make_tensor_value_info('x_in', onnx.TensorProto.FLOAT, [2, 1])
        cond_out = onnx.helper.make_tensor_value_info('cond_out', onnx.TensorProto.BOOL, [])
        x_out = onnx.helper.make_tensor_value_info('x_out', onnx.TensorProto.FLOAT, [2, 1])
        squeeze = onnx.helper.make_node(
            'Squeeze',
            inputs=['x_in'],
            outputs=['squeeze_out'],
            axes=[1]
        )
        unsqueeze = onnx.helper.make_node(
            'Unsqueeze',
            inputs=['squeeze_out'],
            outputs=['x_out'],
            axes=[1]
        )
        identity = onnx.helper.make_node(
            'Identity',
            inputs=['cond_in'],
            outputs=['cond_out']
        )
        loop_body = onnx.helper.make_graph(
            [squeeze, unsqueeze, identity],
            'loop_body',
            [iter_count, cond_in, x_in],
            [cond_out, x_out]
        )
        self._test_op_upgrade('Loop', 12, [[], '', [2, 1]], [[2, 1]],
            [TensorProto.INT64, TensorProto.BOOL, TensorProto.FLOAT],
            attrs={'body': loop_body}
        )

    def test_LpNormalization(self):  # type: () -> None
        self._test_op_upgrade('LpNormalization', 1)

    def test_LpPool(self):  # type: () -> None
        # 1->2 adapter is missing
        self._test_op_upgrade('LpPool', 2, [[1, 1, 5, 5]], [[1, 1, 4, 4]],
            attrs={'kernel_shape': [2, 2]}
        )

    def test_LRN_1(self):  # type: () -> None
        self._test_op_upgrade('LRN', 1, attrs={'size': 3})

    def test_LRN_2(self):  # type: () -> None
        self._test_op_upgrade('LRN', 1, [[2, 3, 4, 5]], [[2, 3, 4, 5]],
            attrs={'size': 3}
        )

    def test_LSTM_1(self):  # type: () -> None
        # 6->7 adapter is missing
        self._test_op_upgrade('LSTM', 7,
            [[5, 3, 4], [1, 24, 4], [1, 24, 4]],
            [[5, 1, 3, 6], [1, 3, 6], [1, 3, 6]],
            attrs={'hidden_size': 6}
        )

    def test_LSTM_2(self):  # type: () -> None
        # 6->7 adapter is missing
        self._test_op_upgrade('LSTM', 7,
            [[5, 3, 4], [2, 24, 4], [2, 24, 4]],
            [[5, 2, 3, 6], [2, 3, 6], [2, 3, 6]],
            attrs={'hidden_size': 6, 'direction': 'bidirectional'}
        )

    def test_LSTM_3(self):  # type: () -> None
        # 6->7 adapter is missing
        self._test_op_upgrade('LSTM', 7,
            [[5, 3, 4], [1, 24, 4], [1, 24, 4], [1, 48], [5], [1, 5, 6], [1, 5, 6], [1, 18]],
            [[5, 1, 3, 6], [1, 3, 6], [1, 3, 6]],
            [TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.INT64, TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.FLOAT],
            attrs={'hidden_size': 6}
        )

    def test_MatMul_1(self):  # type: () -> None
        self._test_op_upgrade('MatMul', 1, [[2, 3], [3, 4]], [[2, 4]])

    def test_MatMul_2(self):  # type: () -> None
        self._test_op_upgrade('MatMul', 1, [[5, 2, 3], [5, 3, 4]], [[5, 2, 4]])

    def test_MatMulInteger_1(self):  # type: () -> None
        self._test_op_upgrade('MatMulInteger', 10, [[2, 3], [3, 4]], [[2, 4]],
            [TensorProto.INT8, TensorProto.INT8], [TensorProto.INT32]
        )

    def test_MatMulInteger_2(self):  # type: () -> None
        self._test_op_upgrade('MatMulInteger', 10, [[2, 3], [3, 4], [], []], [[2, 4]],
            [TensorProto.INT8, TensorProto.INT8, TensorProto.INT8, TensorProto.INT8],
            [TensorProto.INT32]
        )

    def test_MatMulInteger_3(self):  # type: () -> None
        self._test_op_upgrade('MatMulInteger', 10, [[2, 3], [3, 4], [2], [4]], [[2, 4]],
            [TensorProto.INT8, TensorProto.INT8, TensorProto.INT8, TensorProto.INT8],
            [TensorProto.INT32]
        )

    def test_Max(self):  # type: () -> None
        self._test_op_upgrade('Max', 1, [[2, 3, 4], [2, 3, 4]], [[2, 3, 4]],
            attrs={'consumed_inputs': [0]}
        )

    def test_MaxPool_1(self):  # type: () -> None
        self._test_op_upgrade('MaxPool', 1, [[1, 1, 5, 5]], [[1, 1, 4, 4]],
            attrs={'kernel_shape': [2, 2]}
        )

    def test_MaxPool_2(self):  # type: () -> None
        self._test_op_upgrade('MaxPool', 8, [[1, 1, 5, 5]], [[1, 1, 4, 4], [1, 1, 4, 4]],
            output_types=[TensorProto.FLOAT, TensorProto.INT64],
            attrs={'kernel_shape': [2, 2]}
        )

    def test_MaxRoiPool(self):  # type: () -> None
        self._test_op_upgrade('MaxRoiPool', 1, [[2, 3, 20, 20], [4, 5]], [[4, 3, 3, 3]],
            attrs={'pooled_shape': [3, 3]}
        )

    def test_MaxUnpool(self):  # type: () -> None
        self._test_op_upgrade('MaxUnpool', 9, [[1, 1, 5, 5], [1, 1, 5, 5]], [[1, 1, 6, 6]],
            [TensorProto.FLOAT, TensorProto.INT64],
            attrs={'kernel_shape': [2, 2]}
        )

    def test_Mean(self):  # type: () -> None
        self._test_op_upgrade('Mean', 1, [[2, 3, 4], [2, 3, 4]], [[2, 3, 4]],
            attrs={'consumed_inputs': [0]}
        )

    def test_MeanVarianceNormalization(self):  # type: () -> None
        self._test_op_upgrade('MeanVarianceNormalization', 9, attrs={'axes': [1, 2]})

    def test_Min(self):  # type: () -> None
        self._test_op_upgrade('Min', 1, [[2, 3, 4], [2, 3, 4]], [[2, 3, 4]],
            attrs={'consumed_inputs': [0]}
        )

    def test_Mod_1(self):  # type: () -> None
        self._test_op_upgrade('Mod', 10, [[2, 3], [2, 3]], [[2, 3]])

    def test_Mod_2(self):  # type: () -> None
        self._test_op_upgrade('Mod', 10, [[2, 3], [2, 3]], [[2, 3]], attrs={'fmod': 1})

    def test_Mul(self):  # type: () -> None
        self._test_op_upgrade('Mul', 1, [[2, 3, 4], [2, 1, 4]], [[2, 3, 4]],
            attrs={'consumed_inputs': [0]}
        )

    def test_Multinomial(self):  # type: () -> None
        self._test_op_upgrade('Multinomial', 7, [[3, 5]], [[3, 7]],
            output_types=[TensorProto.INT32],
            attrs={'sample_size': 7}
        )

    def test_Neg(self):  # type: () -> None
        self._test_op_upgrade('Neg', 1, attrs={'consumed_inputs': [0]})

    def test_NegativeLogLikelihoodLoss_1(self):  # type: () -> None
        self._test_op_upgrade('NegativeLogLikelihoodLoss', 12, [[3, 4, 5], [3, 5]], [[]],
            [TensorProto.FLOAT, TensorProto.INT64]
        )

    def test_NegativeLogLikelihoodLoss_2(self):  # type: () -> None
        self._test_op_upgrade('NegativeLogLikelihoodLoss', 12, [[3, 4, 5], [3, 5], [4]], [[]],
            [TensorProto.FLOAT, TensorProto.INT64, TensorProto.FLOAT]
        )

    def test_NonMaxSuppression(self):  # type: () -> None
        self._test_op_upgrade('NonMaxSuppression', 10, [[2, 3, 4], [3, 5, 6]], [[2, 3]],
            output_types=[TensorProto.INT64]
        )

    def test_NonZero(self):  # type: () -> None
        self._test_op_upgrade('NonZero', 9, [[3, 3]], [[2, 4]], output_types=[TensorProto.INT64])

    def test_Not(self):  # type: () -> None
        self._test_op_upgrade('Not', 1, [[2, 3]], [[2, 3]], [TensorProto.BOOL], [TensorProto.BOOL])

    def test_OneHot(self):  # type: () -> None
        self._test_op_upgrade('OneHot', 9, [[3, 4, 5], [], [2]], [[3, 4, 5, 6]])

    def test_Or(self):  # type: () -> None
        # 6->7 adapter is missing
        self._test_op_upgrade('Or', 7, [[2, 3], [2, 3]], [[2, 3]],
            [TensorProto.BOOL, TensorProto.BOOL], [TensorProto.BOOL]
        )

    def test_Pad(self):  # type: () -> None
        # 1->2 adapter is missing
        self._test_op_upgrade('Pad', 2, [[3, 4]], [[5, 8]],
            attrs={'pads': [1, 2, 1, 2], 'value': 1.5}
        )

    def test_Pow(self):  # type: () -> None
        self._test_op_upgrade('Pow', 1, [[2, 3, 4], [2, 3, 4]], [[2, 3, 4]])

    def test_PRelu(self):  # type: () -> None
        self._test_op_upgrade('PRelu', 1, [[2, 3, 4], [2, 3, 4]], [[2, 3, 4]],
            attrs={'consumed_inputs': [0]}
        )

    def test_QLinearConv(self):  # type: () -> None
        self._test_op_upgrade('QLinearConv', 10,
            [[1, 3, 5, 5], [], [], [4, 3, 2, 2], [], [], [], []], [[1, 4, 4, 4]]
        )

    def test_QLinearMatMul(self):  # type: () -> None
        self._test_op_upgrade('QLinearMatMul', 10, [[2, 3], [], [], [3, 4], [], [], [], []], [[2, 4]])

    def test_QuantizeLinear(self):  # type: () -> None
        self._test_op_upgrade('QuantizeLinear', 10, [[3, 4, 5], [], []], [[3, 4, 5]],
            [TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.UINT8], [TensorProto.UINT8]
        )

    def test_RandomNormal(self):  # type: () -> None
        self._test_op_upgrade('RandomNormal', 1, [], [[3, 4, 5]], attrs={'shape': [3, 4, 5]})

    def test_RandomNormalLike(self):  # type: () -> None
        like = helper.make_tensor(
            'a',
            TensorProto.FLOAT,
            dims=[3, 4, 5],
            vals=np.random.rand(3, 4, 5).astype(np.float32).tobytes(), raw=True
        )
        self._test_op_upgrade('RandomNormalLike', 1, [[3, 4, 5]], [[3, 4, 5]],
            initializer=[like]
        )

    def test_RandomUniform(self):  # type: () -> None
        self._test_op_upgrade('RandomUniform', 1, [], [[3, 4, 5]], attrs={'shape': [3, 4, 5]})

    def test_RandomUniformLike(self):  # type: () -> None
        like = helper.make_tensor(
            'a',
            TensorProto.FLOAT,
            dims=[3, 4, 5],
            vals=np.random.rand(3, 4, 5).astype(np.float32).tobytes(), raw=True
        )
        self._test_op_upgrade('RandomUniformLike', 1, [[3, 4, 5]], [[3, 4, 5]],
            initializer=[like]
        )

    def test_Range(self):  # type: () -> None
        start = helper.make_tensor('a', TensorProto.FLOAT, dims=[], vals=np.array([0]))
        end = helper.make_tensor('b', TensorProto.FLOAT, dims=[], vals=np.array([12]))
        step = helper.make_tensor('c', TensorProto.FLOAT, dims=[], vals=np.array([2]))
        self._test_op_upgrade('Range', 11, [[], [], []], [[6]],
            initializer=[start, end, step]
        )

    def test_Reciprocal(self):  # type: () -> None
        self._test_op_upgrade('Reciprocal', 1, attrs={'consumed_inputs': [0]})

    def test_ReduceL1(self):  # type: () -> None
        self._test_op_upgrade('ReduceL1', 1, [[3, 4, 5]], [[1, 1, 1]])

    def test_ReduceL2(self):  # type: () -> None
        self._test_op_upgrade('ReduceL2', 1, [[3, 4, 5]], [[1, 1, 1]])

    def test_ReduceLogSum(self):  # type: () -> None
        self._test_op_upgrade('ReduceLogSum', 1, [[3, 4, 5]], [[1, 1, 1]])

    def test_ReduceLogSumExp(self):  # type: () -> None
        self._test_op_upgrade('ReduceLogSumExp', 1, [[3, 4, 5]], [[1, 1, 1]])

    def test_ReduceMean(self):  # type: () -> None
        self._test_op_upgrade('ReduceMean', 1, [[3, 4, 5]], [[1, 1, 1]])

    def test_ReduceMax(self):  # type: () -> None
        self._test_op_upgrade('ReduceMax', 1, [[3, 4, 5]], [[1, 1, 1]])

    def test_ReduceMin(self):  # type: () -> None
        self._test_op_upgrade('ReduceMin', 1, [[3, 4, 5]], [[1, 1, 1]])

    def test_ReduceProd(self):  # type: () -> None
        self._test_op_upgrade('ReduceProd', 1, [[3, 4, 5]], [[1, 1, 1]])

    def test_ReduceSum(self):  # type: () -> None
        self._test_op_upgrade('ReduceSum', 1, [[3, 4, 5]], [[1, 1, 1]])

    def test_ReduceSumSquare(self):  # type: () -> None
        self._test_op_upgrade('ReduceSumSquare', 1, [[3, 4, 5]], [[1, 1, 1]])

    def test_Relu(self):  # type: () -> None
        self._test_op_upgrade('Relu', 1, attrs={'consumed_inputs': [0]})

    def test_Reshape(self):  # type: () -> None
        self._test_op_upgrade('Reshape', 1, [[3, 4, 5]], [[3, 10, 2]],
            attrs={'consumed_inputs': [0], 'shape': [3, 10, 2]}
        )

    def test_Resize(self):  # type: () -> None
        self._test_op_upgrade('Resize', 10, [[3, 4, 5], [3]], [[3, 8, 15]])

    def test_ReverseSequence(self):  # type: () -> None
        self._test_op_upgrade('ReverseSequence', 10, [[3, 4, 5], [4]], [[3, 4, 5]],
            [TensorProto.FLOAT, TensorProto.INT64]
        )

    def test_RNN_1(self):  # type: () -> None
        # 6->7 adapter is missing
        self._test_op_upgrade('RNN', 7, [[5, 3, 4], [1, 6, 4], [1, 6, 4]], [[5, 1, 3, 6], [1, 3, 6]],
            attrs={'hidden_size': 6}
        )

    def test_RNN_2(self):  # type: () -> None
        # 6->7 adapter is missing
        self._test_op_upgrade('RNN', 7, [[5, 3, 4], [2, 6, 4], [2, 6, 4]], [[5, 2, 3, 6], [2, 3, 6]],
            attrs={'hidden_size': 6, 'direction': 'bidirectional'}
        )

    def test_RNN_3(self):  # type: () -> None
        # 6->7 adapter is missing
        self._test_op_upgrade('RNN', 7,
            [[5, 3, 4], [1, 6, 4], [1, 6, 4], [1, 12], [5], [1, 5, 6]],
            [[5, 1, 3, 6], [1, 3, 6]],
            [TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.INT64, TensorProto.FLOAT],
            attrs={'hidden_size': 6}
        )

    def test_RoiAlign(self):  # type: () -> None
        self._test_op_upgrade('RoiAlign', 10, [[2, 3, 20, 20], [10, 4], [10]], [[10, 3, 1, 1]],
            [TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.INT64]
        )

    def test_Round(self):  # type: () -> None
        self._test_op_upgrade('Round', 11)

    def test_Scatter(self):  # type: () -> None
        self._test_op_upgrade('Scatter', 9, [[2, 3], [1, 2], [1, 2]], [[2, 3]],
            [TensorProto.FLOAT, TensorProto.INT64, TensorProto.FLOAT],
            [TensorProto.FLOAT]
        )

    def test_ScatterElements(self):  # type: () -> None
        self._test_op_upgrade('ScatterElements', 11, [[2, 3], [1, 2], [1, 2]], [[2, 3]],
            [TensorProto.FLOAT, TensorProto.INT64, TensorProto.FLOAT],
            [TensorProto.FLOAT]
        )

    def test_ScatterND(self):  # type: () -> None
        self._test_op_upgrade('ScatterND', 11, [[2, 3], [1, 2], [1, 2]], [[2, 3]],
            [TensorProto.FLOAT, TensorProto.INT64, TensorProto.FLOAT],
            [TensorProto.FLOAT]
        )

    def test_Scan(self):  # type: () -> None
        sum_in = onnx.helper.make_tensor_value_info('sum_in', onnx.TensorProto.FLOAT, [2])
        next_in = onnx.helper.make_tensor_value_info('next_in', onnx.TensorProto.FLOAT, [2])
        sum_out = onnx.helper.make_tensor_value_info('sum_out', onnx.TensorProto.FLOAT, [2])
        scan_out = onnx.helper.make_tensor_value_info('scan_out', onnx.TensorProto.FLOAT, [2])
        add_node = onnx.helper.make_node(
            'Add',
            inputs=['sum_in', 'next_in'],
            outputs=['sum_out']
        )
        id_node = onnx.helper.make_node(
            'Identity',
            inputs=['sum_out'],
            outputs=['scan_out']
        )
        body = onnx.helper.make_graph(
            [add_node, id_node],
            'scan_body',
            [sum_in, next_in],
            [sum_out, scan_out]
        )
        self._test_op_upgrade('Scan', 8, ['', [1, 2], [1, 3, 2]], [[1, 2], [1, 3, 2]],
            attrs={'body': body, 'num_scan_inputs': 1}
        )

    def test_Selu(self):  # type: () -> None
        self._test_op_upgrade('Selu', 1, attrs={'consumed_inputs': [0]})

    def test_Shape(self):  # type: () -> None
        self._test_op_upgrade('Shape', 1, [[3, 4, 5]], [[3]], output_types=[TensorProto.INT64])

    def test_Shrink(self):  # type: () -> None
        self._test_op_upgrade('Shrink', 9)

    def test_Sigmoid(self):  # type: () -> None
        self._test_op_upgrade('Sigmoid', 1, attrs={'consumed_inputs': [0]})

    def test_Sign(self):  # type: () -> None
        self._test_op_upgrade('Sign', 9)

    def test_Sinh(self):  # type: () -> None
        self._test_op_upgrade('Sinh', 9)

    def test_Sin(self):  # type: () -> None
        self._test_op_upgrade('Sin', 7)

    def test_Size(self):  # type: () -> None
        self._test_op_upgrade('Size', 1, [[3, 4, 5]], [[]], output_types=[TensorProto.INT64])

    def test_Slice(self):  # type: () -> None
        self._test_op_upgrade('Slice', 1, [[3, 4, 5]], [[3, 2, 2]],
            attrs={'axes': [1, 2], 'starts': [0, 1], 'ends': [2, 3]}
        )

    def test_Softmax_0(self):  # type: () -> None
        self._test_op_upgrade('Softmax', 1, attrs={'axis': 0})

    def test_Softmax_1(self):  # type: () -> None
        self._test_op_upgrade('Softmax', 1, attrs={'axis': 1})

    def test_Softmax_2(self):  # type: () -> None
        self._test_op_upgrade('Softmax', 1, attrs={'axis': 2})

    def test_Softmax_3(self):  # type: () -> None
        self._test_op_upgrade('Softmax', 1, attrs={'axis': -1})

    def test_Softmax_4(self):  # type: () -> None
        self._test_op_upgrade('Softmax', 1, attrs={'axis': -2})

    def test_Softmax_5(self):  # type: () -> None
        self._test_op_upgrade('Softmax', 1, attrs={'axis': -3})

    def test_Softplus(self):  # type: () -> None
        self._test_op_upgrade('Softplus', 1)

    def test_Softsign(self):  # type: () -> None
        self._test_op_upgrade('Softsign', 1)

    def test_SoftmaxCrossEntropyLoss(self):  # type: () -> None
        self._test_op_upgrade('SoftmaxCrossEntropyLoss', 12, [[3, 4, 5, 6], [3, 6]], [[]],
            [TensorProto.FLOAT, TensorProto.INT64]
        )

    def test_SpaceToDepth(self):  # type: () -> None
        self._test_op_upgrade('SpaceToDepth', 1, [[1, 3, 8, 8]], [[1, 12, 4, 4]],
            attrs={'blocksize': 2}
        )

    def test_Split(self):  # type: () -> None
        # 1->2 adapter is missing
        self._test_op_upgrade('Split', 2, [[3, 4, 7]], [[3, 4, 2], [3, 4, 1], [3, 4, 4]],
            attrs={'axis': 2, 'split': [2, 1, 4]}
        )

    def test_Sqrt(self):  # type: () -> None
        self._test_op_upgrade('Sqrt', 1, attrs={'consumed_inputs': [0]})

    def test_Squeeze(self):  # type: () -> None
        self._test_op_upgrade('Squeeze', 1, [[2, 1, 3, 4, 1]], [[2, 3, 4]])

    def test_StringNormalizer(self):  # type: () -> None
        self._test_op_upgrade('StringNormalizer', 10, [[1, 3]], [[1, 3]],
            [TensorProto.STRING], [TensorProto.STRING],
            attrs={'case_change_action': 'LOWER'}
        )

    def test_Sub(self):  # type: () -> None
        self._test_op_upgrade('Sub', 1, [[2, 3, 4], [2, 3, 4]], [[2, 3, 4]],
            attrs={'consumed_inputs': [0]}
        )

    def test_Sum(self):  # type: () -> None
        self._test_op_upgrade('Sum', 1, [[2, 3, 4], [2, 3, 4]], [[2, 3, 4]],
            attrs={'consumed_inputs': [0]}
        )

    def test_Tanh(self):  # type: () -> None
        self._test_op_upgrade('Tanh', 1, attrs={'consumed_inputs': [0]})

    def test_Tan(self):  # type: () -> None
        self._test_op_upgrade('Tan', 7)

    def test_TfIdfVectorizer(self):  # type: () -> None
        self._test_op_upgrade('TfIdfVectorizer', 9, [[3]], [[5]],
            attrs={'max_gram_length': 3, 'max_skip_count': 1, 'min_gram_length': 2, 'mode': 'TFIDF', 'ngram_counts': [0, 20], 'ngram_indexes': [3, 4]}
        )

    def test_ThresholdedRelu(self):  # type: () -> None
        self._test_op_upgrade('ThresholdedRelu', 10)

    def test_Tile(self):  # type: () -> None
        # 5->6 adapter is missing
        repeats = helper.make_tensor('b', TensorProto.INT64, dims=[3], vals=np.array([1, 2, 3]))
        self._test_op_upgrade('Tile', 6, [[3, 4, 5], [3]], [[3, 8, 15]],
            [TensorProto.FLOAT, TensorProto.INT64],
            initializer=[repeats]
        )

    def test_TopK(self):  # type: () -> None
        self._test_op_upgrade('TopK', 1, [[3, 4, 5]], [[3, 4, 2], [3, 4, 2]],
            output_types=[TensorProto.FLOAT, TensorProto.INT64],
            attrs={'k': 2}
        )

    def test_Transpose(self):  # type: () -> None
        self._test_op_upgrade('Transpose', 1, [[1, 2, 5, 3, 7]], [[1, 7, 5, 2, 3]],
            attrs={'perm': [0, 4, 2, 1, 3]}
        )

    def test_Trilu(self):  # type: () -> None
        self._test_op_upgrade('Trilu', 14)

    def test_Unique_1(self):  # type: () -> None
        self._test_op_upgrade('Unique', 11, [[3, 4, 5]], [[None]])

    def test_Unique_2(self):  # type: () -> None
        self._test_op_upgrade('Unique', 11, [[3, 4, 5]], [[3, None, 5]],
            attrs={'axis': 1}
        )

    def test_Unsqueeze(self):  # type: () -> None
        self._test_op_upgrade('Unsqueeze', 1, [[3, 4, 5]], [[3, 4, 1, 5]], attrs={'axes': [2]})

    def test_Upsample(self):  # type: () -> None
        self._test_op_upgrade('Upsample', 1, [[1, 3, 4, 5]], [[1, 3, 6, 10]],
            attrs={'width_scale': 2., 'height_scale': 1.5}
        )

    def test_Where(self):  # type: () -> None
        self._test_op_upgrade('Where', 9, [[2, 3], [2, 3], [2, 3]], [[2, 3]],
            [TensorProto.BOOL, TensorProto.FLOAT, TensorProto.FLOAT]
        )

    def test_Xor(self):  # type: () -> None
        # 6->7 adapter is missing
        self._test_op_upgrade('Xor', 7, [[2, 3], [2, 3]], [[2, 3]],
            [TensorProto.BOOL, TensorProto.BOOL], [TensorProto.BOOL]
        )

    def test_CastLike(self):  # type: () -> None
        self._test_op_upgrade('CastLike', 15,
            [[2, 3, 4], [2, 1, 4]],
            [[2, 3, 4]],
            input_types=[TensorProto.FLOAT, TensorProto.FLOAT16],
            output_types=[TensorProto.FLOAT16])

    def test_ops_tested(self):  # type: () -> None
        all_schemas = onnx.defs.get_all_schemas()
        all_op_names = [schema.name for schema in all_schemas if schema.domain == '']
        excluded_ops = [
            # Sequence-based and Optional-based ops disabled because
            # the version converter doesn't play nicely with sequences
            'ConcatFromSequence',
            'SequenceAt',
            'SequenceConstruct',
            'SequenceEmpty',
            'SequenceErase',
            'SequenceInsert',
            'SequenceLength',
            'SplitToSequence',
            'Optional',
            'OptionalGetElement',
            "OptionalHasElement"
        ]
        all_op_names = [op for op in all_op_names if op not in excluded_ops]

        untested_ops = set(all_op_names) - set(tested_ops)
        assert len(untested_ops) == 0


if __name__ == '__main__':
    unittest.main()
