import onnx
from onnx import helper, TensorProto, numpy_helper, shape_inference, version_converter
import string
import numpy as np
np.random.seed(0)

#####################################################################################
# This file is for generating test cases for models created in old opset versions, so
# we can test that the ONNX version converter is able to properly upgrade them to the
# opset version we currently support.
####################################################################################

ir_version = 6
target_opset = 14

test_performed = 0
tested_ops = []
failures = []

all_schemas = onnx.onnx_cpp2py_export.defs.get_all_schemas()
all_op_names = [schema.name for schema in all_schemas if schema.domain == '']

seq_ops = ['ConcatFromSequence', 'SequenceAt', 'SequenceConstruct', 'SequenceEmpty', 'SequenceErase', 'SequenceInsert', 'SequenceLength', 'SplitToSequence']
all_op_names = [op for op in all_op_names if op not in seq_ops]

def test_op_upgrade(op, from_opset, input_shapes=[[3, 4, 5]], output_shapes=[[3, 4, 5]], input_types=None, output_types=None, initializer=[], attrs={}, seq_inputs=[], seq_outputs=[], suffix=''):
    global test_performed
    global tested_ops
    test_performed = test_performed + 1
    tested_ops.append(op)

    if suffix != '':
        suffix = '_' + suffix
    test_name = op + suffix

    input_number = len(input_shapes)
    letters = list(string.ascii_lowercase)[:input_number]
    input_names = [
        letter if shape != '' else '' for (letter, shape) in zip(letters, input_shapes)
    ]
    if input_types is None:
        input_types = [TensorProto.FLOAT] * input_number
    is_sequence = [0 if id not in seq_inputs else 1 for id in range(input_number)]
    inputs = [
        helper.make_tensor_value_info(name, ttype, shape) if is_seq == 0 else helper.make_sequence_value_info(name, ttype, shape) for (name, ttype, shape, is_seq) 
            in zip(input_names, input_types, input_shapes, is_sequence) if name != ''
    ]

    output_number = len(output_shapes)
    output_names = list(string.ascii_lowercase)[input_number:input_number + output_number]
    if output_types is None:
        output_types = [TensorProto.FLOAT] * output_number
    is_sequence = [0 if id not in seq_outputs else 1 for id in range(output_number)]
    outputs = [
        helper.make_tensor_value_info(name, ttype, shape) if is_seq == 0 else helper.make_sequence_value_info(name, ttype, shape) for (name, ttype, shape, is_seq) 
            in zip(output_names, output_types, output_shapes, is_sequence)
    ]

    node = helper.make_node(op, input_names, output_names, **attrs)
    graph = helper.make_graph([node], test_name, inputs, outputs, initializer)
    original = helper.make_model(
        graph,
        producer_name='test', 
        ir_version=ir_version,
        opset_imports=[helper.make_opsetid('', from_opset)]
    )
    onnx.checker.check_model(original)
    shape_inference.infer_shapes(original)

    try:
        converted = version_converter.convert_version(original, target_opset)
        onnx.checker.check_model(converted)
        shape_inference.infer_shapes(converted)
    except Exception as e:
        print(test_name)
        print(e)
        print('**************************')
        failures.append(test_name)


def main():
    test_op_upgrade('Abs', 1, attrs={'consumed_inputs': [0]})
    test_op_upgrade('Acosh', 9)
    test_op_upgrade('Acos', 7)
    test_op_upgrade('And', 7, [[2, 3], [2, 3]], [[2, 3]], 
        [TensorProto.BOOL, TensorProto.BOOL], [TensorProto.BOOL]
    )
    test_op_upgrade('Asinh', 9)
    test_op_upgrade('Atanh', 9)
    test_op_upgrade('Add', 1, 
        [[3, 4, 5], [3, 4, 5]], 
        suffix='no-broadcast', 
        attrs={'consumed_inputs': [0]}
    )
    test_op_upgrade('Add', 1, [[3, 4, 5], [5]], 
        suffix='broadcast-1', 
        attrs={'consumed_inputs': [0], 'broadcast': 1}
    )
    test_op_upgrade('Add', 1, [[3, 4, 5], [3]], 
        suffix='broadcast-2', 
        attrs={'consumed_inputs': [0], 'broadcast': 1, 'axis': 0}
    )
    test_op_upgrade('ArgMax', 7, [[2, 3, 4]], [[1, 3, 4]], 
        output_types=[TensorProto.INT64], 
        suffix='axis-0'
    )
    test_op_upgrade('ArgMax', 7, [[2, 3, 4]], [[2, 1, 4]], 
        output_types=[TensorProto.INT64], 
        attrs={'axis':1}, 
        suffix='axis-1'
    )
    test_op_upgrade('ArgMin', 7, [[2, 3, 4]], [[1, 3, 4]], 
        output_types=[TensorProto.INT64], 
        suffix='axis-0'
    )
    test_op_upgrade('ArgMin', 7, [[2, 3, 4]], [[2, 1, 4]], 
        output_types=[TensorProto.INT64], 
        attrs={'axis':1}, 
        suffix='axis-1'
    )
    test_op_upgrade('Asin', 7)
    test_op_upgrade('Atan', 7)
    test_op_upgrade('AveragePool', 1, [[1, 1, 5, 5]], [[1, 1, 4, 4]], 
        attrs={'kernel_shape': [2, 2]}
    )
    test_op_upgrade('BitShift', 11, [[2, 3], [2, 3]], [[2, 3]], 
        [TensorProto.UINT8, TensorProto.UINT8], [TensorProto.UINT8], 
        attrs={'direction': 'RIGHT'}
    )
    test_op_upgrade('BatchNormalization', 1, [[1, 3], [3], [3], [3], [3]], [[1, 3]], 
        suffix='1-output', 
        attrs={'consumed_inputs': [1, 1], 'is_test': 1, 'spatial': 1}
    )
    test_op_upgrade('BatchNormalization', 1, 
        [[1, 3], [3], [3], [3], [3]], [[1, 3], [3], [3], [3], [3]], 
        suffix='all-outputs', 
        attrs={'consumed_inputs': [1, 1], 'is_test': 1, 'spatial': 1}
    )
    test_op_upgrade('Cast', 6, [[2, 3]], [[2, 3]], [TensorProto.INT64], attrs={'to': 1})
    test_op_upgrade('Ceil', 1, attrs={'consumed_inputs': [0]})
    test_op_upgrade('Celu', 12)
    test_op_upgrade('Clip', 1, suffix='default', attrs={'consumed_inputs': [0]})
    test_op_upgrade('Clip', 1, suffix='min', attrs={'consumed_inputs': [0], 'min': -1.4})
    test_op_upgrade('Clip', 1, suffix='max', attrs={'consumed_inputs': [0], 'max': 2.6})
    test_op_upgrade('Clip', 1, 
        suffix='minmax', 
        attrs={'consumed_inputs': [0], 'min': -1.4, 'max': 2.6}
    )
    test_op_upgrade('Compress', 9, [[6, 7], [3]], [[3]], 
        [TensorProto.FLOAT, TensorProto.BOOL], [TensorProto.FLOAT]
    )
    test_op_upgrade('Concat', 1, [[2, 3], [2, 4]], [[2, 7]])
    test_op_upgrade('Constant', 1, [],
        attrs={
            'value': helper.make_tensor(
                'Value', 
                TensorProto.FLOAT, 
                dims=[3, 4, 5], 
                vals=np.random.rand(3, 4, 5).astype(np.float32).tobytes(), raw=True
            )
        }
    )
    test_op_upgrade('ConstantOfShape', 9, [[3]])
    test_op_upgrade('Conv', 1, [[1, 3, 5, 5], [4, 3, 2, 2], [4]], [[1, 4, 4, 4]])
    test_op_upgrade('Conv', 1, [[1, 3, 5, 5], [4, 3, 2, 2], [4]], [[1, 4, 4, 4]], 
        suffix='bias'
    )
    test_op_upgrade('Conv', 1, [[1, 3, 5, 5], [4, 1, 2, 2], [4]], [[1, 4, 3, 7]], 
        attrs={'dilations': [1, 2], 'group': 3, 'pads': [0, 1, 2, 3], 'strides': [2, 1]}, 
        suffix='misc-attributes'
    )
    test_op_upgrade('ConvInteger', 10, [[1, 3, 5, 5], [4, 3, 2, 2], [4]], [[1, 4, 4, 4]], 
        [TensorProto.UINT8, TensorProto.UINT8, TensorProto.UINT8], [TensorProto.INT32]
    )
    test_op_upgrade('ConvTranspose', 1, [[1, 1, 5, 5], [1, 1, 3, 3]], [[1, 1, 7, 7]])
    test_op_upgrade('Cosh', 9)
    test_op_upgrade('Cos', 7)
    test_op_upgrade('CumSum', 11, [[3, 4, 5], []], [[3, 4, 5]], 
        [TensorProto.FLOAT, TensorProto.INT64]
    )
    test_op_upgrade('DepthToSpace', 1, [[1, 8, 3, 3]], [[1, 2, 6, 6]], 
        attrs={'blocksize': 2}
    )
    test_op_upgrade('DequantizeLinear', 10, [[2, 3], [], []], [[2, 3]], 
        [TensorProto.INT8, TensorProto.FLOAT, TensorProto.INT8]
    )
    test_op_upgrade('Det', 11, [[3, 5, 5]], [[3]], suffix='batch')
    test_op_upgrade('Det', 11, [[5, 5]], [[]], suffix='single')
    test_op_upgrade('DynamicQuantizeLinear', 11, [[3, 4, 5]], [[3, 4, 5], [], []],
        output_types=[TensorProto.UINT8, TensorProto.FLOAT, TensorProto.UINT8]
    )
    test_op_upgrade('Div', 1, [[3, 4, 5], [3, 1, 5]], attrs={'consumed_inputs': [0]})
    test_op_upgrade('Dropout', 1, attrs={'consumed_inputs': [0], 'is_test': 1})
    test_op_upgrade('Einsum', 12, [[3, 4, 5], [3, 5, 6]], [[3, 4, 6]], 
        attrs={'equation': 'bij, bjk -> bik'}, 
        suffix='batch-matmul'
    )
    test_op_upgrade('Einsum', 12, [[4, 5]], [[5, 4]], 
        attrs={'equation': 'ij->ji'}, 
        suffix='transpose'
    )
    test_op_upgrade('Elu', 1, attrs={'consumed_inputs': [0]})
    test_op_upgrade('Equal', 7, [[2, 3], [2, 3]], [[2, 3]], output_types=[TensorProto.BOOL])
    test_op_upgrade('Erf', 9)
    test_op_upgrade('Exp', 1, attrs={'consumed_inputs': [0]})
    expand_shape = helper.make_tensor(
        'b', 
        TensorProto.INT64, 
        dims=[4], 
        vals=np.array([5, 2, 6, 4])
    )
    test_op_upgrade('Expand', 8, [[2, 1, 4], [4]], [[5, 2, 6, 4]], 
        [TensorProto.FLOAT, TensorProto.INT64], 
        initializer=[expand_shape]
    )
    test_op_upgrade('EyeLike', 9, [[4, 5]], [[4, 5]])
    test_op_upgrade('Flatten', 1, [[3, 4, 5]], [[3, 20]], attrs={'axis': 1})
    test_op_upgrade('Floor', 1, attrs={'consumed_inputs': [0]})
    test_op_upgrade('Gather', 1, [[3, 4, 5], [6, 7]], [[6, 7, 4, 5]], 
        [TensorProto.FLOAT, TensorProto.INT64]
    )
    test_op_upgrade('GatherElements', 11, [[3, 4, 5], [6, 7]], [[6, 7]], 
        [TensorProto.FLOAT, TensorProto.INT64]
    )
    test_op_upgrade('GatherND', 11, [[1, 2, 3], [1, 2, 3]], [[1, 2]])
    test_op_upgrade('Gemm', 1, [[5, 4], [4, 3], [3]], [[5, 3]])
    test_op_upgrade('GlobalAveragePool', 1, [[1, 3, 10, 10]], [[1, 3, 1, 1]])
    test_op_upgrade('GlobalMaxPool', 1, [[1, 3, 10, 10]], [[1, 3, 1, 1]])
    test_op_upgrade('GlobalLpPool', 2, [[1, 3, 10, 10]], [[1, 3, 1, 1]])
    test_op_upgrade('GRU', 7, 
        [[5, 3, 4], [1, 18, 4], [1, 18, 4]], [[5, 1, 3, 6], [1, 3, 6]], 
        attrs={'hidden_size': 6}
    )
    test_op_upgrade('GRU', 7, 
        [[5, 3, 4], [2, 18, 4], [2, 18, 4]], [[5, 2, 3, 6], [2, 3, 6]], 
        attrs={'hidden_size': 6, 'direction': 'bidirectional'}, 
        suffix='bidirectional'
    )
    test_op_upgrade('GRU', 7, 
        [[5, 3, 4], [1, 18, 4], [1, 18, 4], [1, 24], [5], [1, 5, 6]], 
        [[5, 1, 3, 6], [1, 3, 6]], 
        [TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.INT64, TensorProto.FLOAT], 
        attrs={'hidden_size': 6}, 
        suffix='all-inputs'
    )
    test_op_upgrade('Greater', 7, [[2, 3], [2, 3]], [[2, 3]], 
        output_types=[TensorProto.BOOL]
    )
    test_op_upgrade('GreaterOrEqual', 12, [[2, 3], [2, 3]], [[2, 3]], 
        output_types=[TensorProto.BOOL]
    )
    test_op_upgrade('HardSigmoid', 1, attrs={'consumed_inputs': [0]})
    test_op_upgrade('Hardmax', 1)
    test_op_upgrade('Identity', 1)
    if_sub_output = [helper.make_tensor_value_info('out', TensorProto.FLOAT, [3, 4, 5])]
    if_then_tensor = helper.make_tensor(
        'Value', 
        TensorProto.FLOAT, 
        dims=[3, 4, 5], 
        vals=np.random.rand(3, 4, 5).astype(np.float32).tobytes(), raw=True
    )
    if_then_node = helper.make_node('Constant', [], ['out'], value=if_then_tensor)
    if_then_graph = helper.make_graph([if_then_node], 'then_graph', [], if_sub_output, [])
    if_else_tensor = helper.make_tensor(
        'Value', 
        TensorProto.FLOAT, 
        dims=[3, 4, 5], 
        vals=np.random.rand(3, 4, 5).astype(np.float32).tobytes(), raw=True
    )
    if_else_node = helper.make_node('Constant', [], ['out'], value=if_else_tensor)
    if_else_graph = helper.make_graph([if_else_node], 'else_graph', [], if_sub_output, [])
    test_op_upgrade('If', 1, [[0]], [[3, 4, 5]], [TensorProto.BOOL], 
        attrs={'then_branch': if_then_graph, 'else_branch': if_else_graph}
    )    
    test_op_upgrade('InstanceNormalization', 1, [[1, 3], [3], [3]], [[1, 3]], 
        attrs={'consumed_inputs': [0]}
    )
    test_op_upgrade('IsInf', 10, [[2, 3]], [[2, 3]], output_types=[TensorProto.BOOL])
    test_op_upgrade('IsNaN', 9, [[2, 3]], [[2, 3]], output_types=[TensorProto.BOOL])
    test_op_upgrade('LeakyRelu', 1, attrs={'consumed_inputs': [0]})
    test_op_upgrade('Less', 7, [[2, 3], [2, 3]], [[2, 3]], output_types=[TensorProto.BOOL])
    test_op_upgrade('LessOrEqual', 12, [[2, 3], [2, 3]], [[2, 3]], 
        output_types=[TensorProto.BOOL]
    )
    test_op_upgrade('Log', 1, attrs={'consumed_inputs': [0]})
    test_op_upgrade('LogSoftmax', 1)
    y_in = onnx.helper.make_tensor_value_info('y_in', onnx.TensorProto.FLOAT, [1])
    y_out = onnx.helper.make_tensor_value_info('y_out', onnx.TensorProto.FLOAT, [1])
    scan_out = onnx.helper.make_tensor_value_info('scan_out', onnx.TensorProto.FLOAT, [1])
    cond_in = onnx.helper.make_tensor_value_info('cond_in', onnx.TensorProto.BOOL, [])
    cond_out = onnx.helper.make_tensor_value_info('cond_out', onnx.TensorProto.BOOL, [])
    iter_count = onnx.helper.make_tensor_value_info('iter_count', onnx.TensorProto.INT64, [])
    x_const_node = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=['x'],
        value=onnx.helper.make_tensor(
            name='const_tensor_x',
            data_type=onnx.TensorProto.FLOAT,
            dims=[5],
            vals=np.array([1, 2, 3, 4, 5]).astype(np.float32),
        )
    )
    one_const_node = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=['one'],
        value=onnx.helper.make_tensor(
            name='const_tensor_one',
            data_type=onnx.TensorProto.INT64,
            dims=(),
            vals=[1]
        )
    )
    i_add_node = onnx.helper.make_node(
        'Add',
        inputs=['iter_count', 'one'],
        outputs=['end']
    )
    start_unsqueeze_node = onnx.helper.make_node(
        'Unsqueeze',
        inputs=['iter_count'],
        outputs=['slice_start'],
        axes=[0]
    )
    end_unsqueeze_node = onnx.helper.make_node(
        'Unsqueeze',
        inputs=['end'],
        outputs=['slice_end'],
        axes=[0]
    )
    slice_node = onnx.helper.make_node(
        'Slice',
        inputs=['x', 'slice_start', 'slice_end'],
        outputs=['slice_out']
    )
    y_add_node = onnx.helper.make_node(
        'Add',
        inputs=['y_in', 'slice_out'],
        outputs=['y_out']
    )
    identity_node = onnx.helper.make_node(
        'Identity',
        inputs=['cond_in'],
        outputs=['cond_out']
    )
    scan_identity_node = onnx.helper.make_node(
        'Identity',
        inputs=['y_out'],
        outputs=['scan_out']
    )
    loop_body = onnx.helper.make_graph(
        [identity_node, x_const_node, one_const_node, i_add_node,
         start_unsqueeze_node, end_unsqueeze_node, slice_node, y_add_node,
         scan_identity_node],
        'loop_body',
        [iter_count, cond_in, y_in],
        [cond_out, y_out, scan_out]
    )
    test_op_upgrade('Loop', 11, [[], [], [1]], [[1], [5, 1]], 
        [TensorProto.INT64, TensorProto.BOOL, TensorProto.FLOAT], 
        attrs={'body': loop_body}
    )
    test_op_upgrade('LpNormalization', 1)
    test_op_upgrade('LpPool', 2, [[1, 1, 5, 5]], [[1, 1, 4, 4]], 
        attrs={'kernel_shape': [2, 2]}
    )
    test_op_upgrade('LRN', 1, suffix='2-dim', attrs={'size': 3})
    test_op_upgrade('LRN', 1, [[2, 3, 4, 5]], [[2, 3, 4, 5]], 
        suffix='3-dim', 
        attrs={'size': 3}
    )
    test_op_upgrade('LSTM', 7, 
        [[5, 3, 4], [1, 24, 4], [1, 24, 4]], 
        [[5, 1, 3, 6], [1, 3, 6], [1, 3, 6]], 
        attrs={'hidden_size': 6}
    )
    test_op_upgrade('LSTM', 7, 
        [[5, 3, 4], [2, 24, 4], [2, 24, 4]], 
        [[5, 2, 3, 6], [2, 3, 6], [2, 3, 6]], 
        attrs={'hidden_size': 6, 'direction': 'bidirectional'}, 
        suffix='bidirectional'
    )
    test_op_upgrade('LSTM', 7, 
        [[5, 3, 4], [1, 24, 4], [1, 24, 4], [1, 48], [5], [1, 5, 6], [1, 5, 6], [1, 18]], 
        [[5, 1, 3, 6], [1, 3, 6], [1, 3, 6]], 
        [TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.INT64, TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.FLOAT], 
        attrs={'hidden_size': 6}, 
        suffix='all-inputs'
    )
    test_op_upgrade('MatMul', 1, [[2, 3], [3, 4]], [[2, 4]], suffix='2-dim')
    test_op_upgrade('MatMul', 1, [[5, 2, 3], [5, 3, 4]], [[5, 2, 4]], suffix='3-dim')
    test_op_upgrade('MatMulInteger', 10, [[2, 3], [3, 4]], [[2, 4]], 
        [TensorProto.INT8, TensorProto.INT8], [TensorProto.INT32], 
        suffix='2-inputs'
    )
    test_op_upgrade('MatMulInteger', 10, [[2, 3], [3, 4], [], []], [[2, 4]], 
        [TensorProto.INT8, TensorProto.INT8, TensorProto.INT8, TensorProto.INT8], 
        [TensorProto.INT32], 
        suffix='4-inputs-scalar'
    )
    test_op_upgrade('MatMulInteger', 10, [[2, 3], [3, 4], [2], [4]], [[2, 4]], 
        [TensorProto.INT8, TensorProto.INT8, TensorProto.INT8, TensorProto.INT8], 
        [TensorProto.INT32], 
        suffix='4-inputs-vector'
    )
    test_op_upgrade('Max', 1, [[2, 3, 4], [2, 3, 4]], [[2, 3, 4]], 
        attrs={'consumed_inputs': [0]}
    )
    test_op_upgrade('MaxPool', 1, [[1, 1, 5, 5]], [[1, 1, 4, 4]], 
        suffix='1-output', 
        attrs={'kernel_shape': [2, 2]}
    )
    test_op_upgrade('MaxPool', 8, [[1, 1, 5, 5]], [[1, 1, 4, 4], [1, 1, 4, 4]], 
        output_types=[TensorProto.FLOAT, TensorProto.INT64], 
        suffix='2-outputs', 
        attrs={'kernel_shape': [2, 2]}
    )
    test_op_upgrade('MaxRoiPool', 1, [[2, 3, 20, 20], [4, 5]], [[4, 3, 3, 3]], 
        attrs={'pooled_shape': [3, 3]}
    )
    test_op_upgrade('MaxUnpool', 9, [[1, 1, 5, 5], [1, 1, 5, 5]], [[1, 1, 6, 6]], 
        [TensorProto.FLOAT, TensorProto.INT64], 
        attrs={'kernel_shape': [2, 2]}
    )
    test_op_upgrade('Mean', 1, [[2, 3, 4], [2, 3, 4]], [[2, 3, 4]], 
        attrs={'consumed_inputs': [0]}
    )
    test_op_upgrade('MeanVarianceNormalization', 9, attrs={'axes': [1, 2]})
    test_op_upgrade('Min', 1, [[2, 3, 4], [2, 3, 4]], [[2, 3, 4]], 
        attrs={'consumed_inputs': [0]}
    )
    test_op_upgrade('Mod', 10, [[2, 3], [2, 3]], [[2, 3]])
    test_op_upgrade('Mod', 10, [[2, 3], [2, 3]], [[2, 3]], attrs={'fmod':1}, suffix='fmod')
    test_op_upgrade('Mul', 1, [[2, 3, 4], [2, 1, 4]], [[2, 3, 4]], 
        attrs={'consumed_inputs': [0]}
    )
    test_op_upgrade('Multinomial', 7, [[3, 5]], [[3, 7]], 
        output_types=[TensorProto.INT32], 
        attrs={'sample_size': 7}
    )
    test_op_upgrade('Neg', 1, attrs={'consumed_inputs': [0]})
    # test_op_upgrade('NegativeLogLikelihoodLoss', 12, [[3, 4, 5], [3, 5]], [[]], 
    #     [TensorProto.FLOAT, TensorProto.INT64], 
    #     suffix='2-inputs'
    # )
    # test_op_upgrade('NegativeLogLikelihoodLoss', 12, [[3, 4, 5], [3, 5], [4]], [[]], 
    #     [TensorProto.FLOAT, TensorProto.INT64, TensorProto.FLOAT], 
    #     suffix='3-inputs'
    # )
    test_op_upgrade('NonMaxSuppression', 10, [[2, 3, 4], [3, 5, 6]], [[2, 3]], 
        output_types=[TensorProto.INT64]
    )
    test_op_upgrade('NonZero', 9, [[3, 3]], [[2, 4]], output_types=[TensorProto.INT64])
    test_op_upgrade('Not', 1, [[2, 3]], [[2, 3]], [TensorProto.BOOL], [TensorProto.BOOL])
    test_op_upgrade('OneHot', 9, [[3, 4, 5], [], [2]], [[3, 4, 5, 6]])
    test_op_upgrade('Or', 7, [[2, 3], [2, 3]], [[2, 3]], 
        [TensorProto.BOOL, TensorProto.BOOL], [TensorProto.BOOL]
    )
    test_op_upgrade('Pad', 2, [[3, 4]], [[5, 8]], 
        attrs={'pads': [1, 2, 1, 2], 'value': 1.5}
    )
    test_op_upgrade('Pow', 1, [[2, 3, 4], [2, 3, 4]], [[2, 3, 4]])
    test_op_upgrade('PRelu', 1, [[2, 3, 4], [2, 3, 4]], [[2, 3, 4]], 
        attrs={'consumed_inputs': [0]}
    )
    test_op_upgrade('QLinearConv', 10, 
        [[1, 3, 5, 5], [], [], [4, 3, 2, 2], [], [], [], []], [[1, 4, 4, 4]]
    )
    test_op_upgrade('QLinearMatMul', 10, [[2, 3], [], [], [3, 4], [], [], [], []], [[2, 4]])
    test_op_upgrade('QuantizeLinear', 10, [[3, 4, 5], [], []], [[3, 4, 5]], 
        [TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.UINT8], [TensorProto.UINT8]
    )
    test_op_upgrade('RandomNormal', 1, [], [[3, 4, 5]], attrs={'shape': [3, 4, 5]})
    randomnormallike_like = helper.make_tensor(
        'a', 
        TensorProto.FLOAT, 
        dims=[3, 4, 5], 
        vals=np.random.rand(3, 4, 5).astype(np.float32).tobytes(), raw=True
    )    
    test_op_upgrade('RandomUniformLike', 1, [[3, 4, 5]], [[3, 4, 5]], 
        initializer=[randomnormallike_like]
    )
    test_op_upgrade('RandomUniform', 1, [], [[3, 4, 5]], attrs={'shape': [3, 4, 5]})
    randomuniformlike_like = helper.make_tensor(
        'a', 
        TensorProto.FLOAT, 
        dims=[3, 4, 5], 
        vals=np.random.rand(3, 4, 5).astype(np.float32).tobytes(), raw=True
    )    
    test_op_upgrade('RandomNormalLike', 1, [[3, 4, 5]], [[3, 4, 5]], 
        initializer=[randomuniformlike_like]
    )
    range_start = helper.make_tensor('a', TensorProto.FLOAT, dims=[], vals=np.array([0]))
    range_end = helper.make_tensor('b', TensorProto.FLOAT, dims=[], vals=np.array([12]))
    range_step = helper.make_tensor('c', TensorProto.FLOAT, dims=[], vals=np.array([2]))
    test_op_upgrade('Range', 11, [[], [], []], [[6]], 
        initializer=[range_start, range_end, range_step]
    )
    test_op_upgrade('Reciprocal', 1, attrs={'consumed_inputs': [0]})
    test_op_upgrade('ReduceL1', 1, [[3, 4, 5]], [[1, 1, 1]])
    test_op_upgrade('ReduceL2', 1, [[3, 4, 5]], [[1, 1, 1]])
    test_op_upgrade('ReduceLogSum', 1, [[3, 4, 5]], [[1, 1, 1]])
    test_op_upgrade('ReduceLogSumExp', 1, [[3, 4, 5]], [[1, 1, 1]])
    test_op_upgrade('ReduceMean', 1, [[3, 4, 5]], [[1, 1, 1]])
    test_op_upgrade('ReduceMax', 1, [[3, 4, 5]], [[1, 1, 1]])
    test_op_upgrade('ReduceMin', 1, [[3, 4, 5]], [[1, 1, 1]])
    test_op_upgrade('ReduceProd', 1, [[3, 4, 5]], [[1, 1, 1]])
    test_op_upgrade('ReduceSum', 1, [[3, 4, 5]], [[1, 1, 1]])
    test_op_upgrade('ReduceSumSquare', 1, [[3, 4, 5]], [[1, 1, 1]])
    test_op_upgrade('Relu', 1, attrs={'consumed_inputs': [0]})
    test_op_upgrade('Reshape', 1, [[3, 4, 5]], [[3, 10, 2]], 
        attrs={'consumed_inputs': [0], 'shape': [3, 10, 2]}
    )
    test_op_upgrade('Resize', 10, [[3, 4, 5], [3]], [[3, 8, 15]])
    test_op_upgrade('ReverseSequence', 10, [[3, 4, 5], [4]], [[3, 4, 5]], 
        [TensorProto.FLOAT, TensorProto.INT64]
    )
    test_op_upgrade('RNN', 7, [[5, 3, 4], [1, 6, 4], [1, 6, 4]], [[5, 1, 3, 6], [1, 3, 6]], 
        attrs={'hidden_size': 6}
    )
    test_op_upgrade('RNN', 7, [[5, 3, 4], [2, 6, 4], [2, 6, 4]], [[5, 2, 3, 6], [2, 3, 6]], 
        attrs={'hidden_size': 6, 'direction': 'bidirectional'}, 
        suffix='bidirectional'
    )
    test_op_upgrade('RNN', 7, 
        [[5, 3, 4], [1, 6, 4], [1, 6, 4], [1, 12], [5], [1, 5, 6]], 
        [[5, 1, 3, 6], [1, 3, 6]], 
        [TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.INT64, TensorProto.FLOAT], 
        attrs={'hidden_size': 6}, 
        suffix='all-inputs'
    )
    test_op_upgrade('RoiAlign', 10, [[2, 3, 20, 20], [10, 4], [10]], [[10, 3, 1, 1]], 
        [TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.INT64]
    )
    test_op_upgrade('Round', 11)
    test_op_upgrade('Scatter', 9, [[2, 3], [1, 2], [1, 2]], [[2, 3]], 
        [TensorProto.FLOAT, TensorProto.INT64, TensorProto.FLOAT], 
        [TensorProto.FLOAT]
    )
    test_op_upgrade('ScatterElements', 11, [[2, 3], [1, 2], [1, 2]], [[2, 3]], 
        [TensorProto.FLOAT, TensorProto.INT64, TensorProto.FLOAT], 
        [TensorProto.FLOAT]
    )
    test_op_upgrade('ScatterND', 11, [[2, 3], [1, 2], [1, 2]], [[2, 3]], 
        [TensorProto.FLOAT, TensorProto.INT64, TensorProto.FLOAT], 
        [TensorProto.FLOAT]
    )
    scan_sum_in = onnx.helper.make_tensor_value_info('sum_in', onnx.TensorProto.FLOAT, [2])
    scan_next_in = onnx.helper.make_tensor_value_info('next_in', onnx.TensorProto.FLOAT, [2])
    scan_sum_out = onnx.helper.make_tensor_value_info('sum_out', onnx.TensorProto.FLOAT, [2])
    scan_scan_out = onnx.helper.make_tensor_value_info('scan_out', onnx.TensorProto.FLOAT, [2])
    scan_add_node = onnx.helper.make_node(
        'Add',
        inputs=['sum_in', 'next_in'],
        outputs=['sum_out']
    )
    scan_id_node = onnx.helper.make_node(
        'Identity',
        inputs=['sum_out'],
        outputs=['scan_out']
    )
    scan_body = onnx.helper.make_graph(
        [scan_add_node, scan_id_node],
        'scan_body',
        [scan_sum_in, scan_next_in],
        [scan_sum_out, scan_scan_out]
    )
    test_op_upgrade('Scan', 8, ['', [1, 2], [1, 3, 2]], [[1, 2], [1, 3, 2]], 
        attrs={'body': scan_body, 'num_scan_inputs':1}
    )
    test_op_upgrade('Selu', 1, attrs={'consumed_inputs': [0]})
    test_op_upgrade('Shape', 1, [[3, 4, 5]], [[3]], output_types=[TensorProto.INT64])
    test_op_upgrade('Shrink', 9)
    test_op_upgrade('Sigmoid', 1, attrs={'consumed_inputs': [0]})
    test_op_upgrade('Sign', 9)
    test_op_upgrade('Sinh', 9)
    test_op_upgrade('Sin', 7)
    test_op_upgrade('Size', 1, [[3, 4, 5]], [[]], output_types=[TensorProto.INT64])
    test_op_upgrade('Slice', 1, [[3, 4, 5]], [[3, 2, 2]], 
        attrs={'axes': [1, 2], 'starts': [0, 1], 'ends': [2, 3]}
    )
    test_op_upgrade('Softmax', 1)
    test_op_upgrade('Softplus', 1)
    test_op_upgrade('Softsign', 1)
    test_op_upgrade('SoftmaxCrossEntropyLoss', 12, [[3, 4, 5, 6], [3, 6]], [[]], 
        [TensorProto.FLOAT, TensorProto.INT64]
    )
    test_op_upgrade('SpaceToDepth', 1, [[1, 3, 8, 8]], [[1, 12, 4, 4]], 
        attrs={'blocksize': 2}
    )
    test_op_upgrade('Split', 2, [[3, 4, 7]], [[3, 4, 2], [3, 4, 1], [3, 4, 4]], 
        attrs={'axis': 2, 'split': [2, 1, 4]}
    )
    test_op_upgrade('Sqrt', 1, attrs={'consumed_inputs': [0]})
    test_op_upgrade('Squeeze', 1, [[2, 1, 3, 4, 1]], [[2, 3, 4]])
    test_op_upgrade('StringNormalizer', 10, [[1, 3]], [[1, 3]], 
        [TensorProto.STRING], [TensorProto.STRING], 
        attrs={'case_change_action': 'LOWER'}
    )
    test_op_upgrade('Sub', 1, [[2, 3, 4], [2, 3, 4]], [[2, 3, 4]], 
        attrs={'consumed_inputs': [0]}
    )
    test_op_upgrade('Sum', 1, [[2, 3, 4], [2, 3, 4]], [[2, 3, 4]], 
        attrs={'consumed_inputs': [0]}
    )
    test_op_upgrade('Tanh', 1, attrs={'consumed_inputs': [0]})
    test_op_upgrade('Tan', 7)
    test_op_upgrade('TfIdfVectorizer', 9, [[3]], [[5]], 
        attrs={'max_gram_length': 3, 'max_skip_count': 1, 'min_gram_length': 2, 'mode': 'TFIDF', 'ngram_counts': [0, 20], 'ngram_indexes': [3, 4]}
    )
    test_op_upgrade('ThresholdedRelu', 10)
    tile_repeats = helper.make_tensor('b', TensorProto.INT64, dims=[3], vals=np.array([1, 2, 3]))
    test_op_upgrade('Tile', 6, [[3, 4, 5], [3]], [[3, 8, 15]], 
        [TensorProto.FLOAT, TensorProto.INT64], 
        initializer=[tile_repeats]
    )
    test_op_upgrade('TopK', 1, [[3, 4, 5]], [[3, 4, 2], [3, 4, 2]], 
        output_types=[TensorProto.FLOAT, TensorProto.INT64], 
        attrs={'k': 2}
    )
    test_op_upgrade('Transpose', 1, [[1, 2, 5, 3, 7]], [[1, 7, 5, 2, 3]], 
        attrs={'perm': [0, 4, 2, 1, 3]}
    )
    test_op_upgrade('Unique', 11, [[3, 4, 5]], [[None]], suffix='flat')
    test_op_upgrade('Unique', 11, [[3, 4, 5]], [[3, None, 5]], 
        attrs={'axis': 1}, 
        suffix='axis'
    )
    test_op_upgrade('Unsqueeze', 1, [[3, 4, 5]], [[3, 4, 1, 5]], attrs={'axes': [2]})
    test_op_upgrade('Upsample', 7, [[3, 4, 5]], [[6, 6, 10]], 
        attrs={'scales': [2., 1.5, 2.]}
    )
    test_op_upgrade('Where', 9, [[2, 3], [2, 3], [2, 3]], [[2, 3]], 
        [TensorProto.BOOL, TensorProto.FLOAT, TensorProto.FLOAT]
    )
    test_op_upgrade('Xor', 7, [[2, 3], [2, 3]], [[2, 3]], 
        [TensorProto.BOOL, TensorProto.BOOL], [TensorProto.BOOL]
    )


    if len(failures) == 0:
        print('All tests passed')
    else:
        print('')
        print( str(len(failures)) + '/' + str(test_performed) + ' tests failed:' )
        for failure in failures:
            print(failure)

    untested_ops = set(all_op_names) - set(tested_ops)

    if len(untested_ops) != 0:
        print('')
        print( str(len(untested_ops)) + ' ops are untested:' )
        for untested in sorted(untested_ops):
            print(untested)

if __name__ == '__main__':
    main()