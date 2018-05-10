from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from onnx import checker, helper, TensorProto
from onnx.helper import make_node, make_tensor_value_info

import onnx.shape_inference
import unittest

import numpy as np


class TestShapeInference(unittest.TestCase):
    def _make_graph(self, seed_values, nodes, value_info):
        input_value_infos = []
        for name, dtype, shape in seed_values:
            # introduce the starting values as the output of reshape,
            # so that the sizes are guaranteed to be unknown
            value_info.append(make_tensor_value_info(name, dtype, shape))
            input_value_infos.append(make_tensor_value_info('SEED_' + name, TensorProto.UNDEFINED, ()))
            input_value_infos.append(make_tensor_value_info('UNKNOWN_SHAPE_' + name, TensorProto.UNDEFINED, ()))
            nodes[:0] = [make_node("Reshape", ['SEED_' + name, 'UNKNOWN_SHAPE_' + name], [name])]
        return helper.make_graph(nodes, "test", input_value_infos, [], value_info=value_info)

    def _inferred(self, graph):
        orig_model = helper.make_model(graph, producer_name='onnx-test')
        inferred_model = onnx.shape_inference.infer_shapes(orig_model)
        checker.check_model(inferred_model)
        return inferred_model

    def _assert_inferred(self, graph, vis):
        names_in_vis = set(x.name for x in vis)
        vis = list(x for x in graph.value_info if x.name not in names_in_vis) + vis
        inferred_model = self._inferred(graph)
        inferred_vis = list(inferred_model.graph.value_info)

        vis = list(sorted(vis, key=lambda x: x.name))
        inferred_vis = list(sorted(inferred_vis, key=lambda x: x.name))
        if vis == inferred_vis:
            return
        # otherwise some custom logic to give a nicer diff
        vis_names = set(x.name for x in vis)
        inferred_vis_names = set(x.name for x in inferred_vis)
        assert vis_names == inferred_vis_names, (vis_names, inferred_vis_names)
        for vi, inferred_vi in zip(vis, inferred_vis):
            assert vi == inferred_vi, '\n%s\n%s\n' % (vi, inferred_vi)
        assert False

    def _identity_prop(self, op, **kwargs):
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (30, 4, 5))],
            [make_node(op, 'x', 'y', **kwargs)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (30, 4, 5))])

    def test_transpose(self):
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (2, 3, 4))],
            [make_node("Transpose", ["X"], ["Y"], perm=[1, 0, 2])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (3, 2, 4))])

    def test_transpose_preexisting(self):
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (2, 3, 4))],
            [make_node("Transpose", ["X"], ["Y"], perm=[1, 0, 2])],
            [make_tensor_value_info("Y", TensorProto.FLOAT, None)])
        self._assert_inferred(graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (3, 2, 4))])

    def test_transpose_partial(self):
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (2, 3, 4))],
            [make_node("Transpose", ["X"], ["Y"], perm=[1, 0, 2])],
            [make_tensor_value_info("Y", TensorProto.UNDEFINED, (3, "a", "b"))])
        self._assert_inferred(graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (3, 2, 4))])

    def test_transpose_preexisting_incorrect_shape(self):
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (2, 3, 4))],
            [make_node("Transpose", ["X"], ["Y"], perm=[1, 0, 2])],
            [make_tensor_value_info("Y", TensorProto.FLOAT, (5, 5, 5))])
        self.assertRaises(RuntimeError, self._inferred, graph)

    def test_transpose_preexisting_incorrect_type(self):
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (2, 3, 4))],
            [make_node("Transpose", ["X"], ["Y"], perm=[1, 0, 2])],
            [make_tensor_value_info("Y", TensorProto.STRING, (3, 2, 4))])
        self.assertRaises(RuntimeError, self._inferred, graph)

    def _make_matmul_test_all_dims_known(self, shape1, shape2):
        expected_out_shape = np.matmul(np.arange(np.product(shape1)).reshape(shape1),
                                       np.arange(np.product(shape2)).reshape(shape2)).shape
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, shape1),
             ('y', TensorProto.FLOAT, shape2)],
            [make_node('MatMul', ['x', 'y'], ['z'])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, expected_out_shape)])

    def test_matmul_all_dims_known(self):
        self._make_matmul_test_all_dims_known((2,), (2,))

        self._make_matmul_test_all_dims_known((4, 2), (2, 4))
        self._make_matmul_test_all_dims_known((5, 2), (2, 4))
        self._make_matmul_test_all_dims_known((5, 2), (2, 1))
        self._make_matmul_test_all_dims_known((1, 2), (2, 3))
        self._make_matmul_test_all_dims_known((2,), (2, 3))
        self._make_matmul_test_all_dims_known((4, 2), (2,))
        self._make_matmul_test_all_dims_known((1, 4, 2), (3, 2, 3))
        self._make_matmul_test_all_dims_known((3, 4, 2), (3, 2, 3))
        self._make_matmul_test_all_dims_known((5, 1, 4, 2), (1, 3, 2, 3))
        self._make_matmul_test_all_dims_known((4, 2), (3, 2, 3))

    def _make_matmul_test_allow_unknown(self, shape1, shape2, expected_out_shape):
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, shape1),
             ('y', TensorProto.FLOAT, shape2)],
            [make_node('MatMul', ['x', 'y'], ['z'])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, expected_out_shape)])

    def test_matmul_allow_unknown(self):
        self._make_matmul_test_allow_unknown((None,), (None,), ())
        self._make_matmul_test_allow_unknown((3,), (None,), ())
        self._make_matmul_test_allow_unknown((2,), (2, "a"), ("a",))
        self._make_matmul_test_allow_unknown((4, 2), (2, "a"), (4, "a"))
        self._make_matmul_test_allow_unknown((4, None), (2, "a"), (4, "a"))
        self._make_matmul_test_allow_unknown((4, None), (None, "a"), (4, "a"))
        self._make_matmul_test_allow_unknown((1, 4, 2), ("a", 2, 5), ("a", 4, 5))
        self._make_matmul_test_allow_unknown((1, 3, 4, 2), ("a", 2, 5), (1, 3, 4, 5))

    def test_cast(self):
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (2, 4, 3))],
            [make_node("Cast", ["x"], ["y"], to=TensorProto.UINT8)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info("y", TensorProto.UINT8, (2, 4, 3))])

    def test_concat(self):
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (2, 4, 3)),
             ("y", TensorProto.FLOAT, (7, 4, 3))],
            [make_node("Concat", ['x', 'y'], ['z'], axis=0)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, (9, 4, 3))])

    def test_concat_3d_axis_2(self):
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (2, 2, 2)),
             ('y', TensorProto.FLOAT, (2, 2, 2))],
            [make_node('Concat', ['x', 'y'], ['z'], axis=2)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, (2, 2, 4))])

    def test_reshape(self):
        graph = self._make_graph(
            [('x', TensorProto.UINT8, (2, 4, 3)),
             ('shape', TensorProto.UNDEFINED, (2,))],
            [make_node("Reshape", ['x', 'shape'], ['y'])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.UINT8, None)])

    def test_shape(self):
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (2, 4, 3))],
            [make_node("Shape", ['x'], ['y'])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.INT64, (3,))])

    def test_size(self):
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (2, 4, 3))],
            [make_node("Size", ['x'], ['y'])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.INT64, ())])

    def test_gather(self):
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (3, 2)),
             ('i', TensorProto.INT64, (2,))],
            [make_node("Gather", ['x', 'i'], ['y'])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (None, None))])

    def test_gather_into_scalar(self):
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (3,)),
             ('i', TensorProto.INT64, ())],
            [make_node("Gather", ['x', 'i'], ['y'])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, ())])

    def test_squeeze(self):
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (1, 3, 1, 1, 2, 1))],
            [make_node('Squeeze', 'x', 'y', axes=[0, 2, 3, 5])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (3, 2))])

    def test_unsqueeze(self):
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (3, 2))],
            [make_node('Unsqueeze', 'x', 'y', axes=[0, 1, 3, 5])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (1, 1, 3, 1, 2, 1))])

    def test_slice(self):
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (3, 2))],
            [make_node('Slice', 'x', 'y', axes=[0, 1], starts=[1, 0], ends=[2, 2])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (1, 2))])

    def test_slice_unsorted_axes(self):
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (3, 2))],
            [make_node('Slice', 'x', 'y', axes=[1, 0], starts=[1, 0], ends=[2, 2])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, None)])

    def test_slice_giant_number(self):
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (3, 2))],
            [make_node('Slice', 'x', 'y', axes=[0, 1], starts=[1, 0], ends=[200, 22000])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (2, 2))])

    def test_slice_negative_end(self):
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (3, 2))],
            [make_node('Slice', 'x', 'y', axes=[0, 1], starts=[1, 0], ends=[200, -1])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (2, None))])

    def test_slice_negative_start(self):
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (3, 2))],
            [make_node('Slice', 'x', 'y', axes=[0, 1], starts=[1, -2], ends=[200, 3])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (2, None))])

    def test_slice_variable_copy(self):
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, ("a", 2))],
            [make_node('Slice', 'x', 'y', axes=[1], starts=[1], ends=[200])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, ("a", 1))])

    def test_pad(self):
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (1, None, 2))],
            [make_node('Pad', 'x', 'y', pads=[1, 3, 1, 1, 0, 1])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (3, None, 4))])

    def test_constant_pad_2d(self):
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (2, 3, 4, 4))],
            [make_node('Pad', 'x', 'y', pads=[0, 0, 3, 1, 0, 0, 4, 2], mode="constant", value=2.0)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (2, 3, 11, 7))])

    def test_conv(self):
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (3, 4, 5, 6, 7)),
             ('y', TensorProto.FLOAT, (5, 4, 2, 4, 3))],
            [make_node('Conv', ['x', 'y'], 'z', pads=[0, 1, 1, 0, 0, 1], dilations=[1, 2, 2], strides=[1, 1, 2])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, (3, 5, 4, 1, 3))])

    def test_conv_1d_simple(self):
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (30, 4, 5)),
             ('y', TensorProto.FLOAT, (50, 4, 2))],
            [make_node('Conv', ['x', 'y'], 'z', dilations=[1])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, (30, 50, 4))])

    def test_conv_dilations(self):
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (30, 4, 8, 8, 8)),
             ('y', TensorProto.FLOAT, (50, 4, 3, 3, 3))],
            [make_node('Conv', ['x', 'y'], 'z', dilations=[1, 2, 3])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, (30, 50, 6, 4, 2))])

    def test_conv_strides(self):
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (30, 4, 8, 8, 8)),
             ('y', TensorProto.FLOAT, (50, 4, 3, 3, 3))],
            [make_node('Conv', ['x', 'y'], 'z', strides=[1, 2, 3])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, (30, 50, 6, 3, 2))])

    def test_conv_pads(self):
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (30, 4, 7, 6, 4)),
             ('y', TensorProto.FLOAT, (50, 4, 3, 3, 3))],
            [make_node('Conv', ['x', 'y'], 'z', pads=[1, 1, 2, 0, 1, 2])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, (30, 50, 6, 6, 6))])

    def test_conv_only_one_pos(self):
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (30, 4, 5)),
             ('y', TensorProto.FLOAT, (50, 4, 5))],
            [make_node('Conv', ['x', 'y'], 'z', strides=[2])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, (30, 50, 1))])

    def test_conv_partial_missing_shape(self):
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (30, 4, None, 6, 4)),
             ('y', TensorProto.FLOAT, (50, 4, 3, 3, 3))],
            [make_node('Conv', ['x', 'y'], 'z', pads=[1, 1, 2, 0, 1, 2])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, (30, 50, None, 6, 6))])

    def test_conv_partial_missing_weight_shape(self):
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (30, 4, 7, 6, 4)),
             ('y', TensorProto.FLOAT, (50, 4, None, 3, 3))],
            [make_node('Conv', ['x', 'y'], 'z', pads=[1, 1, 2, 0, 1, 2])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, None)])

    def test_relu(self):
        self._identity_prop('Relu')

    def test_add(self):
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (30, 4, 5)),
             ('y', TensorProto.FLOAT, (30, 4, 5))],
            [make_node('Add', ['x', 'y'], 'z')],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, (30, 4, 5))])

    def test_sum_single(self):
        self._identity_prop('Sum')

    def test_sum_multi(self):
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (30, 4, 5)),
             ('y', TensorProto.FLOAT, (30, 4, 5)),
             ('z', TensorProto.FLOAT, (30, 4, 5))],
            [make_node('Sum', ['x', 'y', 'z'], ['out'])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('out', TensorProto.FLOAT, (30, 4, 5))])

    def test_random_normal(self):
        graph = self._make_graph(
            [],
            [make_node('RandomNormal', [], ['out'], dtype=TensorProto.DOUBLE, shape=(3, 4, 5))],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('out', TensorProto.DOUBLE, (3, 4, 5))])

    def test_random_normal_like(self):
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (2, 3, 4))],
            [make_node('RandomNormalLike', ['X'], ['out'])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('out', TensorProto.FLOAT, (2, 3, 4))])

    def test_random_normal_like_with_dtype(self):
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (2, 3, 4))],
            [make_node('RandomNormalLike', ['X'], ['out'], dtype=TensorProto.DOUBLE,)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('out', TensorProto.DOUBLE, (2, 3, 4))])

    def test_constant_fill(self):
        graph = self._make_graph(
            [],
            [make_node('ConstantFill', [], ['out'], dtype=TensorProto.INT32, shape=(3, 4, 5))],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('out', TensorProto.INT32, (3, 4, 5))])

    def test_constant_fill_with_input(self):
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (2, 3, 4))],
            [make_node('ConstantFill', ['X'], ['out'], dtype=TensorProto.INT32)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('out', TensorProto.INT32, (2, 3, 4))])

    def test_constant_fill_with_extra_shape(self):
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (2, 3, 4))],
            [make_node('ConstantFill', ['X'], ['out'], dtype=TensorProto.INT32, extra_shape=(5, 6))],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('out', TensorProto.INT32, (2, 3, 4, 5, 6))])

    def _logical_binary_op(self, op, input_type):
        graph = self._make_graph(
            [('x', input_type, (30, 4, 5)),
             ('y', input_type, (30, 4, 5))],
            [make_node(op, ['x', 'y'], 'z')],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.BOOL, (30, 4, 5))])

    def test_logical_and(self):
        self._logical_binary_op('And', TensorProto.BOOL)

    def test_logical_or(self):
        self._logical_binary_op('Or', TensorProto.BOOL)

    def test_logical_xor(self):
        self._logical_binary_op('Xor', TensorProto.BOOL)

    def test_greater(self):
        self._logical_binary_op('Greater', TensorProto.FLOAT)

    def test_less(self):
        self._logical_binary_op('Less', TensorProto.FLOAT)

    def test_equal(self):
        self._logical_binary_op('Equal', TensorProto.FLOAT)

    def test_logical_not(self):
        graph = self._make_graph(
            [('x', TensorProto.BOOL, (30, 4, 5))],
            [make_node('Not', ['x'], 'z')],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.BOOL, (30, 4, 5))])

    def test_gemm(self):
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (7, 5)),
             ('y', TensorProto.FLOAT, (5, 11)),
             ('z', TensorProto.FLOAT, None)],
            [make_node('Gemm', ['x', 'y', 'z'], ['out'])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('out', TensorProto.FLOAT, (7, 11))])

    def test_gemm_transA(self):
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (5, 7)),
             ('y', TensorProto.FLOAT, (5, 11)),
             ('z', TensorProto.FLOAT, None)],
            [make_node('Gemm', ['x', 'y', 'z'], ['out'], transA=1)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('out', TensorProto.FLOAT, (7, 11))])

    def test_gemm_transB(self):
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (7, 5)),
             ('y', TensorProto.FLOAT, (11, 5)),
             ('z', TensorProto.FLOAT, None)],
            [make_node('Gemm', ['x', 'y', 'z'], ['out'], transB=1)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('out', TensorProto.FLOAT, (7, 11))])

    def test_gemm_transA_and_transB(self):
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (5, 7)),
             ('y', TensorProto.FLOAT, (11, 5)),
             ('z', TensorProto.FLOAT, None)],
            [make_node('Gemm', ['x', 'y', 'z'], ['out'], transA=1, transB=1)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('out', TensorProto.FLOAT, (7, 11))])

    def test_gemm_shape_from_C(self):
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, None),
             ('y', TensorProto.FLOAT, (11, 5)),
             ('z', TensorProto.FLOAT, (7, None))],
            [make_node('Gemm', ['x', 'y', 'z'], ['out'], transA=1, transB=1)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('out', TensorProto.FLOAT, (7, None))])

    def test_gemm_shape_from_C_broadcast_false(self):
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, None),
             ('y', TensorProto.FLOAT, (11, 5)),
             ('z', TensorProto.FLOAT, (7, None))],
            [make_node('Gemm', ['x', 'y', 'z'], ['out'], transA=1, transB=1, broadcast=0)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('out', TensorProto.FLOAT, (7, None))])

    def test_dropout(self):
        self._identity_prop('Dropout')

    def test_LRN(self):
        self._identity_prop('LRN', alpha=0.5, beta=0.5, size=1)

    def test_batch_norm(self):
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (3, 4, 5, 6, 7)),
             ('scale', TensorProto.FLOAT, (4,)),
             ('b', TensorProto.FLOAT, (4,)),
             ('mean', TensorProto.FLOAT, (4,)),
             ('var', TensorProto.FLOAT, (4,))],
            [make_node('BatchNormalization', ['x', 'scale', 'b', 'mean', 'var'], ['out'], is_test=1)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('out', TensorProto.FLOAT, (3, 4, 5, 6, 7))])

    def test_batch_norm_train(self):
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (3, 4, 5, 6, 7)),
             ('scale', TensorProto.FLOAT, (4,)),
             ('b', TensorProto.FLOAT, (4,)),
             ('mean', TensorProto.FLOAT, (4,)),
             ('var', TensorProto.FLOAT, (4,))],
            [make_node('BatchNormalization', ['x', 'scale', 'b', 'mean', 'var'], ['out'], is_test=0)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('out', TensorProto.FLOAT, (3, 4, 5, 6, 7))])

    def test_softmax(self):
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (4, 5))],
            [make_node('Softmax', ['x'], 'z')],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, (4, 5))])

    def test_lp_norm(self):
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (3, 4, 5, 6, 7))],
            [make_node('LpNormalization', ['x'], ['out'])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('out', TensorProto.FLOAT, (3, 4, 5, 6, 7))])

    def test_instance_norm(self):
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (3, 4, 5, 6, 7)),
             ('scale', TensorProto.FLOAT, (4,)),
             ('b', TensorProto.FLOAT, (4,))],
            [make_node('InstanceNormalization', ['x', 'scale', 'b'], ['out'])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('out', TensorProto.FLOAT, (3, 4, 5, 6, 7))])


if __name__ == '__main__':
    unittest.main()
