from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from onnx import checker, helper, TensorProto, NodeProto, GraphProto, ValueInfoProto, ModelProto
from onnx.helper import make_node, make_tensor, make_tensor_value_info, make_empty_tensor_value_info
from typing import Sequence, Union, Text, Tuple, List, Any, Optional
import onnx.shape_inference
import unittest

import numpy as np  # type: ignore


class TestShapeInference(unittest.TestCase):
    def _make_graph(self,
                    seed_values,  # type: Sequence[Union[Text, Tuple[Text, TensorProto.DataType, Any]]]
                    nodes,  # type: List[NodeProto]
                    value_info,  # type: List[ValueInfoProto]
                    initializer=None  # type: Optional[Sequence[TensorProto]]
                    ):  # type: (...) -> GraphProto
        if initializer is None:
            initializer = []
        names_in_initializer = set(x.name for x in initializer)
        input_value_infos = []
        # If the starting values are not also initializers,
        # introduce the starting values as the output of reshape,
        # so that the sizes are guaranteed to be unknown
        for seed_value in seed_values:
            if isinstance(seed_value, tuple):
                seed_name = seed_value[0]
                seed_value_info = make_tensor_value_info(*seed_value)
            else:
                seed_name = seed_value
                seed_value_info = make_empty_tensor_value_info(seed_value)

            if seed_name in names_in_initializer:
                input_value_infos.append(seed_value_info)
            else:
                value_info.append(seed_value_info)
                input_value_infos.append(make_tensor_value_info('SEED_' + seed_name, TensorProto.UNDEFINED, ()))
                input_value_infos.append(make_tensor_value_info('UNKNOWN_SHAPE_' + seed_name, TensorProto.UNDEFINED, ()))
                nodes[:0] = [make_node("Reshape", ['SEED_' + seed_name, 'UNKNOWN_SHAPE_' + seed_name], [seed_name])]
        return helper.make_graph(nodes, "test", input_value_infos, [], initializer=initializer, value_info=value_info)

    def _inferred(self, graph, **kwargs):  # type: (GraphProto, **Any) -> ModelProto
        kwargs[str('producer_name')] = 'onnx-test'
        orig_model = helper.make_model(graph, **kwargs)
        inferred_model = onnx.shape_inference.infer_shapes(orig_model)
        checker.check_model(inferred_model)
        return inferred_model

    def _assert_inferred(self, graph, vis, **kwargs):  # type: (GraphProto, List[ValueInfoProto], **Any) -> None
        names_in_vis = set(x.name for x in vis)
        vis = list(x for x in graph.value_info if x.name not in names_in_vis) + vis
        inferred_model = self._inferred(graph, **kwargs)
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

    def test_empty_graph(self):  # type: () -> None
        graph = self._make_graph(
            ['y'],
            [], [])
        self._assert_inferred(graph, [])

    def _identity_prop(self, op, **kwargs):  # type: (Text, **Any) -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (30, 4, 5))],
            [make_node(op, 'x', 'y', **kwargs)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (30, 4, 5))])

    def test_transpose(self):  # type: () -> None
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (2, 3, 4))],
            [make_node("Transpose", ["X"], ["Y"], perm=[1, 0, 2])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (3, 2, 4))])

    def test_transpose_preexisting(self):  # type: () -> None
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (2, 3, 4))],
            [make_node("Transpose", ["X"], ["Y"], perm=[1, 0, 2])],
            [make_tensor_value_info("Y", TensorProto.FLOAT, None)])
        self._assert_inferred(graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (3, 2, 4))])

    def test_transpose_partial(self):  # type: () -> None
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (2, 3, 4))],
            [make_node("Transpose", ["X"], ["Y"], perm=[1, 0, 2])],
            [make_tensor_value_info("Y", TensorProto.UNDEFINED, (3, "a", "b"))])  # type: ignore
        self._assert_inferred(graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (3, 2, 4))])

    def test_transpose_preexisting_incorrect_shape(self):  # type: () -> None
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (2, 3, 4))],
            [make_node("Transpose", ["X"], ["Y"], perm=[1, 0, 2])],
            [make_tensor_value_info("Y", TensorProto.FLOAT, (5, 5, 5))])
        self.assertRaises(RuntimeError, self._inferred, graph)

    def test_transpose_preexisting_incorrect_type(self):  # type: () -> None
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (2, 3, 4))],
            [make_node("Transpose", ["X"], ["Y"], perm=[1, 0, 2])],
            [make_tensor_value_info("Y", TensorProto.STRING, (3, 2, 4))])
        self.assertRaises(RuntimeError, self._inferred, graph)

    def _make_matmul_test_all_dims_known(self, shape1, shape2):  # type: (Sequence[int], Sequence[int]) -> None
        expected_out_shape = np.matmul(np.arange(np.product(shape1)).reshape(shape1),
                                       np.arange(np.product(shape2)).reshape(shape2)).shape
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, shape1),
             ('y', TensorProto.FLOAT, shape2)],
            [make_node('MatMul', ['x', 'y'], ['z'])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, expected_out_shape)])

    def test_matmul_all_dims_known(self):  # type: () -> None
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

    def _make_matmul_test_allow_unknown(self, shape1, shape2, expected_out_shape):  # type: (Any, Any, Any) -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, shape1),
             ('y', TensorProto.FLOAT, shape2)],
            [make_node('MatMul', ['x', 'y'], ['z'])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, expected_out_shape)])

    def test_matmul_allow_unknown(self):  # type: () -> None
        self._make_matmul_test_allow_unknown((None,), (None,), ())
        self._make_matmul_test_allow_unknown((3,), (None,), ())
        self._make_matmul_test_allow_unknown((2,), (2, "a"), ("a",))
        self._make_matmul_test_allow_unknown((4, 2), (2, "a"), (4, "a"))
        self._make_matmul_test_allow_unknown((4, None), (2, "a"), (4, "a"))
        self._make_matmul_test_allow_unknown((4, None), (None, "a"), (4, "a"))
        self._make_matmul_test_allow_unknown((1, 4, 2), ("a", 2, 5), ("a", 4, 5))
        self._make_matmul_test_allow_unknown((1, 3, 4, 2), ("a", 2, 5), (1, 3, 4, 5))

    def test_cast(self):  # type: () -> None
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (2, 4, 3))],
            [make_node("Cast", ["x"], ["y"], to=TensorProto.UINT8)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info("y", TensorProto.UINT8, (2, 4, 3))])

    def test_concat(self):  # type: () -> None
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (2, 4, 3)),
             ("y", TensorProto.FLOAT, (7, 4, 3))],
            [make_node("Concat", ['x', 'y'], ['z'], axis=0)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, (9, 4, 3))])

    def test_concat_missing_shape(self):  # type: () -> None
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (2, 4, 3)),
             "y",
             ("z", TensorProto.FLOAT, (None, None, None))],
            [make_node("Concat", ['x', 'y', 'z'], ['out'], axis=0)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('out', TensorProto.FLOAT, None)])

    def test_concat_3d_axis_2(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (2, 2, 2)),
             ('y', TensorProto.FLOAT, (2, 2, 2))],
            [make_node('Concat', ['x', 'y'], ['z'], axis=2)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, (2, 2, 4))])

    def test_concat_param(self):  # type: () -> None
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, ("a", 2)),
             ("y", TensorProto.FLOAT, ("a", 3))],
            [make_node("Concat", ['x', 'y'], ['z'], axis=1)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, ("a", 5))])

    def test_reshape_dynamic_shape(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.UINT8, (2, 4, 3)),
             ('shape', TensorProto.UNDEFINED, (2,))],
            [make_node("Reshape", ['x', 'shape'], ['y'])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.UINT8, None)])

    def test_reshape_static_shape(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.UINT8, (2, 4, 3)),
             ('shape', TensorProto.INT64, (2,))],
            [make_node("Reshape", ['x', 'shape'], ['y'])],
            [],
            initializer=[make_tensor('shape', TensorProto.INT64, (2,), (3, 8))])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.UINT8, (3, 8))])

    def test_reshape_static_shape_inferred(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.UINT8, (2, 4, 3)),
             ('shape', TensorProto.INT64, (3,))],
            [make_node("Reshape", ['x', 'shape'], ['y'])],
            [],
            initializer=[make_tensor('shape', TensorProto.INT64, (3,), (0, 3, -1))])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.UINT8, (2, 3, 4))])

    def test_reshape_static_shape_constant(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.UINT8, (2, 4, 3))],
            [make_node("Constant", [], ['shape'],
                       value=make_tensor('shape', TensorProto.INT64, (2,), (3, 8))),
             make_node("Reshape", ['x', 'shape'], ['y'])],
            [])
        self._assert_inferred(graph, [
            make_tensor_value_info('shape', TensorProto.INT64, (2,)),
            make_tensor_value_info('y', TensorProto.UINT8, (3, 8))])

    def test_upsample(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.INT32, (2, 4, 3, 5)),
             ('scales', TensorProto.FLOAT, (4,))],
            [make_node("Upsample", ['x', 'scales'], ['y'])],
            [],
            initializer=[make_tensor('scales', TensorProto.FLOAT, (4,), (1.0, 1.1, 1.3, 1.9))])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.INT32, (2, 4, 3, 9))])

    def test_shape(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (2, 4, 3))],
            [make_node("Shape", ['x'], ['y'])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.INT64, (3,))])

    def test_size(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (2, 4, 3))],
            [make_node("Size", ['x'], ['y'])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.INT64, ())])

    def test_gather(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (4, 3)),
             ('i', TensorProto.INT64, (2,))],
            [make_node("Gather", ['x', 'i'], ['y'])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (2, 3))])  # type: ignore

    def test_gather_axis1(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (4, 3, 5)),
             ('i', TensorProto.INT64, (1, 2))],
            [make_node("Gather", ['x', 'i'], ['y'], axis=1)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (4, 1, 2, 5))])  # type: ignore

    def test_gather_into_scalar(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (3,)),
             ('i', TensorProto.INT64, ())],
            [make_node("Gather", ['x', 'i'], ['y'])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, ())])

    def test_scatter(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (3, 3)),
             ('i', TensorProto.INT64, (2, 3)),
             ('u', TensorProto.FLOAT, (2, 3))],
            [make_node("Scatter", ['x', 'i', 'u'], ['y'])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (3, 3))])  # type: ignore

    def test_scatter_axis1(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (1, 5)),
             ('i', TensorProto.INT64, (1, 2)),
             ('u', TensorProto.FLOAT, (1, 2))],
            [make_node("Scatter", ['x', 'i', 'u'], ['y'], axis=1)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (1, 5))])  # type: ignore

    def test_squeeze(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (1, 3, 1, 1, 2, 1))],
            [make_node('Squeeze', 'x', 'y', axes=[0, 2, 3, 5])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (3, 2))])

    def test_unsqueeze(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (3, 2))],
            [make_node('Unsqueeze', 'x', 'y', axes=[0, 1, 3, 5])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (1, 1, 3, 1, 2, 1))])

    def test_slice(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (3, 2))],
            [make_node('Slice', 'x', 'y', axes=[0, 1], starts=[1, 0], ends=[2, 2])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (1, 2))])

    def test_slice_unsorted_axes(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (3, 2))],
            [make_node('Slice', 'x', 'y', axes=[1, 0], starts=[1, 0], ends=[2, 2])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, None)])

    def test_slice_giant_number(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (3, 2))],
            [make_node('Slice', 'x', 'y', axes=[0, 1], starts=[1, 0], ends=[200, 22000])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (2, 2))])

    def test_slice_negative_end(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (3, 2))],
            [make_node('Slice', 'x', 'y', axes=[0, 1], starts=[1, 0], ends=[200, -1])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (2, None))])  # type: ignore

    def test_slice_negative_start(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (3, 2))],
            [make_node('Slice', 'x', 'y', axes=[0, 1], starts=[1, -2], ends=[200, 3])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (2, None))])  # type: ignore

    def test_slice_variable_copy(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, ("a", 2))],
            [make_node('Slice', 'x', 'y', axes=[1], starts=[1], ends=[200])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, ("a", 1))])  # type: ignore

    def test_pad(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (1, None, 2))],
            [make_node('Pad', 'x', 'y', pads=[1, 3, 1, 1, 0, 1])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (3, None, 4))])  # type: ignore

    def test_constant_pad_2d(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (2, 3, 4, 4))],
            [make_node('Pad', 'x', 'y', pads=[0, 0, 3, 1, 0, 0, 4, 2], mode="constant", value=2.0)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (2, 3, 11, 7))])

    def test_conv(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (3, 4, 5, 6, 7)),
             ('y', TensorProto.FLOAT, (5, 4, 2, 4, 3))],
            [make_node('Conv', ['x', 'y'], 'z', pads=[0, 1, 1, 0, 0, 1], dilations=[1, 2, 2], strides=[1, 1, 2])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, (3, 5, 4, 1, 3))])

    def test_conv_1d_simple(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (30, 4, 5)),
             ('y', TensorProto.FLOAT, (50, 4, 2))],
            [make_node('Conv', ['x', 'y'], 'z', dilations=[1])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, (30, 50, 4))])

    def test_conv_dilations(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (30, 4, 8, 8, 8)),
             ('y', TensorProto.FLOAT, (50, 4, 3, 3, 3))],
            [make_node('Conv', ['x', 'y'], 'z', dilations=[1, 2, 3])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, (30, 50, 6, 4, 2))])

    def test_conv_strides(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (30, 4, 8, 8, 8)),
             ('y', TensorProto.FLOAT, (50, 4, 3, 3, 3))],
            [make_node('Conv', ['x', 'y'], 'z', strides=[1, 2, 3])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, (30, 50, 6, 3, 2))])

    def test_conv_pads(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (30, 4, 7, 6, 4)),
             ('y', TensorProto.FLOAT, (50, 4, 3, 3, 3))],
            [make_node('Conv', ['x', 'y'], 'z', pads=[1, 1, 2, 0, 1, 2])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, (30, 50, 6, 6, 6))])

    def test_conv_only_one_pos(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (30, 4, 5)),
             ('y', TensorProto.FLOAT, (50, 4, 5))],
            [make_node('Conv', ['x', 'y'], 'z', strides=[2])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, (30, 50, 1))])

    def test_conv_partial_missing_shape(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (30, 4, None, 6, 4)),
             ('y', TensorProto.FLOAT, (50, 4, 3, 3, 3))],
            [make_node('Conv', ['x', 'y'], 'z', pads=[1, 1, 2, 0, 1, 2])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, (30, 50, None, 6, 6))])  # type: ignore

    def test_conv_partial_missing_weight_shape(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (30, 4, 7, 6, 4)),
             ('y', TensorProto.FLOAT, (50, 4, None, 3, 3))],
            [make_node('Conv', ['x', 'y'], 'z', pads=[1, 1, 2, 0, 1, 2])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, None)])

    def test_relu(self):  # type: () -> None
        self._identity_prop('Relu')

    def test_add(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (30, 4, 5)),
             ('y', TensorProto.FLOAT, (30, 4, 5))],
            [make_node('Add', ['x', 'y'], 'z')],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, (30, 4, 5))])

    def test_pow(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (30, 4, 5)),
             ('y', TensorProto.FLOAT, (30, 4, 5))],
            [make_node('Pow', ['x', 'y'], 'z')],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, (30, 4, 5))])

    def test_sum_single(self):  # type: () -> None
        self._identity_prop('Sum')

    def test_sum_multi(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (30, 4, 5)),
             ('y', TensorProto.FLOAT, (30, 4, 5)),
             ('z', TensorProto.FLOAT, (30, 4, 5))],
            [make_node('Sum', ['x', 'y', 'z'], ['out'])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('out', TensorProto.FLOAT, (30, 4, 5))])

    def test_sum_multi_broadcasting(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (30, 1, 5)),
             ('y', TensorProto.FLOAT, ("a", 4, 1)),
             ('z', TensorProto.FLOAT, (4, "b"))],
            [make_node('Sum', ['x', 'y', 'z'], ['out'])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('out', TensorProto.FLOAT, (30, 4, 5))])

    def test_sum_broadcasting_param(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, ("a", 1, 5)),
             ('y', TensorProto.FLOAT, ("a", 4, 1))],
            [make_node('Sum', ['x', 'y'], ['out'])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('out', TensorProto.FLOAT, ("a", 4, 5))])

    def test_random_normal(self):  # type: () -> None
        graph = self._make_graph(
            [],
            [make_node('RandomNormal', [], ['out'], dtype=TensorProto.DOUBLE, shape=(3, 4, 5))],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('out', TensorProto.DOUBLE, (3, 4, 5))])

    def test_random_normal_like(self):  # type: () -> None
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (2, 3, 4))],
            [make_node('RandomNormalLike', ['X'], ['out'])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('out', TensorProto.FLOAT, (2, 3, 4))])

    def test_random_normal_like_with_dtype(self):  # type: () -> None
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (2, 3, 4))],
            [make_node('RandomNormalLike', ['X'], ['out'], dtype=TensorProto.DOUBLE,)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('out', TensorProto.DOUBLE, (2, 3, 4))])

    def test_constant_fill(self):  # type: () -> None
        graph = self._make_graph(
            [],
            [make_node('ConstantFill', [], ['out'], dtype=TensorProto.INT32, shape=(3, 4, 5))],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('out', TensorProto.INT32, (3, 4, 5))])

    def test_constant_fill_with_input(self):  # type: () -> None
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (2, 3, 4))],
            [make_node('ConstantFill', ['X'], ['out'], dtype=TensorProto.INT32)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('out', TensorProto.INT32, (2, 3, 4))])

    def test_constant_fill_with_extra_shape(self):  # type: () -> None
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (2, 3, 4))],
            [make_node('ConstantFill', ['X'], ['out'], dtype=TensorProto.INT32, extra_shape=(5, 6))],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('out', TensorProto.INT32, (2, 3, 4, 5, 6))])

    def _logical_binary_op(self, op, input_type):  # type: (Text, TensorProto.DataType) -> None
        graph = self._make_graph(
            [('x', input_type, (30, 4, 5)),
             ('y', input_type, (30, 4, 5))],
            [make_node(op, ['x', 'y'], 'z')],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.BOOL, (30, 4, 5))])

    def test_logical_and(self):  # type: () -> None
        self._logical_binary_op('And', TensorProto.BOOL)

    def test_logical_or(self):  # type: () -> None
        self._logical_binary_op('Or', TensorProto.BOOL)

    def test_logical_xor(self):  # type: () -> None
        self._logical_binary_op('Xor', TensorProto.BOOL)

    def test_greater(self):  # type: () -> None
        self._logical_binary_op('Greater', TensorProto.FLOAT)

    def test_less(self):  # type: () -> None
        self._logical_binary_op('Less', TensorProto.FLOAT)

    def test_equal(self):  # type: () -> None
        self._logical_binary_op('Equal', TensorProto.FLOAT)

    def test_logical_not(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.BOOL, (30, 4, 5))],
            [make_node('Not', ['x'], 'z')],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.BOOL, (30, 4, 5))])

    def test_flatten(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (2, 3, 4, 5))],
            [make_node('Flatten', ['x'], ['z'], axis=2)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, (6, 20))])

    def test_flatten_default_axis(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (2, 3, 4, 5))],
            [make_node('Flatten', ['x'], ['z'])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, (2, 60))])

    def test_flatten_zero_axis(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (2, 3, 4, 5))],
            [make_node('Flatten', ['x'], ['z'], axis=0)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, (1, 120))])

    def test_flatten_unknown_dim(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (2, 'N', 4, 5))],
            [make_node('Flatten', ['x'], ['z'], axis=2)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, (None, 20))])  # type: ignore

    def test_space_to_depth(self):  # type: () -> None
        b = 10
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (2, 3, 100, 100))],
            [make_node('SpaceToDepth', ['x'], ['z'], blocksize=b)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, (2, 300, 10, 10))])

    def test_space_to_depth_unknown_dim(self):  # type: () -> None
        b = 10
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (2, 'N', 100, 100))],
            [make_node('SpaceToDepth', ['x'], ['z'], blocksize=b)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, (2, None, 10, 10))])  # type: ignore

    def test_depth_to_space(self):  # type: () -> None
        b = 10
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (2, 300, 10, 10))],
            [make_node('DepthToSpace', ['x'], ['z'], blocksize=b)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, (2, 3, 100, 100))])

    def _rnn_forward(self, seqlen, batchsize, inpsize, hiddensize):  # type: (int, int, int, int) -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (seqlen, batchsize, inpsize)),
             ('w', TensorProto.FLOAT, (1, hiddensize, inpsize)),
             ('r', TensorProto.FLOAT, (1, hiddensize, hiddensize))],
            [make_node('RNN', ['x', 'w', 'r'], ['all', 'last'], hidden_size=hiddensize)],
            [])
        self._assert_inferred(graph, [
            make_tensor_value_info('all', TensorProto.FLOAT, (seqlen, 1, batchsize, hiddensize)),
            make_tensor_value_info('last', TensorProto.FLOAT, (1, batchsize, hiddensize))])

    def test_rnn_forward(self):  # type: () -> None
        self._rnn_forward(64, 32, 10, 4)

    def _rnn_bidirectional(self, seqlen, batchsize, inpsize, hiddensize):  # type: (int, int, int, int) -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (seqlen, batchsize, inpsize)),
             ('w', TensorProto.FLOAT, (2, hiddensize, inpsize)),
             ('r', TensorProto.FLOAT, (2, hiddensize, hiddensize))],
            [make_node('RNN', ['x', 'w', 'r'], ['all', 'last'], hidden_size=hiddensize,
                direction="bidirectional")],
            [])
        self._assert_inferred(graph, [
            make_tensor_value_info('all', TensorProto.FLOAT, (seqlen, 2, batchsize, hiddensize)),
            make_tensor_value_info('last', TensorProto.FLOAT, (2, batchsize, hiddensize))])

    def test_rnn_bidirectional(self):  # type: () -> None
        self._rnn_bidirectional(64, 32, 10, 4)

    def _lstm_forward(self, seqlen, batchsize, inpsize, hiddensize):  # type: (int, int, int, int) -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (seqlen, batchsize, inpsize)),
             ('w', TensorProto.FLOAT, (1, 4 * hiddensize, inpsize)),
             ('r', TensorProto.FLOAT, (1, 4 * hiddensize, hiddensize))],
            [make_node('LSTM', ['x', 'w', 'r'], ['all', 'hidden', 'last'], hidden_size=hiddensize)],
            [])
        self._assert_inferred(graph, [
            make_tensor_value_info('all', TensorProto.FLOAT, (seqlen, 1, batchsize, hiddensize)),
            make_tensor_value_info('hidden', TensorProto.FLOAT, (1, batchsize, hiddensize)),
            make_tensor_value_info('last', TensorProto.FLOAT, (1, batchsize, hiddensize))])

    def test_lstm_forward(self):  # type: () -> None
        self._lstm_forward(64, 32, 10, 4)

    def test_topk_default_axis(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (3, 4, 5, 10))],
            [make_node('TopK', ['x'], ['y', 'z'], k=2)],
            [])
        self._assert_inferred(graph,
                              [make_tensor_value_info('y', TensorProto.FLOAT, (3, 4, 5, 2)),
                               make_tensor_value_info('z', TensorProto.INT64, (3, 4, 5, 2))])

    def test_topk(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (3, 4, 5, 10))],
            [make_node('TopK', ['x'], ['y', 'z'], k=2, axis=2)],
            [])
        self._assert_inferred(graph,
                              [make_tensor_value_info('y', TensorProto.FLOAT, (3, 4, 2, 10)),
                               make_tensor_value_info('z', TensorProto.INT64, (3, 4, 2, 10))])

    def test_gemm(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (7, 5)),
             ('y', TensorProto.FLOAT, (5, 11)),
             ('z', TensorProto.FLOAT, None)],
            [make_node('Gemm', ['x', 'y', 'z'], ['out'])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('out', TensorProto.FLOAT, (7, 11))])

    def test_gemm_transA(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (5, 7)),
             ('y', TensorProto.FLOAT, (5, 11)),
             ('z', TensorProto.FLOAT, None)],
            [make_node('Gemm', ['x', 'y', 'z'], ['out'], transA=1)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('out', TensorProto.FLOAT, (7, 11))])

    def test_gemm_transB(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (7, 5)),
             ('y', TensorProto.FLOAT, (11, 5)),
             ('z', TensorProto.FLOAT, None)],
            [make_node('Gemm', ['x', 'y', 'z'], ['out'], transB=1)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('out', TensorProto.FLOAT, (7, 11))])

    def test_gemm_transA_and_transB(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (5, 7)),
             ('y', TensorProto.FLOAT, (11, 5)),
             ('z', TensorProto.FLOAT, None)],
            [make_node('Gemm', ['x', 'y', 'z'], ['out'], transA=1, transB=1)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('out', TensorProto.FLOAT, (7, 11))])

    def test_reduce_op_shape_2_axis(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (24, 4, 11))],
            [make_node('ReduceL1', 'x', 'y', axes=(1, 2), keepdims=0)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (24,))])

    def test_reduce_op_shape_keep_dims(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (24, 4, 11))],
            [make_node('ReduceL1', 'x', 'y', axes=(1, 2), keepdims=1)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (24, 1, 1))])

    def test_reduce_op_shape_default_value(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (24, 4, 11))],
            [make_node('ReduceL1', 'x', 'y')],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (1, 1, 1))])

    def test_reduce_op_shape_negative_axis(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (24, 4, 11))],
            [make_node('ReduceL1', 'x', 'y', axes=(-1, -2))],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (24, 1, 1))])

    def test_argmax_shape(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (24, 4, 11))],
            [make_node('ArgMax', 'x', 'y', axis=1, keepdims=1)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.INT64, (24, 1, 11))])

    def test_argmax_shape_keepdims(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (24, 4, 11))],
            [make_node('ArgMax', 'x', 'y', axis=0, keepdims=0)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.INT64, (4, 11))])

    def test_argmax_shape_default_value(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (24, 4, 11))],
            [make_node('ArgMax', 'x', 'y')],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.INT64, (1, 4, 11))])

    def test_argmax_shape_negative_axis(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (24, 4, 11))],
            [make_node('ArgMax', 'x', 'y', axis=-2)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.INT64, (24, 1, 11))])

    def test_dropout(self):  # type: () -> None
        self._identity_prop('Dropout')

    def test_LRN(self):  # type: () -> None
        self._identity_prop('LRN', alpha=0.5, beta=0.5, size=1)

    def test_batch_norm(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (3, 4, 5, 6, 7)),
             ('scale', TensorProto.FLOAT, (4,)),
             ('b', TensorProto.FLOAT, (4,)),
             ('mean', TensorProto.FLOAT, (4,)),
             ('var', TensorProto.FLOAT, (4,))],
            [make_node('BatchNormalization', ['x', 'scale', 'b', 'mean', 'var'], ['out'])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('out', TensorProto.FLOAT, (3, 4, 5, 6, 7))])

    def test_split_from_GLU(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (5, 6, 7))],
            [make_node('Split', ['x'], ['y', 'z'], axis=1)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (5, 3, 7)),
                                      make_tensor_value_info('z', TensorProto.FLOAT, (5, 3, 7))])

    def test_GLU_partial(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (5, 6, 7))],
            [make_node('Split', ['x'], ['y', 'z'], axis=1),
             make_node('Sigmoid', ['z'], ['a'])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (5, 3, 7)),
                                      make_tensor_value_info('z', TensorProto.FLOAT, (5, 3, 7)),
                                      make_tensor_value_info('a', TensorProto.FLOAT, (5, 3, 7))])

    def test_GLU(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (5, 6, 7))],
            [make_node('Split', ['x'], ['y', 'z'], axis=1),
             make_node('Sigmoid', ['z'], ['a']),
             make_node('Mul', ['y', 'a'], ['b'])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (5, 3, 7)),
                                      make_tensor_value_info('z', TensorProto.FLOAT, (5, 3, 7)),
                                      make_tensor_value_info('a', TensorProto.FLOAT, (5, 3, 7)),
                                      make_tensor_value_info('b', TensorProto.FLOAT, (5, 3, 7))])

    def test_softmax(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (4, 5))],
            [make_node('Softmax', ['x'], 'z')],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, (4, 5))])

    def test_maxpool(self):  # type: () -> None
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (5, 3, 4, 4))],
            [make_node("MaxPool", ["X"], ["Y"], kernel_shape=[2, 2])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (5, 3, 3, 3))])

    def test_maxpool_with_indices(self):  # type: () -> None
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (5, 3, 4, 4))],
            [make_node("MaxPool", ["X"], ["Y", "Z"], kernel_shape=[2, 2])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (5, 3, 3, 3)),
                                      make_tensor_value_info("Z", TensorProto.INT64, (5, 3, 3, 3))])

    def test_maxpool_3D(self):  # type: () -> None
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (5, 3, 4, 4, 4))],
            [make_node("MaxPool", ["X"], ["Y"], kernel_shape=[2, 2, 2])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (5, 3, 3, 3, 3))])

    def test_maxpool_with_padding(self):  # type: () -> None
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (5, 3, 4, 4))],
            [make_node("MaxPool", ["X"], ["Y"], kernel_shape=[2, 2], pads=[1, 1, 2, 2])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (5, 3, 6, 6))])

    def test_maxpool_with_padding_and_stride(self):  # type: () -> None
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (5, 3, 4, 4))],
            [make_node("MaxPool", ["X"], ["Y"], kernel_shape=[2, 2], pads=[1, 1, 2, 2], strides=[2, 2])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (5, 3, 3, 3))])

    def test_averagepool(self):  # type: () -> None
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (5, 3, 4, 4))],
            [make_node("AveragePool", ["X"], ["Y"], kernel_shape=[2, 2])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (5, 3, 3, 3))])

    def test_averagepool_3D(self):  # type: () -> None
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (5, 3, 4, 4, 4))],
            [make_node("AveragePool", ["X"], ["Y"], kernel_shape=[2, 2, 2])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (5, 3, 3, 3, 3))])

    def test_averagepool_with_padding(self):  # type: () -> None
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (5, 3, 4, 4))],
            [make_node("AveragePool", ["X"], ["Y"], kernel_shape=[2, 2], pads=[1, 1, 2, 2])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (5, 3, 6, 6))])

    def test_averagepool_with_padding_and_stride(self):  # type: () -> None
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (5, 3, 4, 4))],
            [make_node("AveragePool", ["X"], ["Y"], kernel_shape=[2, 2], pads=[1, 1, 2, 2], strides=[2, 2])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (5, 3, 3, 3))])

    def test_lppool(self):  # type: () -> None
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (5, 3, 4, 4))],
            [make_node("LpPool", ["X"], ["Y"], kernel_shape=[2, 2])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (5, 3, 3, 3))])

    def test_lppool_3D(self):  # type: () -> None
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (5, 3, 4, 4, 4))],
            [make_node("LpPool", ["X"], ["Y"], kernel_shape=[2, 2, 2])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (5, 3, 3, 3, 3))])

    def test_lppool_with_padding(self):  # type: () -> None
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (5, 3, 4, 4))],
            [make_node("LpPool", ["X"], ["Y"], kernel_shape=[2, 2], pads=[1, 1, 2, 2])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (5, 3, 6, 6))])

    def test_lppool_with_padding_and_stride(self):  # type: () -> None
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (5, 3, 4, 4))],
            [make_node("LpPool", ["X"], ["Y"], kernel_shape=[2, 2], pads=[1, 1, 2, 2], strides=[2, 2])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (5, 3, 3, 3))])

    def test_roipool(self):  # type: () -> None
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (5, 3, 4, 4)),
            ("rois", TensorProto.INT64, (2, 5))],
            [make_node("MaxRoiPool", ["X", "rois"], ["Y"], pooled_shape=[2, 2])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (2, 3, 2, 2))])

    def test_lp_norm(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (3, 4, 5, 6, 7))],
            [make_node('LpNormalization', ['x'], ['out'])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('out', TensorProto.FLOAT, (3, 4, 5, 6, 7))])

    def test_instance_norm(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (3, 4, 5, 6, 7)),
             ('scale', TensorProto.FLOAT, (4,)),
             ('b', TensorProto.FLOAT, (4,))],
            [make_node('InstanceNormalization', ['x', 'scale', 'b'], ['out'])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('out', TensorProto.FLOAT, (3, 4, 5, 6, 7))])

    def test_global_maxpool(self):  # type: () -> None
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (5, 3, 4, 4))],
            [make_node("GlobalMaxPool", ["X"], ["Y"])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (5, 3, 1, 1))])

    def test_global_averagepool(self):  # type: () -> None
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (5, 3, 4, 4))],
            [make_node("GlobalAveragePool", ["X"], ["Y"])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (5, 3, 1, 1))])

    def test_global_lppool(self):  # type: () -> None
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (5, 3, 4, 4))],
            [make_node("GlobalLpPool", ["X"], ["Y"])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (5, 3, 1, 1))])

    def test_conv_transpose(self):  # type: () -> None
        graph = self._make_graph(
            [('X', TensorProto.FLOAT, (25, 48, 16, 16)),
             ('W', TensorProto.FLOAT, (48, 32, 3, 3))],
            [make_node('ConvTranspose', ['X', 'W'], 'Y', strides=[2, 2])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('Y', TensorProto.FLOAT, (25, 32, 33, 33))])

    def test_conv_transpose_with_pads(self):  # type: () -> None
        graph = self._make_graph(
            [('X', TensorProto.FLOAT, (25, 48, 16, 16)),
             ('W', TensorProto.FLOAT, (48, 32, 3, 3))],
            [make_node('ConvTranspose', ['X', 'W'], 'Y', strides=[2, 2], pads=[1, 1, 2, 2])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('Y', TensorProto.FLOAT, (25, 32, 30, 30))])

    def test_conv_transpose_with_output_shape(self):  # type: () -> None
        graph = self._make_graph(
            [('X', TensorProto.FLOAT, (25, 48, 16, 16)),
             ('W', TensorProto.FLOAT, (48, 32, 3, 3))],
            [make_node('ConvTranspose', ['X', 'W'], 'Y', strides=[2, 2], pads=[1, 1, 2, 2], output_shape=[36, 36])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('Y', TensorProto.FLOAT, (25, 32, 36, 36))])

    def test_conv_transpose_with_kernal_shape(self):  # type: () -> None
        graph = self._make_graph(
            [('X', TensorProto.FLOAT, (25, 48, 16, 16)),
             ('W', TensorProto.FLOAT, (48, 32, None, None))],
            [make_node('ConvTranspose', ['X', 'W'], 'Y', kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 2, 2])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('Y', TensorProto.FLOAT, (25, 32, 30, 30))])

    def test_mvn_function_output_shape(self):  # type: () -> None
        graph = self._make_graph(
            [('X', TensorProto.FLOAT, (25, 48, 16, 16))],
            [make_node('MeanVarianceNormalization', 'X', 'Y', axes=[0, 2, 3])],
            []
        )
        self._assert_inferred(graph, [make_tensor_value_info('Y', TensorProto.FLOAT, (25, 48, 16, 16))])

    def test_scan(self):    # type: () -> None
        batch_size = 1
        seq_len = 'sequence'
        input_size = 2
        loop_state_size = 3

        # can't use self._make_graph for the subgraph as it add more inputs for the Reshape operations it inserts.
        # this breaks the subgraph inferencing as it expects the number of inputs passed from Scan to match
        # the GraphProto, but Scan knows nothing about the additional inputs.
        input_value_infos = [make_tensor_value_info('loop_state_in', TensorProto.UNDEFINED, None),
                             make_tensor_value_info('input', TensorProto.UNDEFINED, None)]
        output_value_infos = [make_tensor_value_info('loop_state_out', TensorProto.UNDEFINED, None),
                              make_tensor_value_info('output', TensorProto.UNDEFINED, None)]

        subgraph = helper.make_graph(
            [make_node('Identity', ['loop_state_in'], ['loop_state_out']),
             make_node('Identity', ['input'], ['output'])],
            "subgraph",
            input_value_infos,
            output_value_infos
        )

        graph = self._make_graph(
            [('loop_state_orig', TensorProto.FLOAT, (batch_size, loop_state_size)),
             ('scan_input', TensorProto.FLOAT, (batch_size, seq_len, input_size))],
            [make_node('Scan', ['', 'loop_state_orig', 'scan_input'], ['loop_state_final', 'scan_output'],
                       num_scan_inputs=1, body=subgraph)],
            []
        )

        self._assert_inferred(
            graph,
            [make_tensor_value_info('loop_state_final', TensorProto.FLOAT, (batch_size, loop_state_size)),
             make_tensor_value_info('scan_output', TensorProto.FLOAT, (batch_size, seq_len, input_size))],
            opset_imports=[helper.make_opsetid("", 8)])

    def test_scan_opset9(self):    # type: () -> None
        seq_len = 'sequence'
        input_size = 2
        loop_state_size = 3

        # can't use self._make_graph for the subgraph as it add more inputs for the Reshape operations it inserts.
        # this breaks the subgraph inferencing as it expects the number of inputs passed from Scan to match
        # the GraphProto, but Scan knows nothing about the additional inputs.
        input_value_infos = [make_tensor_value_info('loop_state_in', TensorProto.UNDEFINED, None),
                             make_tensor_value_info('input', TensorProto.UNDEFINED, None)]
        output_value_infos = [make_tensor_value_info('loop_state_out', TensorProto.UNDEFINED, None),
                              make_tensor_value_info('output', TensorProto.UNDEFINED, None)]

        subgraph = helper.make_graph(
            [make_node('Identity', ['loop_state_in'], ['loop_state_out']),
             make_node('Identity', ['input'], ['output'])],
            "subgraph",
            input_value_infos,
            output_value_infos
        )

        graph = self._make_graph(
            [('loop_state_orig', TensorProto.FLOAT, (loop_state_size,)),
             ('scan_input', TensorProto.FLOAT, (seq_len, input_size))],
            [make_node('Scan', ['loop_state_orig', 'scan_input'], ['loop_state_final', 'scan_output'],
                       num_scan_inputs=1, body=subgraph)],
            []
        )

        self._assert_inferred(
            graph,
            [make_tensor_value_info('loop_state_final', TensorProto.FLOAT, (loop_state_size,)),
             make_tensor_value_info('scan_output', TensorProto.FLOAT, (seq_len, input_size))],
            opset_imports=[helper.make_opsetid("", 9)])

    def test_scan_opset9_axes(self):    # type: () -> None
        axis_0_len = 'axis0'
        seq_len = 'sequence'
        input_size = 2
        loop_state_size = 3

        # can't use self._make_graph for the subgraph as it add more inputs for the Reshape operations it inserts.
        # this breaks the subgraph inferencing as it expects the number of inputs passed from Scan to match
        # the GraphProto, but Scan knows nothing about the additional inputs.
        input_value_infos = [make_tensor_value_info('loop_state_in', TensorProto.UNDEFINED, None),
                             make_tensor_value_info('input', TensorProto.UNDEFINED, None)]
        output_value_infos = [make_tensor_value_info('loop_state_out', TensorProto.UNDEFINED, None),
                              make_tensor_value_info('output', TensorProto.UNDEFINED, None)]

        subgraph = helper.make_graph(
            [make_node('Identity', ['loop_state_in'], ['loop_state_out']),
             make_node('Identity', ['input'], ['output'])],
            "subgraph",
            input_value_infos,
            output_value_infos
        )

        graph = self._make_graph(
            [('loop_state_orig', TensorProto.FLOAT, (loop_state_size,)),
             ('scan_input', TensorProto.FLOAT, (axis_0_len, seq_len, input_size))],
            [make_node('Scan', ['loop_state_orig', 'scan_input'], ['loop_state_final', 'scan_output'],
                       num_scan_inputs=1, body=subgraph, axes=[1])],
            []
        )

        self._assert_inferred(
            graph,
            [make_tensor_value_info('loop_state_final', TensorProto.FLOAT, (loop_state_size,)),
             make_tensor_value_info('scan_output', TensorProto.FLOAT, (seq_len, axis_0_len, input_size))],
            opset_imports=[helper.make_opsetid("", 9)])

    def test_if(self):  # type: () -> None

        # Create a simple If node where the 'then' subgraph adds to the current value, and the 'else' subgraph
        # subtracts.
        # can't use self._make_graph for the subgraphs as that add more inputs for the Reshape operations it inserts.
        # this breaks the subgraph inferencing as it expects the subgraphs to have zero inputs
        then_subgraph = helper.make_graph(
            [make_node('Add', ['current_value', 'add_value'], ['then_output'])],
            "then_subgraph",
            [],  # no inputs
            [make_tensor_value_info('then_output', TensorProto.UNDEFINED, None)],
        )

        else_subgraph = helper.make_graph(
            [make_node('Sub', ['current_value', 'sub_value'], ['else_output'])],
            "else_subgraph",
            [],  # no inputs
            [make_tensor_value_info('else_output', TensorProto.UNDEFINED, None)],
        )

        graph = self._make_graph(
            [('cond', TensorProto.BOOL, (1,)),
             ('current_value', TensorProto.FLOAT, (1,)),
             ('add_value', TensorProto.FLOAT, (1,)),
             ('sub_value', TensorProto.FLOAT, (1,))],
            [make_node('If', ['cond'], ['if_output'],
                       then_branch=then_subgraph, else_branch=else_subgraph)],
            []
        )

        self._assert_inferred(graph, [make_tensor_value_info('if_output', TensorProto.FLOAT, (1,))])

    def test_maxunpool_shape_without_output_shape(self):  # type: () -> None
        graph = self._make_graph(
            [('xT', TensorProto.FLOAT, (1, 1, 2, 2)),
             ('xI', TensorProto.FLOAT, (1, 1, 2, 2))],
            [make_node('MaxUnpool', ['xT', 'xI'], 'Y', kernel_shape=[2, 2], strides=[2, 2])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('Y', TensorProto.FLOAT, (1, 1, 4, 4))])

    def test_maxunpool_shape_with_output_shape(self):  # type: () -> None
        graph = self._make_graph(
            [('xT', TensorProto.FLOAT, (1, 1, 2, 2)),
             ('xI', TensorProto.FLOAT, (1, 1, 2, 2)),
             ('output_shape', TensorProto.FLOAT, (4, ))],
            [make_node('MaxUnpool', ['xT', 'xI', 'output_shape'], 'Y', kernel_shape=[2, 2], strides=[2, 2])],
            [make_tensor_value_info("Y", TensorProto.FLOAT, None)])
        self._assert_inferred(graph, [make_tensor_value_info("Y", TensorProto.FLOAT, None)])

    def test_onehot_without_axis(self):  # type: () -> None
        graph = self._make_graph(
            [('indices', TensorProto.INT64, (2, 2)),
             ('depth', TensorProto.INT64, (1, )),
             ('values', TensorProto.FLOAT, (2, ))],
            [make_node('OneHot', ['indices', 'depth', 'values'], 'Y')],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('Y', TensorProto.FLOAT, (2, 2, None))])  # type: ignore

    def test_onehot_with_axis(self):  # type: () -> None
        graph = self._make_graph(
            [('indices', TensorProto.INT64, (2, 3, 5)),
             ('depth', TensorProto.INT64, (1, )),
             ('values', TensorProto.FLOAT, (2, ))],
            [make_node('OneHot', ['indices', 'depth', 'values'], 'Y', axis=1)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('Y', TensorProto.FLOAT, (2, None, 3, 5))])  # type: ignore

    def test_loop(self):    # type: () -> None
        # can't use self._make_graph for the subgraph as it add more inputs for the Reshape operations it inserts.
        # this breaks the subgraph inferencing as it expects the number of inputs passed from Loop to match
        # the GraphProto, but Loop knows nothing about the additional inputs.
        input_value_infos = [make_tensor_value_info('iter_num_in', TensorProto.INT64, (1,)),
                             make_tensor_value_info('cond_in', TensorProto.UNDEFINED, None),
                             make_tensor_value_info('loop_state_in', TensorProto.UNDEFINED, ())]
        output_value_infos = [make_tensor_value_info('cond_out', TensorProto.UNDEFINED, None),
                              make_tensor_value_info('loop_state_out', TensorProto.UNDEFINED, None),
                              make_tensor_value_info('output', TensorProto.FLOAT, (3,))]

        subgraph = helper.make_graph(
            [make_node('Identity', ['cond_in'], ['cond_out']),
             make_node('Identity', ['loop_state_in'], ['loop_state_out']),
             make_node('Identity', ['outer_scope_input'], ['output'])],
            "subgraph",
            input_value_infos,
            output_value_infos
        )

        graph = self._make_graph(
            [('max_trip_count', TensorProto.INT64, (1,)),
             ('cond_orig', TensorProto.FLOAT, (1,)),
             ('loop_state_orig', TensorProto.FLOAT, (2,)),
             ('outer_scope_input', TensorProto.FLOAT, (3,))],
            [make_node('Loop', ['max_trip_count', 'cond_orig', 'loop_state_orig'], ['loop_state_final', 'loop_output'],
                       body=subgraph)],
            []
        )

        self._assert_inferred(
            graph,
            [make_tensor_value_info('loop_state_final', TensorProto.FLOAT, None),  # shape may change between iterations
             make_tensor_value_info('loop_output', TensorProto.FLOAT, (None, 3))])  # type: ignore


if __name__ == '__main__':
    unittest.main()
