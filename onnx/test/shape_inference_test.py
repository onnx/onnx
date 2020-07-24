from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from onnx import checker, helper, TensorProto, NodeProto, GraphProto, ValueInfoProto, ModelProto, ONNX_ML, SparseTensorProto
from onnx.defs import ONNX_DOMAIN, ONNX_ML_DOMAIN, AI_ONNX_PREVIEW_TRAINING_DOMAIN
from onnx.helper import make_node, make_tensor, make_tensor_value_info, make_empty_tensor_value_info, make_opsetid, make_sequence_value_info
from typing import Sequence, Union, Text, Tuple, List, Any, Optional
import onnx.shape_inference
import unittest
import os
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
        self._make_matmul_test_allow_unknown((3,), None, None)
        self._make_matmul_test_allow_unknown(None, None, None)

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

    def test_concat_param_single_input(self):  # type: () -> None
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, ("a", 2))],
            [make_node("Concat", ['x'], ['z'], axis=0)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, ("a", 2))])

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
        self._assert_inferred(
            graph,
            [make_tensor_value_info('y', TensorProto.INT32, (2, 4, 3, 9))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, 9)])

    def test_upsample_raw_data(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.INT32, (2, 4, 3, 5)),
             ('scales', TensorProto.FLOAT, (4,))],
            [make_node("Upsample", ['x', 'scales'], ['y'])],
            [],
            initializer=[make_tensor('scales', TensorProto.FLOAT, (4,),
                                     vals=np.array([1.0, 1.1, 1.3, 1.9], dtype='<f4').tobytes(), raw=True)])  # Feed raw bytes (force little endian ordering like onnx standard) for test purpose
        self._assert_inferred(
            graph,
            [make_tensor_value_info('y', TensorProto.INT32, (2, 4, 3, 9))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, 9)])

    def test_upsample_raw_data_v7(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.INT32, (1, 3, 4, 5))],
            [make_node("Upsample", ['x'], ['y'], scales=[2.0, 1.1, 2.3, 1.9])],
            [])
        self._assert_inferred(
            graph,
            [make_tensor_value_info('y', TensorProto.INT32, (2, 3, 9, 9))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, 7)])

    def test_expand(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.INT32, (3, 1)),
             ('shape', TensorProto.INT64, (3,))],
            [make_node("Expand", ['x', 'shape'], ['y'])],
            [],
            initializer=[make_tensor('shape', TensorProto.INT64, (3,), (2, 1, 6))])
        self._assert_inferred(
            graph,
            [make_tensor_value_info('y', TensorProto.INT32, (2, 3, 6))])

    def test_expand_scalar_input(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.INT32, ()),
             ('shape', TensorProto.INT64, (2,))],
            [make_node("Expand", ['x', 'shape'], ['y'])],
            [],
            initializer=[make_tensor('shape', TensorProto.INT64, (2,), (4, 8))])
        self._assert_inferred(
            graph,
            [make_tensor_value_info('y', TensorProto.INT32, (4, 8))])

    def test_expand_raw_data(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.INT32, (3, 1)),
             ('shape', TensorProto.INT64, (2,))],
            [make_node("Expand", ['x', 'shape'], ['y'])],
            [],
            initializer=[make_tensor('shape', TensorProto.INT64, (2,),
                                     vals=np.array([3, 4], dtype='<i8').tobytes(), raw=True)])  # Feed raw bytes (force little endian ordering like onnx standard) for test purpose
        self._assert_inferred(
            graph,
            [make_tensor_value_info('y', TensorProto.INT32, (3, 4))])

    def test_resize_size(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.INT32, (2, 4, 3, 5)),
             ('roi', TensorProto.FLOAT, (8,)),
             ('scales', TensorProto.FLOAT, (4,)),
             ('sizes', TensorProto.INT64, (4,))],
            [make_node("Resize", ['x', 'roi', 'scales', 'sizes'], ['y'])],
            [],
            initializer=[make_tensor('sizes', TensorProto.INT64, (4,), (3, 5, 6, 7))])
        self._assert_inferred(
            graph,
            [make_tensor_value_info('y', TensorProto.INT32, (3, 5, 6, 7))])

    def test_resize_scale(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.INT32, (2, 4, 3, 5)),
             ('roi', TensorProto.FLOAT, (8,)),
             ('scales', TensorProto.FLOAT, (4,))],
            [make_node("Resize", ['x', 'roi', 'scales'], ['y'])],
            [],
            initializer=[make_tensor('scales', TensorProto.FLOAT, (4,), (1.0, 1.1, 1.3, 1.9))])
        self._assert_inferred(
            graph,
            [make_tensor_value_info('y', TensorProto.INT32, (2, 4, 3, 9))])

    def test_resize_scale_raw_data(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.INT32, (1, 3, 4, 5)),
             ('roi', TensorProto.FLOAT, (8,)),
             ('scales', TensorProto.FLOAT, (4,))],
            [make_node("Resize", ['x', 'roi', 'scales'], ['y'])],
            [],
            initializer=[make_tensor('scales', TensorProto.FLOAT, (4,),
                                     vals=np.array([2.0, 1.1, 2.3, 1.9], dtype='<f4').tobytes(), raw=True)])
        self._assert_inferred(
            graph,
            [make_tensor_value_info('y', TensorProto.INT32, (2, 3, 9, 9))])

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

    def test_gather_elements(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (2, 2)),
             ('i', TensorProto.INT64, (2, 2))],
            [make_node("GatherElements", ['x', 'i'], ['y'], axis=1)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (2, 2))])  # type: ignore

    def test_gather_elements_axis0(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (3, 3)),
             ('i', TensorProto.INT64, (2, 3))],
            [make_node("GatherElements", ['x', 'i'], ['y'], axis=0)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (2, 3))])  # type: ignore

    def test_scatter(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (3, 3)),
             ('i', TensorProto.INT64, (2, 3)),
             ('u', TensorProto.FLOAT, (2, 3))],
            [make_node("Scatter", ['x', 'i', 'u'], ['y'])],
            [])
        self._assert_inferred(
            graph,
            [make_tensor_value_info('y', TensorProto.FLOAT, (3, 3))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, 10)])  # type: ignore

    def test_scatter_axis1(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (1, 5)),
             ('i', TensorProto.INT64, (1, 2)),
             ('u', TensorProto.FLOAT, (1, 2))],
            [make_node("Scatter", ['x', 'i', 'u'], ['y'], axis=1)],
            [])
        self._assert_inferred(
            graph,
            [make_tensor_value_info('y', TensorProto.FLOAT, (1, 5))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, 10)])  # type: ignore

    def test_scatter_elements(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (3, 3)),
             ('i', TensorProto.INT64, (2, 3)),
             ('u', TensorProto.FLOAT, (2, 3))],
            [make_node("ScatterElements", ['x', 'i', 'u'], ['y'])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (3, 3))])  # type: ignore

    def test_scatter_elements_axis1(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (1, 5)),
             ('i', TensorProto.INT64, (1, 2)),
             ('u', TensorProto.FLOAT, (1, 2))],
            [make_node("ScatterElements", ['x', 'i', 'u'], ['y'], axis=1)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (1, 5))])  # type: ignore

    def test_scatternd(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (4, 5, 6)),
             ('indices', TensorProto.INT64, (3, 3, 2)),
             ('updates', TensorProto.FLOAT, (3, 3, 6))],
            [make_node("ScatterND", ['x', 'indices', 'updates'], ['y'])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (4, 5, 6))])  # type: ignore

    def test_scatternd_noshape(self):  # type: () -> None
        # The shape of 'x_reshaped' cannot be inferred, since it is the output of a dynamic reshape.
        # Thus the shape of 'y' is also None.
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (4, 5, 6)),
             ('indices', TensorProto.INT64, (3, 3, 2)),
             ('updates', TensorProto.FLOAT, (3, 3, 6)),
             ('shape', TensorProto.UNDEFINED, (2,))],
            [make_node("Reshape", ['x', 'shape'], ['x_reshaped']),
             make_node("ScatterND", ['x_reshaped', 'indices', 'updates'], ['y'])],
            [])
        self._assert_inferred(graph, [
            make_tensor_value_info('x_reshaped', TensorProto.FLOAT, None),
            make_tensor_value_info('y', TensorProto.FLOAT, None)])  # type: ignore

    def test_squeeze(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (1, 3, 1, 1, 2, 1))],
            [make_node('Squeeze', 'x', 'y', axes=[0, 2, 3, 5])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (3, 2))])

    def test_unsqueeze_regular(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (3, 2))],
            [make_node('Unsqueeze', 'x', 'y', axes=[0, 1, 3, 5])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (1, 1, 3, 1, 2, 1))])

    def test_unsqueeze_unsorted_axes(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (3, 4, 5))],
            [make_node('Unsqueeze', 'x', 'y', axes=[4, 0])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (1, 3, 4, 5, 1))])

    def test_unsqueeze_negative_axes(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (3, 4, 5))],
            [make_node('Unsqueeze', 'x', 'y', axes=[0, -1])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (1, 3, 4, 5, 1))])

    def test_slice_without_input_shape(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (3, 2)), ('starts', TensorProto.INT64, (1,)), ('ends', TensorProto.INT64, (1,))],
            [make_node('Slice', ['x', 'starts', 'ends'], ['y'])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, None)])

    def test_slice_with_input_shape(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (3, 2)), ('starts', TensorProto.INT64, (2, )), ('ends', TensorProto.INT64, (2, ))],
            [make_node('Slice', ['x', 'starts', 'ends'], ['y'])],
            [],
            initializer=[make_tensor('starts', TensorProto.INT64, (2, ),
                                      vals=np.array([1, 0], dtype='<i8').tobytes(), raw=True),  # Feed raw bytes (force little endian ordering like onnx standard) for test purpose
                         make_tensor('ends', TensorProto.INT64, (2, ), (2, 2))])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (1, 2))])

    def test_slice_with_input_shape_containing_dim_params(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (1, 'a', 1)),
             ('starts', TensorProto.INT64, (3,)),
             ('ends', TensorProto.INT64, (3,))],
            [make_node('Slice', ['x', 'starts', 'ends'], ['y'])],
            [],
            initializer=[make_tensor('starts', TensorProto.INT64, (3,), (0, 0, 0)),
                            make_tensor('ends', TensorProto.INT64, (3,), (1, 1, 1))])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (1, None, 1))])  # type: ignore

    def test_slice_with_input_shape_steps(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (5, 6, 7)),
             ('starts', TensorProto.INT64, (3,)),
             ('ends', TensorProto.INT64, (3,)),
             ('axes'),
             ('steps', TensorProto.INT64, (3,))],
            [make_node('Slice', ['x', 'starts', 'ends', 'axes', 'steps'], ['y'])],
            [],
            initializer=[make_tensor('starts', TensorProto.INT64, (3,), (1, 0, 0)),
                         make_tensor('ends', TensorProto.INT64, (3,), (2, 6, 6)),
                         make_tensor('steps', TensorProto.INT64, (3,), (1, 4, 3))])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (1, 2, 2))])

    def test_slice_with_input_shape_axes(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (3, 6, 2)),
             ('starts', TensorProto.INT64, (2,)),
             ('ends', TensorProto.INT64, (2,)),
             ('axes', TensorProto.INT64, (2,)),
             ('steps')],
            [make_node('Slice', ['x', 'starts', 'ends', 'axes', 'steps'], ['y'])],
            [],
            initializer=[make_tensor('starts', TensorProto.INT64, (2,), (1, 0)),
                         make_tensor('ends', TensorProto.INT64, (2,), (2, 2)),
                         make_tensor('axes', TensorProto.INT64, (2,), (0, 2))])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (1, 6, 2))])

    def test_slice_unsorted_axes(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (3, 2)),
             ('starts', TensorProto.INT64, (2,)),
             ('ends', TensorProto.INT64, (2,)),
             ('axes', TensorProto.INT64, (2,))],
            [make_node('Slice', ['x', 'starts', 'ends', 'axes'], 'y')],
            [],
            initializer=[make_tensor('starts', TensorProto.INT64, (2,), (1, 0)),
                         make_tensor('ends', TensorProto.INT64, (2,), (2, 2)),
                         make_tensor('axes', TensorProto.INT64, (2,), (1, 0))])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (2, 1))])  # can handle unsorted axes

    def test_slice_giant_number(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (3, 2)),
             ('starts', TensorProto.INT64, (2,)),
             ('ends', TensorProto.INT64, (2,)),
             ('axes', TensorProto.INT64, (2,))],
            [make_node('Slice', ['x', 'starts', 'ends', 'axes'], 'y')],
            [],
            initializer=[make_tensor('starts', TensorProto.INT64, (2,), (1, 0)),
                         make_tensor('ends', TensorProto.INT64, (2,), (200, 22000)),
                         make_tensor('axes', TensorProto.INT64, (2,), (0, 1))])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (2, 2))])

    def test_slice_giant_step(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (3, 2)),
             ('starts', TensorProto.INT64, (2,)),
             ('ends', TensorProto.INT64, (2,)),
             ('axes', TensorProto.INT64, (2,)),
             ('steps', TensorProto.INT64, (2,))],
            [make_node('Slice', ['x', 'starts', 'ends', 'axes', 'steps'], 'y')],
            [],
            initializer=[make_tensor('starts', TensorProto.INT64, (2,), (1, 0)),
                         make_tensor('ends', TensorProto.INT64, (2,), (200, 200)),
                         make_tensor('axes', TensorProto.INT64, (2,), (0, 1)),
                         make_tensor('steps', TensorProto.INT64, (2,), (1, 200))])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (2, 1))])

    def test_slice_negative_end(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (3, 2)),
             ('starts', TensorProto.INT64, (2,)),
             ('ends', TensorProto.INT64, (2,)),
             ('axes', TensorProto.INT64, (2,))],
            [make_node('Slice', ['x', 'starts', 'ends', 'axes'], 'y')],
            [],
            initializer=[make_tensor('starts', TensorProto.INT64, (2,), (1, 0)),
                         make_tensor('ends', TensorProto.INT64, (2,), (200, -1)),  # negative end means begin from end of a dimension (here end = 2 - 1 = 1)
                         make_tensor('axes', TensorProto.INT64, (2,), (0, 1))])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (2, 1))])  # type: ignore

    def test_slice_negative_start(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (3, 2)),
             ('starts', TensorProto.INT64, (2,)),
             ('ends', TensorProto.INT64, (2,)),
             ('axes', TensorProto.INT64, (2,))],
            [make_node('Slice', ['x', 'starts', 'ends', 'axes'], 'y')],
            [],
            initializer=[make_tensor('starts', TensorProto.INT64, (2,), (1, -2)),  # negative start means begin from end of a dimension (here end = 2 - 2 = 0)
                         make_tensor('ends', TensorProto.INT64, (2,), (200, 3)),
                         make_tensor('axes', TensorProto.INT64, (2,), (0, 1))])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (2, 2))])  # type: ignore

    def test_slice_negative_step(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (3, 4)),
             ('starts', TensorProto.INT64, (2,)),
             ('ends', TensorProto.INT64, (2,)),
             ('axes', TensorProto.INT64, (2,)),
             ('steps', TensorProto.INT64, (2,))],
            [make_node('Slice', ['x', 'starts', 'ends', 'axes', 'steps'], 'y')],
            [],
            initializer=[make_tensor('starts', TensorProto.INT64, (2,), (1, 4)),  # 4 will be clamped to 3 since we are negative stepping
                         make_tensor('ends', TensorProto.INT64, (2,), (200, 0)),
                         make_tensor('axes', TensorProto.INT64, (2,), (0, 1)),
                         make_tensor('steps', TensorProto.INT64, (2,), (1, -1))])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (2, 3))])  # type: ignore

    def test_slice_variable_copy(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, ("a", 2)),
             ('starts', TensorProto.INT64, (1,)),
             ('ends', TensorProto.INT64, (1,)),
             ('axes', TensorProto.INT64, (1,))],
            [make_node('Slice', ['x', 'starts', 'ends', 'axes'], 'y')],
            [],
            initializer=[make_tensor('starts', TensorProto.INT64, (1,), (1,)),
                         make_tensor('ends', TensorProto.INT64, (1,), (200,)),
                         make_tensor('axes', TensorProto.INT64, (1,), (1,))])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, ("a", 1))])  # type: ignore

    def test_slice_variable_input_types(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.DOUBLE, (3, 2)),
             ('starts', TensorProto.INT32, (2,)),
             ('ends', TensorProto.INT32, (2,)),
             ('axes', TensorProto.INT32, (2,))],
            [make_node('Slice', ['x', 'starts', 'ends', 'axes'], 'y')],
            [],
            initializer=[make_tensor('starts', TensorProto.INT32, (2,), (1, 0)),
                         make_tensor('ends', TensorProto.INT32, (2,), (200, 22000)),
                         make_tensor('axes', TensorProto.INT32, (2,), (0, 1))])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.DOUBLE, (2, 2))])

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

    def test_conv_auto_pad(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (30, 4, 7, 6, 4)),
             ('y', TensorProto.FLOAT, (50, 4, 4, 3, 2))],
            [make_node('Conv', ['x', 'y'], 'z', auto_pad='SAME_UPPER')],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, (30, 50, 7, 6, 4))])

    def test_conv_auto_pad_dilation(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (30, 4, 65, 64, 63)),
             ('y', TensorProto.FLOAT, (50, 4, 4, 3, 2))],
            [make_node('Conv', ['x', 'y'], 'z', auto_pad='SAME_UPPER', dilations=[2, 3, 4])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, (30, 50, 65, 64, 63))])

    def test_conv_group(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (30, 4, 8, 8, 8)),
             ('y', TensorProto.FLOAT, (4, 1, 8, 8, 8))],
            [make_node('Conv', ['x', 'y'], 'z', group=4)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, (30, 4, 1, 1, 1))])

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

    def test_bitshift(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.UINT32, (2, 3, 1)),
             ('y', TensorProto.UINT32, (2, 3, 1))],
            [make_node('BitShift', ['x', 'y'], 'z', direction="RIGHT")],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.UINT32, (2, 3, 1))])

    def test_bitshift_broadcast_to_first(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.UINT32, (16, 4, 1)),
             ('y', TensorProto.UINT32, (1,))],
            [make_node('BitShift', ['x', 'y'], 'z', direction="RIGHT")],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.UINT32, (16, 4, 1))])

    def test_bitshift_broadcast_to_second(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.UINT32, (1,)),
             ('y', TensorProto.UINT32, (2, 3, 1))],
            [make_node('BitShift', ['x', 'y'], 'z', direction="RIGHT")],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.UINT32, (2, 3, 1))])

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

    def _logical_binary_op(self, op, input_type):  # type: (Text, TensorProto.DataType) -> None
        graph = self._make_graph(
            [('x', input_type, (30, 4, 5)),
             ('y', input_type, (30, 4, 5))],
            [make_node(op, ['x', 'y'], 'z')],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.BOOL, (30, 4, 5))])

    def _logical_binary_op_with_broadcasting(self, op, input_type):  # type: (Text, TensorProto.DataType) -> None
        graph = self._make_graph(
            [('x', input_type, (1, 5)),
             ('y', input_type, (30, 4, 5))],
            [make_node(op, ['x', 'y'], 'z')],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.BOOL, (30, 4, 5))])

    def test_logical_and(self):  # type: () -> None
        self._logical_binary_op('And', TensorProto.BOOL)
        self._logical_binary_op_with_broadcasting('And', TensorProto.BOOL)

    def test_logical_or(self):  # type: () -> None
        self._logical_binary_op('Or', TensorProto.BOOL)
        self._logical_binary_op_with_broadcasting('Or', TensorProto.BOOL)

    def test_logical_xor(self):  # type: () -> None
        self._logical_binary_op('Xor', TensorProto.BOOL)
        self._logical_binary_op_with_broadcasting('Xor', TensorProto.BOOL)

    def test_greater(self):  # type: () -> None
        self._logical_binary_op('Greater', TensorProto.BOOL)
        self._logical_binary_op_with_broadcasting('Greater', TensorProto.BOOL)

    def test_less(self):  # type: () -> None
        self._logical_binary_op('Less', TensorProto.BOOL)
        self._logical_binary_op_with_broadcasting('Less', TensorProto.BOOL)

    def test_equal(self):  # type: () -> None
        self._logical_binary_op('Equal', TensorProto.BOOL)
        self._logical_binary_op_with_broadcasting('Equal', TensorProto.BOOL)

    def test_logical_not(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.BOOL, (30, 4, 5))],
            [make_node('Not', ['x'], 'z')],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.BOOL, (30, 4, 5))])

    def test_less_or_equal(self):  # type: () -> None
        self._logical_binary_op('LessOrEqual', TensorProto.BOOL)
        self._logical_binary_op_with_broadcasting('LessOrEqual', TensorProto.BOOL)

    def test_greater_or_equal(self):  # type: () -> None
        self._logical_binary_op('GreaterOrEqual', TensorProto.BOOL)
        self._logical_binary_op_with_broadcasting('GreaterOrEqual', TensorProto.BOOL)

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
            [make_node('DepthToSpace', ['x'], ['z'], blocksize=b, mode='DCR')],
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
            [make_node('TopK', ['x', 'k'], ['y', 'z'])],
            [],
            initializer=[make_tensor('k', TensorProto.INT64, (1,), (2,))])
        self._assert_inferred(graph,
                              [make_tensor_value_info('y', TensorProto.FLOAT, (3, 4, 5, 2)),
                               make_tensor_value_info('z', TensorProto.INT64, (3, 4, 5, 2))])

    def test_topk(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (3, 4, 5, 10))],
            [make_node('TopK', ['x', 'k'], ['y', 'z'], axis=2)],
            [],
            initializer=[make_tensor('k', TensorProto.INT64, (1,), (2,))])
        self._assert_inferred(graph,
                              [make_tensor_value_info('y', TensorProto.FLOAT, (3, 4, 2, 10)),
                               make_tensor_value_info('z', TensorProto.INT64, (3, 4, 2, 10))])

    def test_topk_raw_data(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (3, 4, 5, 10))],
            [make_node('TopK', ['x', 'k'], ['y', 'z'], axis=2)],
            [],
            initializer=[make_tensor('k', TensorProto.INT64, (1,),
                                      vals=np.array([3], dtype='<i8').tobytes(), raw=True)])  # Feed raw bytes (force little endian ordering like onnx standard) for test purpose
        self._assert_inferred(graph,
                              [make_tensor_value_info('y', TensorProto.FLOAT, (3, 4, 3, 10)),
                               make_tensor_value_info('z', TensorProto.INT64, (3, 4, 3, 10))])

    def test_topk_missing_k_value_output_rank_check(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (3, 4, 5, 10)),
            ('k', TensorProto.INT64, (1,))],
            [make_node('TopK', ['x', 'k'], ['y', 'z'], axis=2)],
            [])
        self._assert_inferred(graph,
                              [make_tensor_value_info('y', TensorProto.FLOAT, (None, None, None, None)),  # type: ignore
                               make_tensor_value_info('z', TensorProto.INT64, (None, None, None, None))])  # type: ignore

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

    def test_gemm_no_bias(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (13, 7)),
             ('y', TensorProto.FLOAT, (7, 17))],
            [make_node('Gemm', ['x', 'y'], ['out'])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('out', TensorProto.FLOAT, (13, 17))])

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

    def test_reduce_op_shape_no_axes_do_not_keep_dims(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (24, 4, 11))],
            [make_node('ReduceL1', 'x', 'y', keepdims=0)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, tuple())])

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
        graph = self._make_graph(
            [('data', TensorProto.FLOAT, (3, 4, 5,)),
             ('ratio', TensorProto.FLOAT, ())],
            [make_node('Dropout', ['data', 'ratio'], ['out'])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('out', TensorProto.FLOAT, (3, 4, 5,))])

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

    def test_split_negative_axis(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (2, 4))],
            [make_node('Split', ['x'], ['y', 'z'], axis=-1)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (2, 2)),
                                      make_tensor_value_info('z', TensorProto.FLOAT, (2, 2))])

    def test_split_with_split_attribute(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (2, 4))],
            [make_node('Split', ['x'], ['y', 'z'], axis=1, split=[3, 1])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (2, 3)),
                                      make_tensor_value_info('z', TensorProto.FLOAT, (2, 1))])

    def test_split_with_split_attribute_unknown_split_dim(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (2, 'a', 'b'))],
            [make_node('Split', ['x'], ['y', 'z'], axis=1, split=[3, 1])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (2, None, 'b')),  # type: ignore
                                      make_tensor_value_info('z', TensorProto.FLOAT, (2, None, 'b'))])  # type: ignore

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

    def test_softmax_2d(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (4, 5))],
            [make_node('Softmax', ['x'], 'z')],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, (4, 5))])

    def test_softmax_3d(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (4, 5, 6))],
            [make_node('Softmax', ['x'], 'z')],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, (4, 5, 6))])

    def test_hardmax_2d(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (4, 5))],
            [make_node('Hardmax', ['x'], 'z')],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, (4, 5))])

    def test_hardmax_3d(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (4, 5, 6))],
            [make_node('Hardmax', ['x'], 'z')],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, (4, 5, 6))])

    def test_logsoftmax_2d(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (4, 5))],
            [make_node('LogSoftmax', ['x'], 'z')],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, (4, 5))])

    def test_logsoftmax_3d(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (4, 5, 6))],
            [make_node('LogSoftmax', ['x'], 'z')],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, (4, 5, 6))])

    def test_logsoftmax_3d_negative_axis(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (4, 5, 6))],
            [make_node('LogSoftmax', ['x'], 'z', axis=-1)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, (4, 5, 6))])

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

    def test_maxpool_with_floor_mode(self):  # type: () -> None
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (32, 288, 35, 35))],
            [make_node("MaxPool", ["X"], ["Y"], kernel_shape=[2, 2], strides=[2, 2], ceil_mode=False)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (32, 288, 17, 17))])

    def test_maxpool_with_ceil_mode(self):  # type: () -> None
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (32, 288, 35, 35))],
            [make_node("MaxPool", ["X"], ["Y"], kernel_shape=[2, 2], strides=[2, 2], ceil_mode=True)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (32, 288, 18, 18))])

    def test_maxpool_ceil(self):  # type: () -> None
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (1, 1, 4, 4))],
            [make_node("MaxPool", ["X"], ["Y"], kernel_shape=[3, 3], strides=[2, 2], ceil_mode=True)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (1, 1, 2, 2))])

    def test_maxpool_with_dilations(self):  # type: () -> None
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (5, 3, 4, 4))],
            [make_node("MaxPool", ["X"], ["Y"], kernel_shape=[2, 2], dilations=[2, 2])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (5, 3, 2, 2))])

    def test_maxpool_with_same_upper_padding_and_stride(self):  # type: () -> None
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (5, 3, 4, 4))],
            [make_node("MaxPool", ["X"], ["Y"], auto_pad="SAME_UPPER", kernel_shape=[2, 2], strides=[2, 2])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (5, 3, 2, 2))])

    def test_maxpool_with_same_upper_padding_and_stride_and_dilation(self):  # type: () -> None
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (5, 3, 4, 4))],
            [make_node("MaxPool", ["X"], ["Y"], auto_pad="SAME_UPPER", kernel_shape=[2, 2], strides=[2, 2], dilations=[2, 3])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (5, 3, 2, 2))])

    def test_maxpool_with_same_upper_padding_and_stride_one(self):  # type: () -> None
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (5, 3, 4, 4))],
            [make_node("MaxPool", ["X"], ["Y"], auto_pad="SAME_UPPER", kernel_shape=[2, 2], strides=[1, 1])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (5, 3, 4, 4))])

    def test_maxpool_with_same_lower_padding_and_stride(self):  # type: () -> None
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (5, 3, 9, 9))],
            [make_node("MaxPool", ["X"], ["Y"], auto_pad="SAME_LOWER", kernel_shape=[2, 2], strides=[2, 2])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (5, 3, 5, 5))])

    def test_maxpool_with_same_lower_padding_and_stride_and_dilation(self):  # type: () -> None
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (5, 3, 9, 9))],
            [make_node("MaxPool", ["X"], ["Y"], auto_pad="SAME_LOWER", kernel_shape=[2, 2], strides=[2, 2], dilations=[2, 3])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (5, 3, 5, 5))])

    def test_maxpool_with_same_lower_padding_and_big_stride(self):  # type: () -> None
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (5, 3, 4, 4))],
            [make_node("MaxPool", ["X"], ["Y"], auto_pad="SAME_LOWER", kernel_shape=[2, 2], strides=[4, 4])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (5, 3, 1, 1))])

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

    def test_averagepool_ceil(self):  # type: () -> None
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (1, 1, 4, 4))],
            [make_node("AveragePool", ["X"], ["Y"], kernel_shape=[3, 3], strides=[2, 2], ceil_mode=True)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (1, 1, 2, 2))])

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

    def test_conv_transpose_with_kernel_shape(self):  # type: () -> None
        graph = self._make_graph(
            [('X', TensorProto.FLOAT, (25, 48, 16, 16)),
             ('W', TensorProto.FLOAT, (48, 32, None, None))],
            [make_node('ConvTranspose', ['X', 'W'], 'Y', kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 2, 2])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('Y', TensorProto.FLOAT, (25, 32, 30, 30))])

    def test_conv_transpose_with_dilations(self):  # type: () -> None
        graph = self._make_graph(
            [('X', TensorProto.FLOAT, (25, 48, 16, 16)),
             ('W', TensorProto.FLOAT, (48, 32, 3, 3))],
            [make_node('ConvTranspose', ['X', 'W'], 'Y', strides=[2, 2], pads=[1, 1, 2, 2], dilations=[3, 3])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('Y', TensorProto.FLOAT, (25, 32, 34, 34))])

    def test_conv_transpose_with_group(self):  # type: () -> None
        graph = self._make_graph(
            [('X', TensorProto.FLOAT, (25, 48, 16, 16)),
             ('W', TensorProto.FLOAT, (48, 32, 3, 3))],
            [make_node('ConvTranspose', ['X', 'W'], 'Y', strides=[2, 2], pads=[1, 1, 2, 2], group=2)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('Y', TensorProto.FLOAT, (25, 64, 30, 30))])

    def test_conv_transpose_with_group_and_output_shape(self):  # type: () -> None
        graph = self._make_graph(
            [('X', TensorProto.FLOAT, (25, 48, 16, 16)),
             ('W', TensorProto.FLOAT, (48, 32, 3, 3))],
            [make_node('ConvTranspose', ['X', 'W'], 'Y', strides=[2, 2], pads=[1, 1, 2, 2], group=2, output_shape=[36, 36])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('Y', TensorProto.FLOAT, (25, 64, 36, 36))])

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
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, 8)])

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
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, 9)])

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
                       num_scan_inputs=1, body=subgraph, scan_input_axes=[1])],
            []
        )

        self._assert_inferred(
            graph,
            [make_tensor_value_info('loop_state_final', TensorProto.FLOAT, (loop_state_size,)),
             make_tensor_value_info('scan_output', TensorProto.FLOAT, (seq_len, axis_0_len, input_size))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, 9)])

    def test_scan_opset9_output_axes(self):    # type: () -> None
        axis_0_len = 'axis0'
        seq_len = 'sequence'
        input_size = 2
        loop_state_size = 3

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
                       num_scan_inputs=1, body=subgraph, scan_input_axes=[1], scan_output_axes=[1])],
            []
        )

        self._assert_inferred(
            graph,
            [make_tensor_value_info('loop_state_final', TensorProto.FLOAT, (loop_state_size,)),
             make_tensor_value_info('scan_output', TensorProto.FLOAT, (axis_0_len, seq_len, input_size))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, 9)])

    def test_scan_opset9_negative_axes(self):    # type: () -> None
        axis_0_len = 'axis0'
        seq_len = 'sequence'
        input_size = 2
        loop_state_size = 3

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
                       num_scan_inputs=1, body=subgraph, scan_input_axes=[-2], scan_output_axes=[-2])],
            []
        )

        self._assert_inferred(
            graph,
            [make_tensor_value_info('loop_state_final', TensorProto.FLOAT, (loop_state_size,)),
             make_tensor_value_info('scan_output', TensorProto.FLOAT, (axis_0_len, seq_len, input_size))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, 9)])

    def test_if_ver1(self):  # type: () -> None

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

        self._assert_inferred(
            graph,
            [make_tensor_value_info('if_output', TensorProto.FLOAT, (1,))],
            opset_imports=[make_opsetid(ONNX_DOMAIN, 10)])

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

    def test_if_with_different_shapes_in_then_else_branches(self):  # type: () -> None

        # Create a simple If node where the 'then' subgraph adds to the current value, and the 'else' subgraph
        # subtracts.
        # can't use self._make_graph for the subgraphs as that add more inputs for the Reshape operations it inserts.
        # this breaks the subgraph inferencing as it expects the subgraphs to have zero inputs
        then_subgraph = helper.make_graph(
            [make_node('Add', ['current_value', 'add_value'], ['then_output'])],
            "then_subgraph",
            [],  # no inputs
            [make_tensor_value_info('then_output', TensorProto.UNDEFINED, (1,))],
        )

        else_subgraph = helper.make_graph(
            [make_node('Sub', ['current_value', 'sub_value'], ['else_output'])],
            "else_subgraph",
            [],  # no inputs
            [make_tensor_value_info('else_output', TensorProto.UNDEFINED, (5,))],
        )

        graph = self._make_graph(
            [('cond', TensorProto.BOOL, (1,)),
             ('current_value', TensorProto.FLOAT, (1,)),
             ('add_value', TensorProto.FLOAT, (1,)),
             ('sub_value', TensorProto.FLOAT, (5,))],
            [make_node('If', ['cond'], ['if_output'],
                       then_branch=then_subgraph, else_branch=else_subgraph)],
            []
        )

        self._assert_inferred(graph, [make_tensor_value_info('if_output', TensorProto.FLOAT, (None,))])  # type: ignore

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
             ('depth', TensorProto.INT64, ()),
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

    def test_loop_no_state(self):    # type: () -> None
        input_value_infos = [make_tensor_value_info('iter_num_in', TensorProto.INT64, (1,)),
                             make_tensor_value_info('cond_in', TensorProto.UNDEFINED, None)]
        output_value_infos = [make_tensor_value_info('cond_out', TensorProto.UNDEFINED, None),
                              make_tensor_value_info('output', TensorProto.FLOAT, (3,))]

        subgraph = helper.make_graph(
            [make_node('Identity', ['cond_in'], ['cond_out']),
             make_node('Identity', ['outer_scope_input'], ['output'])],
            "subgraph",
            input_value_infos,
            output_value_infos
        )

        graph = self._make_graph(
            [('max_trip_count', TensorProto.INT64, (1,)),
             ('cond_orig', TensorProto.FLOAT, (1,)),
             ('outer_scope_input', TensorProto.FLOAT, (3,))],
            [make_node('Loop', ['max_trip_count', 'cond_orig'], ['loop_output'],
                       body=subgraph)],
            []
        )

        self._assert_inferred(
            graph,
            [make_tensor_value_info('loop_output', TensorProto.FLOAT, (None, 3))])  # type: ignore

    def test_constantofshape_with_input_shape(self):  # type: () -> None
        graph = self._make_graph([],
            [make_node("Constant", [], ['shape'],
                       value=make_tensor('shape', TensorProto.INT64, (3,), (3, 4, 5))),
             make_node("ConstantOfShape", ['shape'], ['y'], value=make_tensor('value', TensorProto.INT32, (1, ), (2, )))],
            [])
        self._assert_inferred(graph,
            [make_tensor_value_info('shape', TensorProto.INT64, (3,)),
             make_tensor_value_info('y', TensorProto.INT32, (3, 4, 5))])  # type: ignore

    def test_constantofshape_without_input_shape(self):  # type: () -> None
        graph = self._make_graph([('shape', TensorProto.INT64, (3, ))],
            [make_node("ConstantOfShape", ['shape'], ['y'], value=make_tensor('value', TensorProto.UINT8, (1, ), (2, )))],
            [])
        self._assert_inferred(graph,
            [make_tensor_value_info('y', TensorProto.UINT8, (None, None, None))])  # type: ignore

    def test_constantofshape_with_shape_zero(self):  # type: () -> None
        graph = self._make_graph([],
            [make_node("Constant", [], ['shape'],
                       value=make_tensor('shape', TensorProto.INT64, (3,), (0,))),
             make_node("ConstantOfShape", ['shape'], ['y'], value=make_tensor('value', TensorProto.INT32, (1, ), (2, )))],
            [])
        self._assert_inferred(graph,
            [make_tensor_value_info('shape', TensorProto.INT64, (3,)),
             make_tensor_value_info('y', TensorProto.INT32, (0,))])  # type: ignore

    def test_convinteger(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.UINT8, (3, 4, 5, 6, 7)),
             ('y', TensorProto.UINT8, (5, 4, 2, 4, 3))],
            [make_node('ConvInteger', ['x', 'y'], 'z', pads=[0, 1, 1, 0, 0, 1], dilations=[1, 2, 2], strides=[1, 1, 2])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.INT32, (3, 5, 4, 1, 3))])

    def test_convinetger_dilations(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.UINT8, (30, 4, 8, 8, 8)),
             ('y', TensorProto.INT8, (50, 4, 3, 3, 3)),
             ('x_zero_point', TensorProto.UINT8, ()),
             ('y_zero_point', TensorProto.UINT8, ())],
            [make_node('ConvInteger', ['x', 'y', 'x_zero_point', 'y_zero_point'], 'z', dilations=[1, 2, 3])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.INT32, (30, 50, 6, 4, 2))])

    def test_convinteger_strides(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.INT8, (30, 4, 8, 8, 8)),
             ('y', TensorProto.INT8, (50, 4, 3, 3, 3)),
             ('x_zero_point', TensorProto.UINT8, ()),
             ('y_zero_point', TensorProto.UINT8, ())],
            [make_node('ConvInteger', ['x', 'y', 'x_zero_point', 'y_zero_point'], 'z', strides=[1, 2, 3])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.INT32, (30, 50, 6, 3, 2))])

    def test_convineteger_pads(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.UINT8, (30, 4, 7, 6, 4)),
             ('y', TensorProto.INT8, (50, 4, 3, 3, 3))],
            [make_node('ConvInteger', ['x', 'y'], 'z', pads=[1, 1, 2, 0, 1, 2])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.INT32, (30, 50, 6, 6, 6))])

    def test_convineteger_group(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.INT8, (30, 4, 8, 8, 8)),
             ('y', TensorProto.INT8, (4, 1, 8, 8, 8))],
            [make_node('ConvInteger', ['x', 'y'], 'z', group=4)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.INT32, (30, 4, 1, 1, 1))])

    def test_convineteger_partial_missing_shape(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.UINT8, (30, 4, None, 6, 4)),
             ('y', TensorProto.UINT8, (50, 4, 3, 3, 3)),
             ('x_zero_point', TensorProto.UINT8, ()),
             ('y_zero_point', TensorProto.UINT8, ())],
            [make_node('ConvInteger', ['x', 'y', 'x_zero_point', 'y_zero_point'], 'z', pads=[1, 1, 2, 0, 1, 2])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.INT32, (30, 50, None, 6, 6))])  # type: ignore

    def test_convineteger_partial_missing_weight_shape(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.UINT8, (30, 4, 7, 6, 4)),
             ('y', TensorProto.UINT8, (50, 4, None, 3, 3))],
            [make_node('ConvInteger', ['x', 'y'], 'z', pads=[1, 1, 2, 0, 1, 2])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.INT32, None)])

    def test_qlinearconv(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.UINT8, (3, 4, 5, 6, 7)),
             ('x_scale', TensorProto.FLOAT, ()),
             ('x_zero_point', TensorProto.UINT8, ()),
             ('w', TensorProto.UINT8, (5, 4, 2, 4, 3)),
             ('w_scale', TensorProto.FLOAT, ()),
             ('w_zero_point', TensorProto.UINT8, ()),
             ('y_scale', TensorProto.FLOAT, ()),
             ('y_zero_point', TensorProto.UINT8, ())],
            [make_node('QLinearConv', ['x', 'x_scale', 'x_zero_point', 'w', 'w_scale', 'w_zero_point', 'y_scale', 'y_zero_point'], 'y', pads=[0, 1, 1, 0, 0, 1], dilations=[1, 2, 2], strides=[1, 1, 2])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.UINT8, (3, 5, 4, 1, 3))])

    def test_qlinearconv_dilations(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.UINT8, (30, 4, 8, 8, 8)),
             ('x_scale', TensorProto.FLOAT, ()),
             ('x_zero_point', TensorProto.UINT8, ()),
             ('w', TensorProto.UINT8, (50, 4, 3, 3, 3)),
             ('w_scale', TensorProto.FLOAT, ()),
             ('w_zero_point', TensorProto.UINT8, ()),
             ('y_scale', TensorProto.FLOAT, ()),
             ('y_zero_point', TensorProto.UINT8, ())],
            [make_node('QLinearConv', ['x', 'x_scale', 'x_zero_point', 'w', 'w_scale', 'w_zero_point', 'y_scale', 'y_zero_point'], 'y', dilations=[1, 2, 3])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.UINT8, (30, 50, 6, 4, 2))])

    def test_qlinearconv_strides(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.INT8, (30, 4, 8, 8, 8)),
             ('x_scale', TensorProto.FLOAT, ()),
             ('x_zero_point', TensorProto.INT8, ()),
             ('w', TensorProto.INT8, (50, 4, 3, 3, 3)),
             ('w_scale', TensorProto.FLOAT, ()),
             ('w_zero_point', TensorProto.INT8, ()),
             ('y_scale', TensorProto.FLOAT, ()),
             ('y_zero_point', TensorProto.INT8, ())],
            [make_node('QLinearConv', ['x', 'x_scale', 'x_zero_point', 'w', 'w_scale', 'w_zero_point', 'y_scale', 'y_zero_point'], 'y', strides=[1, 2, 3])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.INT8, (30, 50, 6, 3, 2))])

    def test_qlinearconv_pads(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.UINT8, (30, 4, 7, 6, 4)),
             ('x_scale', TensorProto.FLOAT, ()),
             ('x_zero_point', TensorProto.UINT8, ()),
             ('w', TensorProto.INT8, (50, 4, 3, 3, 3)),
             ('w_scale', TensorProto.FLOAT, ()),
             ('w_zero_point', TensorProto.INT8, ()),
             ('y_scale', TensorProto.FLOAT, ()),
             ('y_zero_point', TensorProto.UINT8, ())],
            [make_node('QLinearConv', ['x', 'x_scale', 'x_zero_point', 'w', 'w_scale', 'w_zero_point', 'y_scale', 'y_zero_point'], 'y', pads=[1, 1, 2, 0, 1, 2])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.UINT8, (30, 50, 6, 6, 6))])

    def test_qlinearconv_group(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.INT8, (30, 4, 8, 8, 8)),
             ('x_scale', TensorProto.FLOAT, ()),
             ('x_zero_point', TensorProto.INT8, ()),
             ('w', TensorProto.INT8, (4, 1, 8, 8, 8)),
             ('w_scale', TensorProto.FLOAT, ()),
             ('w_zero_point', TensorProto.INT8, ()),
             ('y_scale', TensorProto.FLOAT, ()),
             ('y_zero_point', TensorProto.INT8, ())],
            [make_node('QLinearConv', ['x', 'x_scale', 'x_zero_point', 'w', 'w_scale', 'w_zero_point', 'y_scale', 'y_zero_point'], 'y', group=4)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.INT8, (30, 4, 1, 1, 1))])

    def test_qlinearconv_partial_missing_shape(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.UINT8, (30, 4, None, 6, 4)),
             ('x_scale', TensorProto.FLOAT, ()),
             ('x_zero_point', TensorProto.UINT8, ()),
             ('w', TensorProto.UINT8, (50, 4, 3, 3, 3)),
             ('w_scale', TensorProto.FLOAT, ()),
             ('w_zero_point', TensorProto.UINT8, ()),
             ('y_scale', TensorProto.FLOAT, ()),
             ('y_zero_point', TensorProto.UINT8, ())],
            [make_node('QLinearConv', ['x', 'x_scale', 'x_zero_point', 'w', 'w_scale', 'w_zero_point', 'y_scale', 'y_zero_point'], 'y', pads=[1, 1, 2, 0, 1, 2])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.UINT8, (30, 50, None, 6, 6))])  # type: ignore

    def test_qlinearconv_partial_missing_weight_shape(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.UINT8, (30, 4, 7, 6, 4)),
             ('x_scale', TensorProto.FLOAT, ()),
             ('x_zero_point', TensorProto.UINT8, ()),
             ('w', TensorProto.UINT8, (50, 4, None, 3, 3)),
             ('w_scale', TensorProto.FLOAT, ()),
             ('w_zero_point', TensorProto.UINT8, ()),
             ('y_scale', TensorProto.FLOAT, ()),
             ('y_zero_point', TensorProto.UINT8, ())],
            [make_node('QLinearConv', ['x', 'x_scale', 'x_zero_point', 'w', 'w_scale', 'w_zero_point', 'y_scale', 'y_zero_point'], 'y', pads=[1, 1, 2, 0, 1, 2])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.UINT8, None)])

    def _make_qlinearmatmul_test(self, shape1, shape2):  # type: (Sequence[int], Sequence[int]) -> None
        expected_out_shape = np.matmul(np.arange(np.product(shape1)).reshape(shape1),
                                       np.arange(np.product(shape2)).reshape(shape2)).shape
        graph = self._make_graph(
            [('a', TensorProto.UINT8, shape1),
             ('a_scale', TensorProto.FLOAT, ()),
             ('a_zero_point', TensorProto.UINT8, ()),
             ('b', TensorProto.UINT8, shape2),
             ('b_scale', TensorProto.FLOAT, ()),
             ('b_zero_point', TensorProto.UINT8, ()),
             ('y_scale', TensorProto.FLOAT, ()),
             ('y_zero_point', TensorProto.UINT8, ())],
            [make_node('QLinearMatMul', ['a', 'a_scale', 'a_zero_point', 'b', 'b_scale', 'b_zero_point', 'y_scale', 'y_zero_point'], ['y'])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.UINT8, expected_out_shape)])

    def test_qlinearmatmul(self):  # type: () -> None
        self._make_qlinearmatmul_test((3,), (3,))
        self._make_qlinearmatmul_test((4, 2), (2, 4))
        self._make_qlinearmatmul_test((2,), (2, 3))
        self._make_qlinearmatmul_test((4, 2), (2,))
        self._make_qlinearmatmul_test((5, 1, 4, 2), (1, 3, 2, 3))
        self._make_qlinearmatmul_test((4, 2), (3, 2, 3))

    def _make_qlinearmatmul_test_allow_unknown(self, shape1, shape2, expected_out_shape):  # type: (Any, Any, Any) -> None
        graph = self._make_graph(
            [('a', TensorProto.UINT8, shape1),
             ('a_scale', TensorProto.FLOAT, ()),
             ('a_zero_point', TensorProto.UINT8, ()),
             ('b', TensorProto.UINT8, shape2),
             ('b_scale', TensorProto.FLOAT, ()),
             ('b_zero_point', TensorProto.UINT8, ()),
             ('y_scale', TensorProto.FLOAT, ()),
             ('y_zero_point', TensorProto.UINT8, ())],
            [make_node('QLinearMatMul', ['a', 'a_scale', 'a_zero_point', 'b', 'b_scale', 'b_zero_point', 'y_scale', 'y_zero_point'], ['y'])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.UINT8, expected_out_shape)])

    def test_qlinearmatmul_allow_unknown(self):  # type: () -> None
        self._make_qlinearmatmul_test_allow_unknown((None,), (None,), ())
        self._make_qlinearmatmul_test_allow_unknown((3,), (None,), ())
        self._make_qlinearmatmul_test_allow_unknown((2,), (2, "a"), ("a",))
        self._make_qlinearmatmul_test_allow_unknown((4, 2), (2, "a"), (4, "a"))
        self._make_qlinearmatmul_test_allow_unknown((4, None), (2, "a"), (4, "a"))
        self._make_qlinearmatmul_test_allow_unknown((4, None), (None, "a"), (4, "a"))
        self._make_qlinearmatmul_test_allow_unknown((1, 4, 2), ("a", 2, 5), ("a", 4, 5))
        self._make_qlinearmatmul_test_allow_unknown((1, 3, 4, 2), ("a", 2, 5), (1, 3, 4, 5))
        self._make_qlinearmatmul_test_allow_unknown(None, ("a", 2, 5), None)
        self._make_qlinearmatmul_test_allow_unknown(None, None, None)

    def _make_matmulinteger_test(self, shape1, shape2):  # type: (Sequence[int], Sequence[int]) -> None
        expected_out_shape = np.matmul(np.arange(np.product(shape1)).reshape(shape1),
                                       np.arange(np.product(shape2)).reshape(shape2)).shape
        graph = self._make_graph(
            [('A', TensorProto.UINT8, shape1),
             ('B', TensorProto.UINT8, shape2),
             ('a_zero_point', TensorProto.UINT8, ()),
             ('b_zero_point', TensorProto.UINT8, ())],
            [make_node('MatMulInteger', ['A', 'B', 'a_zero_point', 'b_zero_point'], ['Y'])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('Y', TensorProto.INT32, expected_out_shape)])

    def test_matmulinteger(self):  # type: () -> None
        self._make_matmulinteger_test((2,), (2,))
        self._make_matmulinteger_test((1, 2), (2, 3))
        self._make_matmulinteger_test((2,), (2, 3))
        self._make_matmulinteger_test((4, 2), (2,))
        self._make_matmulinteger_test((5, 1, 4, 2), (1, 3, 2, 3))
        self._make_matmulinteger_test((4, 2), (3, 2, 3))

    def test_quantizelinear(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (30, 4, 5)),
             ('y_scale', TensorProto.FLOAT, ()),
             ('y_zero_point', TensorProto.UINT8, ())],
            [make_node('QuantizeLinear', ['x', 'y_scale', 'y_zero_point'], ['y'])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.UINT8, (30, 4, 5))])

    def test_dequantizelinear(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.UINT8, (30, 4, 5)),
             ('x_scale', TensorProto.FLOAT, ()),
             ('x_zero_point', TensorProto.UINT8, ())],
            [make_node('DequantizeLinear', ['x', 'x_scale', 'x_zero_point'], ['y'])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (30, 4, 5))])

    def test_reversesequence(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (4, 5, 6)),
             ('sequence_lens', TensorProto.INT64, (5,))],
            [make_node('ReverseSequence', ['x', 'sequence_lens'], ['y'])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (4, 5, 6))])

    def test_unique_without_axis(self):  # type: () -> None
        graph = self._make_graph(
            [('X', TensorProto.FLOAT, (2, 4, 2))],
            [make_node('Unique', ['X'], ['Y', 'indices', 'inverse_indices', 'counts'])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('Y', TensorProto.FLOAT, (None,)),  # type: ignore
                                      make_tensor_value_info('indices', TensorProto.INT64, (None,)),  # type: ignore
                                      make_tensor_value_info('inverse_indices', TensorProto.INT64, (None,)),  # type: ignore
                                      make_tensor_value_info('counts', TensorProto.INT64, (None,))])  # type: ignore

    def test_unique_with_axis(self):  # type: () -> None
        graph = self._make_graph(
            [('X', TensorProto.FLOAT, (2, 4, 2))],
            [make_node('Unique', ['X'], ['Y', 'indices', 'inverse_indices', 'counts'], axis=1)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('Y', TensorProto.FLOAT, (2, None, 2)),  # type: ignore
                                      make_tensor_value_info('indices', TensorProto.INT64, (None,)),  # type: ignore
                                      make_tensor_value_info('inverse_indices', TensorProto.INT64, (None,)),  # type: ignore
                                      make_tensor_value_info('counts', TensorProto.INT64, (None,))])  # type: ignore

    def test_det(self):  # type: () -> None
        graph = self._make_graph(
            [('X', TensorProto.FLOAT, (3, 3))],
            [make_node('Det', ['X'], ['Y'])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('Y', TensorProto.FLOAT, ())])

        graph = self._make_graph(
            [('X', TensorProto.FLOAT, (4, 5, 6, 7, 7))],
            [make_node('Det', ['X'], ['Y'])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('Y', TensorProto.FLOAT, (4, 5, 6))])

    def test_tile(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (4, 5, 6)),
             ('repeats', TensorProto.INT64, (3,))],
            [make_node('Tile', ['x', 'repeats'], ['y'])],
            [],
            initializer=[make_tensor('repeats', TensorProto.INT64, (3,), (1, 2, 3))])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (4, 10, 18))])

    def test_tile_raw_input_data(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (4, 5, 6)),
             ('repeats', TensorProto.INT64, (3,))],
            [make_node('Tile', ['x', 'repeats'], ['y'])],
            [],
            initializer=[make_tensor('repeats', TensorProto.INT64, (3,),
                                     vals=np.array([1, 2, 3], dtype='<i8').tobytes(), raw=True)])  # Feed raw bytes (force little endian ordering like onnx standard) for test purpose
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (4, 10, 18))])

    def test_tile_rank_inference(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (4, 5, 6)),
             ('repeats', TensorProto.INT64, (3,))],
            [make_node('Tile', ['x', 'repeats'], ['y'])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (None, None, None))])  # type: ignore

    def test_linearclassifier_1D_input(self):  # type: () -> None
        if ONNX_ML:
            graph = self._make_graph(
                [('x', TensorProto.FLOAT, (5,))],
                [make_node('LinearClassifier', ['x'], ['y', 'z'], domain=ONNX_ML_DOMAIN, coefficients=[0.0008, -0.0008], intercepts=[2.0, 2.0], classlabels_ints=[1, 2])],
                [])
            self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.INT64, (1,)),
                                          make_tensor_value_info('z', TensorProto.FLOAT, (1, 2))],
                                          opset_imports=[make_opsetid(ONNX_ML_DOMAIN, 1), make_opsetid(ONNX_DOMAIN, 11)])

    def test_linearclassifier_2D_input(self):  # type: () -> None
        if ONNX_ML:
            graph = self._make_graph(
                [('x', TensorProto.FLOAT, (4, 5))],
                [make_node('LinearClassifier', ['x'], ['y', 'z'], domain=ONNX_ML_DOMAIN, coefficients=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], intercepts=[2.0, 2.0, 3.0], classlabels_ints=[1, 2, 3])],
                [])
            self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.INT64, (4,)),
                                          make_tensor_value_info('z', TensorProto.FLOAT, (4, 3))],
                                          opset_imports=[make_opsetid(ONNX_ML_DOMAIN, 1), make_opsetid(ONNX_DOMAIN, 11)])

    def test_roialign_symbolic(self):   # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, ('N', 'C', 'H', 'W')),
             ('rois', TensorProto.FLOAT, ('num_rois', 4)),
             ('batch_indices', TensorProto.INT64, ('num_rois',))],
            [make_node('RoiAlign', ['x', 'rois', 'batch_indices'], ['y'], output_height=10, output_width=5)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, ('num_rois', 'C', 10, 5))])  # type: ignore

    def test_roialign_symbolic_defaults(self):   # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, ('N', 'C', 'H', 'W')),
             ('rois', TensorProto.FLOAT, ('num_rois', 4)),
             ('batch_indices', TensorProto.INT64, ('num_rois',))],
            [make_node('RoiAlign', ['x', 'rois', 'batch_indices'], ['y'])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, ('num_rois', 'C', 1, 1))])  # type: ignore

    def test_roialign_num_rois(self):   # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, ('N', 'C', 'H', 'W')),
             ('rois', TensorProto.FLOAT, ('num_rois', 4)),
             ('batch_indices', TensorProto.INT64, (15,))],
            [make_node('RoiAlign', ['x', 'rois', 'batch_indices'], ['y'])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (15, 'C', 1, 1))])  # type: ignore

    def test_label_encoder_string_int64(self):  # type: () -> None
        if ONNX_ML:
            string_list = ['A', 'm', 'y']
            float_list = [94.17, 36.00]
            int64_list = [12, 28, 86]
            graph = self._make_graph(
                [('x', TensorProto.STRING, (6, 1))],
                [make_node('LabelEncoder', ['x'], ['y'], domain=ONNX_ML_DOMAIN,
                           keys_strings=string_list, values_int64s=int64_list)], [])
            self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.INT64, (6, 1))],
                                          opset_imports=[make_opsetid(ONNX_ML_DOMAIN, 2), make_opsetid(ONNX_DOMAIN, 11)])

            graph = self._make_graph(
                [('x', TensorProto.INT64, (2, 3))],
                [make_node('LabelEncoder', ['x'], ['y'], domain=ONNX_ML_DOMAIN,
                           keys_int64s=int64_list, values_strings=string_list)], [])
            self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.STRING, (2, 3))],
                                  opset_imports=[make_opsetid(ONNX_ML_DOMAIN, 2), make_opsetid(ONNX_DOMAIN, 11)])

            graph = self._make_graph(
                [('x', TensorProto.FLOAT, (2,))],
                [make_node('LabelEncoder', ['x'], ['y'], domain=ONNX_ML_DOMAIN,
                           keys_floats=float_list, values_int64s=int64_list)], [])
            self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.INT64, (2,))],
                                  opset_imports=[make_opsetid(ONNX_ML_DOMAIN, 2), make_opsetid(ONNX_DOMAIN, 11)])

            graph = self._make_graph(
                [('x', TensorProto.INT64, (8,))],
                [make_node('LabelEncoder', ['x'], ['y'], domain=ONNX_ML_DOMAIN,
                           keys_int64s=int64_list, values_floats=float_list)], [])
            self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (8,))],
                                  opset_imports=[make_opsetid(ONNX_ML_DOMAIN, 2), make_opsetid(ONNX_DOMAIN, 11)])

            graph = self._make_graph(
                [('x', TensorProto.FLOAT, ())],
                [make_node('LabelEncoder', ['x'], ['y'], domain=ONNX_ML_DOMAIN,
                           keys_floats=float_list, values_strings=string_list)], [])
            self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.STRING, ())],
                                  opset_imports=[make_opsetid(ONNX_ML_DOMAIN, 2), make_opsetid(ONNX_DOMAIN, 11)])

            graph = self._make_graph(
                [('x', TensorProto.STRING, (1, 2))],
                [make_node('LabelEncoder', ['x'], ['y'], domain=ONNX_ML_DOMAIN,
                           keys_strings=string_list, values_floats=float_list)], [])
            self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (1, 2))],
                                  opset_imports=[make_opsetid(ONNX_ML_DOMAIN, 2), make_opsetid(ONNX_DOMAIN, 11)])

    def make_sparse(self,
                    shape,  # type: Sequence[int]
                    values,  # type: Sequence[int]
                    indices_shape,  # type: Sequence[int]
                    indices  # type: Sequence[int]
                    ):  # type: (...) -> SparseTensorProto
        sparse = SparseTensorProto()
        sparse.dims.extend(shape)
        nnz = len(values)
        sparse.values.CopyFrom(helper.make_tensor('spval', TensorProto.INT64, (nnz,), values))
        sparse.indices.CopyFrom(helper.make_tensor('spind', TensorProto.INT64, indices_shape, indices))
        return sparse

    def test_constant_sparse(self):  # type: () -> None
        y_shape = [100]
        y_value = self.make_sparse(y_shape, [13, 17, 19], [3], [9, 27, 81])
        graph = self._make_graph(
            [],
            [make_node('Constant', [], ['y'], sparse_value=y_value)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.INT64, y_shape)])  # type: ignore

    def test_constant_value_int(self):  # type: () -> None
        graph = self._make_graph(
            [],
            [make_node('Constant', [], ['y'], value_int=42)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.INT64, [])])

    def test_constant_value_ints(self):  # type: () -> None
        value_ints = [1, 2, 3]
        graph = self._make_graph(
            [],
            [make_node('Constant', [], ['y'], value_ints=value_ints)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.INT64, [len(value_ints)])])

    def test_constant_value_float(self):  # type: () -> None
        graph = self._make_graph(
            [],
            [make_node('Constant', [], ['y'], value_float=1.42)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, [])])

    def test_constant_value_floats(self):  # type: () -> None
        value_floats = [1.0, 1.1, 1.2]
        graph = self._make_graph(
            [],
            [make_node('Constant', [], ['y'], value_floats=value_floats)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, [len(value_floats)])])

    def test_constant_value_string(self):  # type: () -> None
        graph = self._make_graph(
            [],
            [make_node('Constant', [], ['y'], value_string="String value")],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.STRING, [])])

    def test_constant_value_strings(self):  # type: () -> None
        value_strings = ["o", "n", "n", "x"]
        graph = self._make_graph(
            [],
            [make_node('Constant', [], ['y'], value_strings=value_strings)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.STRING, [len(value_strings)])])

    def test_range(self):  # type: () -> None
        graph = self._make_graph(
            [('start', TensorProto.FLOAT, ()),
             ('limit', TensorProto.FLOAT, ()),
             ('delta', TensorProto.FLOAT, ())],
            [make_node('Range', ['start', 'limit', 'delta'], ['output'])],
            [],
            initializer=[make_tensor('start', TensorProto.FLOAT, (), (1,)),
                         make_tensor('limit', TensorProto.FLOAT, (), (5,)),
                         make_tensor('delta', TensorProto.FLOAT, (), (2,))])
        self._assert_inferred(graph, [make_tensor_value_info('output', TensorProto.FLOAT, (2,))])

    def test_range_rank_inference(self):  # type: () -> None
        graph = self._make_graph(
            [('start', TensorProto.INT32, ()),
             ('limit', TensorProto.INT32, ()),
             ('delta', TensorProto.INT32, ())],
            [make_node('Range', ['start', 'limit', 'delta'], ['output'])],
            [],
            initializer=[make_tensor('start', TensorProto.INT32, (), (1,)),
                         make_tensor('limit', TensorProto.INT32, (), (5,))])  # Missing 'delta' initializer
        self._assert_inferred(graph, [make_tensor_value_info('output', TensorProto.INT32, (None,))])  # type: ignore

    def test_gathernd(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (4, 5, 6)),
             ('indices', TensorProto.INT64, (2,))],
            [make_node('GatherND', ['x', 'indices'], ['y'])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (6,))])

    def test_gathernd_batchdim_1(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (2, 2, 2)),
             ('indices', TensorProto.INT64, (2, 1))],
            [make_node('GatherND', ['x', 'indices'], ['y'], batch_dims=1)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (2, 2))])

    def test_cumsum(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (2, 3)),
             ('axis', TensorProto.FLOAT, (1,))],
            [make_node('CumSum', ['x', 'axis'], 'z')],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, (2, 3))])

    def test_nonmaxsuppression(self):  # type: () -> None
        graph = self._make_graph(
            [('boxes', TensorProto.FLOAT, (1, 3, 4)),
             ('scores', TensorProto.FLOAT, (1, 5, 3))],
            [make_node('NonMaxSuppression', ['boxes', 'scores'], ['y'])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.INT64, (None, 3))])  # type: ignore

    def test_sequence_empty(self):  # type: () -> None
        graph = self._make_graph(
            [],
            [make_node('SequenceEmpty', [], ['output'])],
            [])
        self._assert_inferred(graph, [make_sequence_value_info('output', TensorProto.FLOAT, None)])  # type: ignore

    def test_sequence_construct(self):  # type: () -> None
        graph = self._make_graph(
            [('input1', TensorProto.FLOAT, (2, 3, 4)),
             ('input2', TensorProto.FLOAT, (2, 3, 4)),
             ('input3', TensorProto.FLOAT, (2, 3, 4))],
            [make_node('SequenceConstruct', ['input1', 'input2', 'input3'], ['output_sequence'])],
            [])
        self._assert_inferred(graph,
            [make_sequence_value_info('output_sequence', TensorProto.FLOAT, (2, 3, 4))])  # type: ignore

    def test_sequence_construct_one_input(self):  # type: () -> None
        graph = self._make_graph(
            [('input1', TensorProto.FLOAT, (2, 3, 4))],
            [make_node('SequenceConstruct', ['input1'], ['output_sequence'])],
            [])
        self._assert_inferred(graph,
            [make_sequence_value_info('output_sequence', TensorProto.FLOAT, (2, 3, 4))])  # type: ignore

    def test_sequence_construct_diff_rank(self):  # type: () -> None
        graph = self._make_graph(
            [('input1', TensorProto.FLOAT, (2, 3, 4)),
             ('input2', TensorProto.FLOAT, (2, 3)),
             ('input3', TensorProto.FLOAT, (2, 3))],
            [make_node('SequenceConstruct', ['input1', 'input2', 'input3'], ['output_sequence'])],
            [])
        self._assert_inferred(graph,
            [make_sequence_value_info('output_sequence', TensorProto.FLOAT, None)])  # type: ignore

    def test_sequence_construct_diff_dim_size(self):  # type: () -> None
        graph = self._make_graph(
            [('input1', TensorProto.FLOAT, (2, 3, 4)),
             ('input2', TensorProto.FLOAT, (2, 3, 5)),
             ('input3', TensorProto.FLOAT, (2, 3, 6))],
            [make_node('SequenceConstruct', ['input1', 'input2', 'input3'], ['output_sequence'])],
            [])
        self._assert_inferred(graph,
            [make_sequence_value_info('output_sequence', TensorProto.FLOAT, (2, 3, None))])  # type: ignore

    def test_sequence_insert(self):  # type: () -> None
        graph = self._make_graph(
            [('input1', TensorProto.FLOAT, (2, 3, 4)),
             ('input2', TensorProto.FLOAT, (2, 3, 4)),
             ('input3', TensorProto.FLOAT, (2, 3, 4)),
             ('input4', TensorProto.FLOAT, (2, 3, 4))],
            [make_node('SequenceConstruct', ['input1', 'input2', 'input3'], ['in_sequence']),
             make_node('SequenceInsert', ['in_sequence', 'input4'], ['output_sequence'])],
            [])
        self._assert_inferred(
            graph,
            [make_sequence_value_info('in_sequence', TensorProto.FLOAT, (2, 3, 4)),
             make_sequence_value_info('output_sequence', TensorProto.FLOAT, (2, 3, 4))])  # type: ignore

    def test_sequence_insert_diff_rank(self):  # type: () -> None
        graph = self._make_graph(
            [('input1', TensorProto.FLOAT, (2, 3, 4)),
             ('input2', TensorProto.FLOAT, (2, 3, 4)),
             ('input3', TensorProto.FLOAT, (2, 3, 4)),
             ('input4', TensorProto.FLOAT, (2, 3))],
            [make_node('SequenceConstruct', ['input1', 'input2', 'input3'], ['in_sequence']),
             make_node('SequenceInsert', ['in_sequence', 'input4'], ['output_sequence'])],
            [])
        self._assert_inferred(
            graph,
            [make_sequence_value_info('in_sequence', TensorProto.FLOAT, (2, 3, 4)),
             make_sequence_value_info('output_sequence', TensorProto.FLOAT, None)])  # type: ignore

    def test_sequence_insert_diff_shape(self):  # type: () -> None
        graph = self._make_graph(
            [('input1', TensorProto.FLOAT, (2, 3, 4)),
             ('input2', TensorProto.FLOAT, (2, 3, 4)),
             ('input3', TensorProto.FLOAT, (2, 5, 4)),
             ('input4', TensorProto.FLOAT, (2, 5, 2))],
            [make_node('SequenceConstruct', ['input1', 'input2', 'input3'], ['in_sequence']),
             make_node('SequenceInsert', ['in_sequence', 'input4'], ['output_sequence'])],
            [])
        self._assert_inferred(
            graph,
            [make_sequence_value_info('in_sequence', TensorProto.FLOAT, (2, None, 4)),  # type: ignore
             make_sequence_value_info('output_sequence', TensorProto.FLOAT, (2, None, None))])  # type: ignore

    def test_sequence_at(self):  # type: () -> None
        graph = self._make_graph(
            [('input1', TensorProto.FLOAT, (2, 3, 4)),
             ('input2', TensorProto.FLOAT, (2, 3, 4)),
             ('input3', TensorProto.FLOAT, (2, 3, 4)),
             ('ind', TensorProto.INT64, ())],
            [make_node('SequenceConstruct', ['input1', 'input2', 'input3'], ['in_sequence']),
             make_node('SequenceAt', ['in_sequence', 'ind'], ['output'])],
            [])
        self._assert_inferred(
            graph,
            [make_sequence_value_info('in_sequence', TensorProto.FLOAT, (2, 3, 4)),
             make_tensor_value_info('output', TensorProto.FLOAT, (2, 3, 4))])  # type: ignore

    def test_sequence_at_unknown_shape(self):  # type: () -> None
        graph = self._make_graph(
            [('input1', TensorProto.FLOAT, (2, 3, 4)),
             ('input2', TensorProto.FLOAT, (2, 3)),
             ('input3', TensorProto.FLOAT, (2, 3, 4)),
             ('ind', TensorProto.INT64, ())],
            [make_node('SequenceConstruct', ['input1', 'input2', 'input3'], ['in_sequence']),
             make_node('SequenceAt', ['in_sequence', 'ind'], ['output'])],
            [])
        self._assert_inferred(
            graph,
            [make_sequence_value_info('in_sequence', TensorProto.FLOAT, None),
             make_tensor_value_info('output', TensorProto.FLOAT, None)])  # type: ignore

    def test_sequence_at_unknown_dim_size(self):  # type: () -> None
        graph = self._make_graph(
            [('input1', TensorProto.FLOAT, (2, 3, 4)),
             ('input2', TensorProto.FLOAT, (2, 3, 5)),
             ('input3', TensorProto.FLOAT, (2, 3, 4)),
             ('ind', TensorProto.INT64, ())],
            [make_node('SequenceConstruct', ['input1', 'input2', 'input3'], ['in_sequence']),
             make_node('SequenceAt', ['in_sequence', 'ind'], ['output'])],
            [])
        self._assert_inferred(
            graph,
            [make_sequence_value_info('in_sequence', TensorProto.FLOAT, (2, 3, None)),  # type: ignore
             make_tensor_value_info('output', TensorProto.FLOAT, (2, 3, None))])  # type: ignore

    def test_sequence_erase(self):  # type: () -> None
        graph = self._make_graph(
            [('input1', TensorProto.FLOAT, (2, 3, 4)),
             ('input2', TensorProto.FLOAT, (2, 3, 4)),
             ('input3', TensorProto.FLOAT, (2, 3, 4)),
             ('ind', TensorProto.INT64, ())],
            [make_node('SequenceConstruct', ['input1', 'input2', 'input3'], ['in_sequence']),
             make_node('SequenceErase', ['in_sequence', 'ind'], ['output_sequence'])],
            [])
        self._assert_inferred(
            graph,
            [make_sequence_value_info('in_sequence', TensorProto.FLOAT, (2, 3, 4)),
             make_sequence_value_info('output_sequence', TensorProto.FLOAT, (2, 3, 4))])  # type: ignore

    def test_sequence_erase_diff_dim_size(self):  # type: () -> None
        graph = self._make_graph(
            [('input1', TensorProto.FLOAT, (2, 3, 'x')),
             ('input2', TensorProto.FLOAT, (2, 3, 'x')),
             ('input3', TensorProto.FLOAT, (2, 5, 'x')),
             ('ind', TensorProto.INT64, ())],
            [make_node('SequenceConstruct', ['input1', 'input2', 'input3'], ['in_sequence']),
             make_node('SequenceErase', ['in_sequence', 'ind'], ['output_sequence'])],
            [])
        self._assert_inferred(
            graph,
            [make_sequence_value_info('in_sequence', TensorProto.FLOAT, (2, None, 'x')),  # type: ignore
             make_sequence_value_info('output_sequence', TensorProto.FLOAT, (2, None, 'x'))])  # type: ignore

    def test_sequence_length(self):  # type: () -> None
        graph = self._make_graph(
            [('input1', TensorProto.FLOAT, (2, 3, 'x')),
             ('input2', TensorProto.FLOAT, (2, 3, 'x')),
             ('input3', TensorProto.FLOAT, (2, 3, 'x'))],
            [make_node('SequenceConstruct', ['input1', 'input2', 'input3'], ['in_sequence']),
             make_node('SequenceLength', ['in_sequence'], ['len'])],
            [])
        self._assert_inferred(
            graph,
            [make_sequence_value_info('in_sequence', TensorProto.FLOAT, (2, 3, 'x')),
             make_tensor_value_info('len', TensorProto.INT64, ())])  # type: ignore

    def test_split_to_sequence(self):  # type: () -> None
        graph = self._make_graph(
            [('input', TensorProto.FLOAT, (6, 4)),
             ('split', TensorProto.INT32, (2,))],
            [make_node('SplitToSequence', ['input', 'split'], ['output_sequence'])],
            [],
            initializer=[make_tensor('split', TensorProto.INT32, (), (3, 3))])
        self._assert_inferred(graph,
            [make_sequence_value_info('output_sequence', TensorProto.FLOAT, (3, 4))])  # type: ignore

    def test_split_to_sequence_scalar(self):  # type: () -> None
        graph = self._make_graph(
            [('input', TensorProto.FLOAT, (6, 4)),
             ('split', TensorProto.INT32, ())],
            [make_node('SplitToSequence', ['input', 'split'], ['output_sequence'])],
            [],
            initializer=[make_tensor('split', TensorProto.INT32, (), (2, ))])
        self._assert_inferred(graph,
            [make_sequence_value_info('output_sequence', TensorProto.FLOAT, (2, 4))])  # type: ignore

    def test_split_to_sequence_keepdims(self):  # type: () -> None
        graph = self._make_graph(
            [('input', TensorProto.FLOAT, (6, 4))],
            [make_node('SplitToSequence', ['input'], ['output_sequence'], keepdims=1)],
            [])
        self._assert_inferred(graph,
            [make_sequence_value_info('output_sequence', TensorProto.FLOAT, (1, 4))])  # type: ignore

    def test_split_to_sequence_not_keepdims(self):  # type: () -> None
        graph = self._make_graph(
            [('input', TensorProto.FLOAT, (6, 4))],
            [make_node('SplitToSequence', ['input'], ['output_sequence'], keepdims=0)],
            [])
        self._assert_inferred(graph,
            [make_sequence_value_info('output_sequence', TensorProto.FLOAT, (4, ))])  # type: ignore

    def test_split_to_sequence_ignore_keepdims(self):  # type: () -> None
        graph = self._make_graph(
            [('input', TensorProto.FLOAT, (6, 4)),
             ('split', TensorProto.INT32, (2,))],
            [make_node('SplitToSequence', ['input', 'split'], ['output_sequence'], keepdims=0)],
            [],
            initializer=[make_tensor('split', TensorProto.INT32, (), (3, 3))])
        self._assert_inferred(graph,
            [make_sequence_value_info('output_sequence', TensorProto.FLOAT, (3, 4))])  # type: ignore

    def test_split_to_sequence_axis(self):  # type: () -> None
        graph = self._make_graph(
            [('input', TensorProto.FLOAT, (6, 4))],
            [make_node('SplitToSequence', ['input'], ['output_sequence'], axis=1)],
            [])
        self._assert_inferred(graph,
            [make_sequence_value_info('output_sequence', TensorProto.FLOAT, (6, 1))])  # type: ignore

    def test_split_to_sequence_neg_axis(self):  # type: () -> None
        graph = self._make_graph(
            [('input', TensorProto.FLOAT, (6, 4))],
            [make_node('SplitToSequence', ['input'], ['output_sequence'], axis=-2)],
            [])
        self._assert_inferred(graph,
            [make_sequence_value_info('output_sequence', TensorProto.FLOAT, (1, 4))])  # type: ignore

    def test_split_to_sequence_split_sizes(self):  # type: () -> None
        graph = self._make_graph(
            [('input', TensorProto.FLOAT, (6, 4)),
             ('split', TensorProto.INT32, (3,))],
            [make_node('SplitToSequence', ['input', 'split'], ['output_sequence'])],
            [],
            initializer=[make_tensor('split', TensorProto.INT32, (), (2, 1, 3))])
        self._assert_inferred(graph,
            [make_sequence_value_info('output_sequence', TensorProto.FLOAT, (None, 4))])  # type: ignore

    def test_split_to_sequence_non_divisible(self):  # type: () -> None
        graph = self._make_graph(
            [('input', TensorProto.FLOAT, (6, 4)),
             ('split', TensorProto.INT32, ())],
            [make_node('SplitToSequence', ['input', 'split'], ['output_sequence'])],
            [],
            initializer=[make_tensor('split', TensorProto.INT32, (), (4, ))])
        self._assert_inferred(graph,
            [make_sequence_value_info('output_sequence', TensorProto.FLOAT, (None, 4))])  # type: ignore

    def test_concat_from_sequence(self):  # type: () -> None
        graph = self._make_graph(
            [('input1', TensorProto.FLOAT, (2, 3, 'x')),
             ('input2', TensorProto.FLOAT, (2, 3, 'x')),
             ('input3', TensorProto.FLOAT, (2, 3, 'x'))],
            [make_node('SequenceConstruct', ['input1', 'input2', 'input3'], ['in_sequence']),
             make_node('ConcatFromSequence', ['in_sequence'], ['out'], axis=0)],
            [])
        self._assert_inferred(
            graph,
            [make_sequence_value_info('in_sequence', TensorProto.FLOAT, (2, 3, 'x')),
             make_tensor_value_info('out', TensorProto.FLOAT, (None, 3, 'x'))])  # type: ignore

    def test_concat_from_sequence_unknown_shape(self):  # type: () -> None
        graph = self._make_graph(
            [('input1', TensorProto.FLOAT, (2, 3, 'x')),
             ('input2', TensorProto.FLOAT, (2, 3)),
             ('input3', TensorProto.FLOAT, (2, 3, 'x'))],
            [make_node('SequenceConstruct', ['input1', 'input2', 'input3'], ['in_sequence']),
             make_node('ConcatFromSequence', ['in_sequence'], ['out'], axis=0)],
            [])
        self._assert_inferred(
            graph,
            [make_sequence_value_info('in_sequence', TensorProto.FLOAT, None),
             make_tensor_value_info('out', TensorProto.FLOAT, None)])  # type: ignore

    def test_concat_from_sequence_unknown_dim_size(self):  # type: () -> None
        graph = self._make_graph(
            [('input1', TensorProto.FLOAT, (2, 3, 'x')),
             ('input2', TensorProto.FLOAT, (2, 4, 'x')),
             ('input3', TensorProto.FLOAT, (2, 3, 'x'))],
            [make_node('SequenceConstruct', ['input1', 'input2', 'input3'], ['in_sequence']),
             make_node('ConcatFromSequence', ['in_sequence'], ['out'], axis=0)],
            [])
        self._assert_inferred(
            graph,
            [make_sequence_value_info('in_sequence', TensorProto.FLOAT, (2, None, 'x')),  # type: ignore
             make_tensor_value_info('out', TensorProto.FLOAT, (None, None, 'x'))])  # type: ignore

    def test_concat_from_sequence_axis(self):  # type: () -> None
        graph = self._make_graph(
            [('input1', TensorProto.FLOAT, (2, 3, 'x')),
             ('input2', TensorProto.FLOAT, (2, 4, 'x')),
             ('input3', TensorProto.FLOAT, (2, 3, 'x'))],
            [make_node('SequenceConstruct', ['input1', 'input2', 'input3'], ['in_sequence']),
             make_node('ConcatFromSequence', ['in_sequence'], ['out'], axis=2)],
            [])
        self._assert_inferred(
            graph,
            [make_sequence_value_info('in_sequence', TensorProto.FLOAT, (2, None, 'x')),  # type: ignore
             make_tensor_value_info('out', TensorProto.FLOAT, (2, None, None))])  # type: ignore

    def test_concat_from_sequence_neg_axis(self):  # type: () -> None
        graph = self._make_graph(
            [('input1', TensorProto.FLOAT, (2, 3, 'x')),
             ('input2', TensorProto.FLOAT, (2, 4, 'x')),
             ('input3', TensorProto.FLOAT, (2, 3, 'x'))],
            [make_node('SequenceConstruct', ['input1', 'input2', 'input3'], ['in_sequence']),
             make_node('ConcatFromSequence', ['in_sequence'], ['out'], axis=-3)],
            [])
        self._assert_inferred(
            graph,
            [make_sequence_value_info('in_sequence', TensorProto.FLOAT, (2, None, 'x')),  # type: ignore
             make_tensor_value_info('out', TensorProto.FLOAT, (None, None, 'x'))])  # type: ignore

    def test_concat_from_sequence_new_axis(self):  # type: () -> None
        graph = self._make_graph(
            [('input1', TensorProto.FLOAT, (2, 3, 'x')),
             ('input2', TensorProto.FLOAT, (2, 3, 'x')),
             ('input3', TensorProto.FLOAT, (2, 3, 'x'))],
            [make_node('SequenceConstruct', ['input1', 'input2', 'input3'], ['in_sequence']),
             make_node('ConcatFromSequence', ['in_sequence'], ['out'], axis=2, new_axis=1)],
            [])
        self._assert_inferred(
            graph,
            [make_sequence_value_info('in_sequence', TensorProto.FLOAT, (2, 3, 'x')),
             make_tensor_value_info('out', TensorProto.FLOAT, (2, 3, None, 'x'))])  # type: ignore

    def test_concat_from_sequence_neg_new_axis(self):  # type: () -> None
        graph = self._make_graph(
            [('input1', TensorProto.FLOAT, (2, 3, 'x')),
             ('input2', TensorProto.FLOAT, (2, 3, 'x')),
             ('input3', TensorProto.FLOAT, (2, 3, 'x'))],
            [make_node('SequenceConstruct', ['input1', 'input2', 'input3'], ['in_sequence']),
             make_node('ConcatFromSequence', ['in_sequence'], ['out'], axis=-1, new_axis=1)],
            [])
        self._assert_inferred(
            graph,
            [make_sequence_value_info('in_sequence', TensorProto.FLOAT, (2, 3, 'x')),
             make_tensor_value_info('out', TensorProto.FLOAT, (2, 3, 'x', None))])  # type: ignore

    def test_adagrad(self):  # type: () -> None
        graph = self._make_graph(
            [('R', TensorProto.FLOAT, ()),  # scalar's shape is ()
             ('T', TensorProto.INT64, ()),  # scalar's shape is ()
             ('X', TensorProto.FLOAT, (1, 2)),
             ('G', TensorProto.FLOAT, (1, 2)),
             ('H', TensorProto.FLOAT, (1, 2))],
            [make_node('Adagrad', ['R', 'T', 'X', 'G', 'H'], ['X_new', 'H_new'],
                       domain=AI_ONNX_PREVIEW_TRAINING_DOMAIN)],
            [])

        self._assert_inferred(
            graph,
            [make_tensor_value_info('X_new', TensorProto.FLOAT, (1, 2)),
             make_tensor_value_info('H_new', TensorProto.FLOAT, (1, 2))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, 12), helper.make_opsetid(AI_ONNX_PREVIEW_TRAINING_DOMAIN, 1)])

    def test_adagrad_multiple(self):  # type: () -> None
        graph = self._make_graph(
            [('R', TensorProto.FLOAT, ()),  # scalar's shape is ()
             ('T', TensorProto.INT64, ()),  # scalar's shape is ()
             ('X1', TensorProto.FLOAT, (1, 2)),
             ('X2', TensorProto.FLOAT, (3, 4)),
             ('G1', TensorProto.FLOAT, (1, 2)),
             ('G2', TensorProto.FLOAT, (3, 4)),
             ('H1', TensorProto.FLOAT, (1, 2)),
             ('H2', TensorProto.FLOAT, (3, 4))],
            [make_node('Adagrad', ['R', 'T', 'X1', 'X2', 'G1', 'G2', 'H1', 'H2'],
                       ['X1_new', 'X2_new', 'H1_new', 'H2_new'],
                       domain=AI_ONNX_PREVIEW_TRAINING_DOMAIN)],
            [])

        self._assert_inferred(graph,
            [make_tensor_value_info('X1_new', TensorProto.FLOAT, (1, 2)),
             make_tensor_value_info('X2_new', TensorProto.FLOAT, (3, 4)),
             make_tensor_value_info('H1_new', TensorProto.FLOAT, (1, 2)),
             make_tensor_value_info('H2_new', TensorProto.FLOAT, (3, 4))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, 12), helper.make_opsetid(AI_ONNX_PREVIEW_TRAINING_DOMAIN, 1)])

    def test_momentum(self):  # type: () -> None
        graph = self._make_graph(
            [('R', TensorProto.FLOAT, ()),  # scalar's shape is ()
             ('T', TensorProto.INT64, ()),  # scalar's shape is ()
             ('X', TensorProto.FLOAT, (1, 2)),
             ('G', TensorProto.FLOAT, (1, 2)),
             ('V', TensorProto.FLOAT, (1, 2))],
            [make_node('Momentum', ['R', 'T', 'X', 'G', 'V'], ['X_new', 'V_new'],
             alpha=0.9, beta=1.0, norm_coefficient=0.02, mode='standard',
             domain=AI_ONNX_PREVIEW_TRAINING_DOMAIN)],
            [])
        self._assert_inferred(
            graph,
            [make_tensor_value_info('X_new', TensorProto.FLOAT, (1, 2)),
             make_tensor_value_info('V_new', TensorProto.FLOAT, (1, 2))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, 12), helper.make_opsetid(AI_ONNX_PREVIEW_TRAINING_DOMAIN, 1)])

    def test_momentum_multiple(self):  # type: () -> None
        graph = self._make_graph(
            [('R', TensorProto.FLOAT, ()),  # scalar's shape is ()
             ('T', TensorProto.INT64, ()),  # scalar's shape is ()
             ('X1', TensorProto.FLOAT, (1, 2)),
             ('X2', TensorProto.FLOAT, (3, 4)),
             ('G1', TensorProto.FLOAT, (1, 2)),
             ('G2', TensorProto.FLOAT, (3, 4)),
             ('V1', TensorProto.FLOAT, (1, 2)),
             ('V2', TensorProto.FLOAT, (3, 4))],
            [make_node('Momentum', ['R', 'T', 'X1', 'X2', 'G1', 'G2', 'V1', 'V2'],
             ['X1_new', 'X2_new', 'V1_new', 'V2_new'],
             alpha=0.9, beta=1.0, norm_coefficient=0.02, mode='nesterov',
             domain=AI_ONNX_PREVIEW_TRAINING_DOMAIN)],
            [])

        self._assert_inferred(
            graph,
            [make_tensor_value_info('X1_new', TensorProto.FLOAT, (1, 2)),
             make_tensor_value_info('X2_new', TensorProto.FLOAT, (3, 4)),
             make_tensor_value_info('V1_new', TensorProto.FLOAT, (1, 2)),
             make_tensor_value_info('V2_new', TensorProto.FLOAT, (3, 4))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, 12), helper.make_opsetid(AI_ONNX_PREVIEW_TRAINING_DOMAIN, 1)])

    def test_adam(self):  # type: () -> None
        graph = self._make_graph(
            [('R', TensorProto.FLOAT, ()),  # scalar's shape is ()
             ('T', TensorProto.INT64, ()),  # scalar's shape is ()
             ('X', TensorProto.FLOAT, (1, 2)),
             ('G', TensorProto.FLOAT, (1, 2)),
             ('V', TensorProto.FLOAT, (1, 2)),
             ('H', TensorProto.FLOAT, (1, 2))],
            [make_node('Adam', ['R', 'T', 'X', 'G', 'V', 'H'], ['X_new', 'V_new', 'H_new'],
             domain=AI_ONNX_PREVIEW_TRAINING_DOMAIN,
             alpha=0.9, beta=1.0, norm_coefficient=0.02)],
            [])

        infos = [make_tensor_value_info('X_new', TensorProto.FLOAT, (1, 2)),
                 make_tensor_value_info('V_new', TensorProto.FLOAT, (1, 2)),
                 make_tensor_value_info('H_new', TensorProto.FLOAT, (1, 2))]

        self._assert_inferred(
            graph,
            infos,
            opset_imports=[make_opsetid(AI_ONNX_PREVIEW_TRAINING_DOMAIN, 1), make_opsetid(ONNX_DOMAIN, 12)])

    def test_adam_multiple(self):  # type: () -> None
        graph = self._make_graph(
            [('R', TensorProto.FLOAT, ()),  # scalar's shape is ()
             ('T', TensorProto.INT64, ()),  # scalar's shape is ()
             ('X1', TensorProto.FLOAT, (1, 2)),
             ('X2', TensorProto.FLOAT, (3, 4)),
             ('G1', TensorProto.FLOAT, (1, 2)),
             ('G2', TensorProto.FLOAT, (3, 4)),
             ('V1', TensorProto.FLOAT, (1, 2)),
             ('V2', TensorProto.FLOAT, (3, 4)),
             ('H1', TensorProto.FLOAT, (1, 2)),
             ('H2', TensorProto.FLOAT, (3, 4))],
            [make_node('Adam', ['R', 'T', 'X1', 'X2', 'G1', 'G2', 'V1', 'V2', 'H1', 'H2'],
             ['X1_new', 'X2_new', 'V1_new', 'V2_new', 'H1_new', 'H2_new'],
             domain=AI_ONNX_PREVIEW_TRAINING_DOMAIN,
             alpha=0.9, beta=1.0, norm_coefficient=0.02)],
            [])

        infos = [make_tensor_value_info('X1_new', TensorProto.FLOAT, (1, 2)),
                 make_tensor_value_info('X2_new', TensorProto.FLOAT, (3, 4)),
                 make_tensor_value_info('V1_new', TensorProto.FLOAT, (1, 2)),
                 make_tensor_value_info('V2_new', TensorProto.FLOAT, (3, 4)),
                 make_tensor_value_info('H1_new', TensorProto.FLOAT, (1, 2)),
                 make_tensor_value_info('H2_new', TensorProto.FLOAT, (3, 4))]

        self._assert_inferred(
            graph,
            infos,
            opset_imports=[make_opsetid(AI_ONNX_PREVIEW_TRAINING_DOMAIN, 1), make_opsetid(ONNX_DOMAIN, 12)])

    def test_pad_opset10(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (1, None, 2))],
            [make_node('Pad', 'x', 'y', pads=[1, 3, 1, 1, 0, 1])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (3, None, 4))], opset_imports=[helper.make_opsetid(ONNX_DOMAIN, 10)])  # type: ignore

    def test_constant_pad_2d_opset10(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (2, 3, 4, 4))],
            [make_node('Pad', 'x', 'y', pads=[0, 0, 3, 1, 0, 0, 4, 2], mode="constant", value=2.0)],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (2, 3, 11, 7))], opset_imports=[helper.make_opsetid(ONNX_DOMAIN, 10)])

    def test_pad(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (1, None, 2)),
             ('pads', TensorProto.INT64, (6,))],
            [make_node('Pad', ['x', 'pads'], 'y')],
            [],
            initializer=[make_tensor('pads', TensorProto.INT64, (6,), (1, 3, 1, 1, 0, 1,))])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (3, None, 4))])  # type: ignore

    def test_gatherelements_basic(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (6,)),
             ('indices', TensorProto.INT64, (2,))],
            [make_node('GatherElements', ['x', 'indices'], ['y'])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (2,))])

    def test_gatherelements_indices_missing_shape(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (6,)),
             ('indices', TensorProto.INT64, None)],  # type: ignore
            [make_node('GatherElements', ['x', 'indices'], ['y'])],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, None)])  # type: ignore

    def test_einsum_transpose(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (3, 4))],
            [make_node('Einsum', ['x'], ['y'], equation='ij->ji')],
            [],)
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (None, None))])  # type: ignore

    def test_einsum_dot(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (1,)),
             ('y', TensorProto.FLOAT, (1,))],
            [make_node('Einsum', ['x', 'y'], ['z'], equation='i,i->')],
            [],)
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, ())])  # type: ignore

    def test_einsum_scalar(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, ()),
             ('y', TensorProto.FLOAT, ())],
            [make_node('Einsum', ['x', 'y'], ['z'], equation=',->')],
            [],)
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, ())])  # type: ignore

    def test_einsum_outer_prod(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (3, 5)),
             ('y', TensorProto.FLOAT, (7, 9))],
            [make_node('Einsum', ['x', 'y'], ['z'], equation='ij,ab->ijab')],
            [],)
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, (None, None, None, None))])  # type: ignore

    def test_einsum_sum_along_dim(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (3, 4))],
            [make_node('Einsum', ['x'], ['y'], equation='i j->i ')],
            [],)
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (None, ))])  # type: ignore

    def test_einsum_ellipsis(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (3, 4, 4))],
            [make_node('Einsum', ['x'], ['y'], equation='... ii ->... i')],
            [],)
        self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (None, None))])  # type: ignore

    def test_einsum_ellipsis_2(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (2, 2, 2)),
             ('y', TensorProto.FLOAT, (2, 2, 2))],
            [make_node('Einsum', ['x', 'y'], ['z'], equation='...ij,...jk->...ik')],
            [], )
        self._assert_inferred(graph,
                              [make_tensor_value_info('z', TensorProto.FLOAT, (None, None, None))])  # type: ignore

    def test_einsum_ellipsis_3(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (2, 2, 2)),
             ('y', TensorProto.FLOAT, (2, 2, 2))],
            [make_node('Einsum', ['x', 'y'], ['z'], equation='...ij,...jk')],
            [], )
        self._assert_inferred(graph,
                              [make_tensor_value_info('z', TensorProto.FLOAT, (None, None, None))])  # type: ignore

    def test_einsum_contraction(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (5, 6, 7, 8)),
             ('y', TensorProto.FLOAT, (8, 9, 10))],
            [make_node('Einsum', ['x', 'y'], ['z'], equation='abcd,dfg->abcfg')],
            [], )
        self._assert_inferred(graph,
                              [make_tensor_value_info('z', TensorProto.FLOAT, (None, None, None, None, None))])  # type: ignore

    def test_einsum_contraction_2(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (3, 4, 5)),
             ('y', TensorProto.FLOAT, (3, 5))],
            [make_node('Einsum', ['x', 'y'], ['z'], equation='ijk,ik->jk')],
            [], )
        self._assert_inferred(graph,
                              [make_tensor_value_info('z', TensorProto.FLOAT, (None, None))])  # type: ignore

    def test_einsum_batch_matmul(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (5, 2, 3)),
             ('y', TensorProto.FLOAT, (5, 3, 4))],
            [make_node('Einsum', ['x', 'y'], ['z'], equation='bij , b jk-> bik')],
            [],)
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, (None, None, None))])  # type: ignore

    def test_einsum_left_hand_eqn(self):  # type: () -> None
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (2, 3)),
             ('y', TensorProto.FLOAT, (3, 4))],
            [make_node('Einsum', ['x', 'y'], ['z'], equation='ij,kl')],
            [],)
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, (None, None, None, None))])  # type: ignore

    def test_einsum_incorrect_num_inputs(self):  # type: () -> None
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (2, 3)),
             ("y", TensorProto.FLOAT, (2, 3)),
             ("z", TensorProto.FLOAT, (2, 3))],
            [make_node('Einsum', ['x', 'y'], ['z'], equation='i,...j, k, l-> i')],
            [])
        self.assertRaises(checker.ValidationError, self._inferred, graph)

    def test_negative_log_likehood_shape_is_NCdd(self):  # type: () -> None
        N, C = 3, 4
        graph = self._make_graph(
            [('input', TensorProto.FLOAT, (N, C)),
             ('target', TensorProto.INT64, (N,))],
            [make_node('NegativeLogLikelihoodLoss', ['input', 'target'], ['loss'], reduction='none')],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('loss', TensorProto.FLOAT, (N, ))])  # type: ignore

    def test_negative_log_likehood_shape_is_NC_with_weight(self):  # type: () -> None
        N, C = 3, 4
        graph = self._make_graph(
            [('input', TensorProto.FLOAT, (N, C)),
             ('target', TensorProto.INT64, (N,)),
             ('weight', TensorProto.FLOAT, (C,))],
            [make_node('NegativeLogLikelihoodLoss', ['input', 'target', 'weight'], ['loss'], reduction='none')],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('loss', TensorProto.FLOAT, (N, ))])  # type: ignore

    def test_negative_log_likehood_shape_is_NC_reduction_mean(self):  # type: () -> None
        N, C = 3, 4
        graph = self._make_graph(
            [('input', TensorProto.FLOAT, (N, C)),
             ('target', TensorProto.INT64, (N,))],
            [make_node('NegativeLogLikelihoodLoss', ['input', 'target'], ['loss'], reduction='mean')],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('loss', TensorProto.FLOAT, ())])  # type: ignore

    def test_negative_log_likehood_shape_is_NC_with_weight_reduction_mean(self):  # type: () -> None
        N, C = 3, 4
        graph = self._make_graph(
            [('input', TensorProto.FLOAT, (N, C)),
             ('target', TensorProto.INT64, (N,)),
             ('weight', TensorProto.FLOAT, (C,))],
            [make_node('NegativeLogLikelihoodLoss', ['input', 'target', 'weight'], ['loss'], reduction='mean')],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('loss', TensorProto.FLOAT, ())])  # type: ignore

    def test_negative_log_likehood_shape_is_NCd1d2(self):  # type: () -> None
        N, C, d1, d2 = 3, 4, 5, 6
        graph = self._make_graph(
            [("input", TensorProto.FLOAT, (N, C, d1, d2)),
             ("target", TensorProto.INT64, (N, d1, d2))],
            [make_node('NegativeLogLikelihoodLoss', ['input', 'target'], ['loss'], reduction='none')],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('loss', TensorProto.FLOAT, (N, d1, d2))])  # type: ignore

    def test_negative_log_likehood_shape_is_NCd1d2_with_weight(self):  # type: () -> None
        N, C, d1, d2 = 3, 4, 5, 6
        graph = self._make_graph(
            [("input", TensorProto.FLOAT, (N, C, d1, d2)),
             ("target", TensorProto.INT64, (N, d1, d2)),
             ("weight", TensorProto.FLOAT, (C,))],
            [make_node('NegativeLogLikelihoodLoss', ['input', 'target', 'weight'], ['loss'], reduction='none')],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('loss', TensorProto.FLOAT, (N, d1, d2))])  # type: ignore

    def test_negative_log_likehood_shape_is_NCd1d2_reduction_sum(self):  # type: () -> None
        N, C, d1, d2 = 3, 4, 5, 6
        graph = self._make_graph(
            [("input", TensorProto.FLOAT, (N, C, d1, d2)),
             ("target", TensorProto.INT64, (N, d1, d2))],
            [make_node('NegativeLogLikelihoodLoss', ['input', 'target'], ['loss'], reduction='sum')],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('loss', TensorProto.FLOAT, ())])  # type: ignore

    def test_negative_log_likehood_shape_is_NCd1d2_with_weight_reduction_mean(self):  # type: () -> None
        N, C, d1, d2 = 3, 4, 5, 6
        graph = self._make_graph(
            [("input", TensorProto.FLOAT, (N, C, d1, d2)),
             ("target", TensorProto.INT64, (N, d1, d2)),
             ("weight", TensorProto.FLOAT, (C,))],
            [make_node('NegativeLogLikelihoodLoss', ['input', 'target', 'weight'], ['loss'], reduction='mean')],
            [])
        self._assert_inferred(graph, [make_tensor_value_info('loss', TensorProto.FLOAT, ())])  # type: ignore

    def test_negative_log_likehood_input_target_shape_mismatch(self):  # type: () -> None
        N, C, d1, d2 = 3, 4, 5, 6
        graph = self._make_graph(
            [("input", TensorProto.FLOAT, (N, d1, d2)),
             ("target", TensorProto.INT64, (N, d1 + 1, d2)),
             ("weight", TensorProto.FLOAT, (C,)),
             ("loss", TensorProto.FLOAT, ())],
            [make_node('NegativeLogLikelihoodLoss', ['input', 'target', 'weight'], ['loss'], reduction='mean')],
            [])
        self.assertRaises(checker.ValidationError, self._inferred, graph)

    def test_negative_log_likehood_input_weight_shape_mismatch(self):  # type: () -> None
        N, C, d1, d2 = 3, 4, 5, 6
        graph = self._make_graph(
            [("input", TensorProto.FLOAT, (N, C, d1, d2)),
             ("target", TensorProto.INT64, (N, d1, d2)),
             ("weight", TensorProto.FLOAT, (C + 1,)),
             ("loss", TensorProto.FLOAT, (N, d1, d2))],
            [make_node('NegativeLogLikelihoodLoss', ['input', 'target', 'weight'], ['loss'], reduction='none')],
            [])
        self.assertRaises(checker.ValidationError, self._inferred, graph)

    def test_softmax_cross_entropy_none(self):  # type: () -> None
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (2, 3)),
             ("y", TensorProto.FLOAT, (2,))],
            [make_node('SoftmaxCrossEntropyLoss', ['x', 'y'], ['z'], reduction='none')],
            [],)
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, (2,))])  # type: ignore

    def test_softmax_cross_entropy_mean(self):  # type: () -> None
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (2, 3)),
             ("y", TensorProto.FLOAT, (2,))],
            [make_node('SoftmaxCrossEntropyLoss', ['x', 'y'], ['z'], reduction='mean')],
            [],)
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, ())])  # type: ignore

    def test_softmax_cross_entropy_none_NCD1D2(self):  # type: () -> None
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (2, 3, 5, 8)),
             ("y", TensorProto.FLOAT, (2, 5, 8))],
            [make_node('SoftmaxCrossEntropyLoss', ['x', 'y'], ['z'], reduction='none')],
            [],)
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, (2, 5, 8))])  # type: ignore

    def test_softmax_cross_entropy_mean_NCD1D2(self):  # type: () -> None
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (2, 3, 4, 5)),
             ("y", TensorProto.FLOAT, (2, 4, 5))],
            [make_node('SoftmaxCrossEntropyLoss', ['x', 'y'], ['z'], reduction='mean')],
            [],)
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, ())])  # type: ignore

    def test_celu_function_output_shape(self):  # type: () -> None
        graph = self._make_graph(
            [('X', TensorProto.FLOAT, (25, 48, 16, 16))],
            [make_node('Celu', ['X'], ['Y'], alpha=2.0)],
            []
        )
        self._assert_inferred(graph, [make_tensor_value_info('Y', TensorProto.FLOAT, (25, 48, 16, 16))])


if __name__ == '__main__':
    unittest.main()
