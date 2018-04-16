from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from onnx import checker, helper, TensorProto
from onnx.helper import make_node, make_tensor_value_info

import onnx.shape_inference
import unittest


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

    def test_cast(self):
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (2, 4, 3))],
            [make_node("Cast", ["x"], ["y"], to="UINT8")],
            [])
        self._assert_inferred(graph, [make_tensor_value_info("y", TensorProto.UINT8, (2, 4, 3))])

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


if __name__ == '__main__':
    unittest.main()
