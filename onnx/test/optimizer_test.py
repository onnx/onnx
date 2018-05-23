from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from onnx import checker, helper, ModelProto, TensorProto, GraphProto, NodeProto
from typing import Sequence, Text, Tuple, List, Callable

import numpy as np  # type: ignore

import onnx.optimizer
import unittest


class TestOptimizer(unittest.TestCase):

    def _optimized(self, graph, opts):  # type: (GraphProto, Sequence[Text]) -> ModelProto
        orig_model = helper.make_model(graph, producer_name='onnx-test')
        optimized_model = onnx.optimizer.optimize(orig_model, opts)
        checker.check_model(optimized_model)
        return optimized_model

    # input_types and output_types are lists of triples of (name, type, shape)
    def _make_fake_loop_op(self,
                           body_nodes,  # type: Sequence[NodeProto]
                           input_types,  # type: Sequence[Tuple[TensorProto.DataType, Sequence[int], Text]]
                           output_types  # type: Sequence[Tuple[TensorProto.DataType, Sequence[int], Text]]
                           ):  # type: (...) -> List[NodeProto]
        zero = helper.make_tensor("trip_count_value", TensorProto.INT32, (), [10])
        true = helper.make_tensor("condition", TensorProto.BOOL, (), [True])
        # lcd is a dummy loop-carried dependency that only exists because
        # right now the schema checker is broken and assumes a variadic
        # input needs at least one value.
        graph_inputs = [helper.make_tensor_value_info("i", TensorProto.INT32, ()),
                        helper.make_tensor_value_info("cond", TensorProto.BOOL, ())]
        for type, shape, name in input_types:
            graph_inputs.append(helper.make_tensor_value_info("_" + name, type, shape))
        graph_outputs = [helper.make_tensor_value_info("cond", TensorProto.BOOL, ())]
        for type, shape, name in output_types:
            graph_outputs.append(helper.make_tensor_value_info("_" + name, type, shape))
        body_graph = helper.make_graph(body_nodes, "body_graph", graph_inputs,
                                       graph_outputs)
        loop_inputs = ["trip_count", "condition"]
        loop_inputs.extend([name for _, _, name in input_types])
        # TODO: fix checker to accept 0-input variadic inputs
        if len(loop_inputs) == 2:
            loop_inputs.append("")
        loop_outputs = [name for _, _, name in output_types]
        retval_nodes = [
            helper.make_node("Constant", [], ["trip_count"], value=zero),
            helper.make_node("Constant", [], ["condition"], value=true),
            helper.make_node("Loop", loop_inputs, loop_outputs, body=body_graph)
        ]
        return retval_nodes

    def _make_fake_if_op(self,
                         true_nodes,  # type: Sequence[NodeProto]
                         false_nodes,  # type: Sequence[NodeProto]
                         output_types  # type: Sequence[Tuple[TensorProto.DataType, Sequence[int], Text]]
                         ):  # type: (...) -> List[NodeProto]
        true = helper.make_tensor("condition", TensorProto.BOOL, (), [True])
        true_graph = helper.make_graph(true_nodes, "true_graph", [], [])
        false_graph = helper.make_graph(false_nodes, "false_graph", [], [])
        if_inputs = ["condition"]
        if_outputs = [name for _, _, name in output_types]
        retval_nodes = [
            helper.make_node("Constant", [], ["condition"], value=true),
            helper.make_node("If", if_inputs, if_outputs, then_branch=true_graph,
                             else_branch=false_graph)
        ]
        return retval_nodes

    # fn is a function that takes a single node as argument
    def _visit_all_nodes_recursive(self, graph, fn):  # type: (GraphProto, Callable[[NodeProto], None]) -> None
        for node in graph.node:
            fn(node)
            for attr in node.attribute:
                if attr.g is not None:
                    self._visit_all_nodes_recursive(attr.g, fn)
                if len(attr.graphs):
                    for gr in attr.graphs:
                        self._visit_all_nodes_recursive(gr, fn)

    def test_eliminate_identity_single_use(self):  # type: () -> None
        nodes = [helper.make_node("Identity", ["X"], ["Y"])]
        nodes.extend(self._make_fake_loop_op(
            [helper.make_node("Identity", ["_Y"], ["_Y2"])],
            [(TensorProto.FLOAT, (5,), "Y")],
            [(TensorProto.FLOAT, (5,), "Y2")]))
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (5,))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (5,)),
             helper.make_tensor_value_info("Y2", TensorProto.FLOAT, (5,))])
        optimized_model = self._optimized(graph, ["eliminate_identity"])

        # All identity nodes should have been eliminated
        def check_identity(node):  # type: (NodeProto) -> None
            assert node.op_type != "Identity"
        self._visit_all_nodes_recursive(optimized_model.graph, check_identity)
        # Use of the output from the Identity node in the main graph should
        # have been replaced with the input to the identity node
        assert len(optimized_model.graph.output) == 2
        assert optimized_model.graph.output[0].name == "X"
        # Use of the output from the Identity node in the loop graph should
        # have been replaced with the input to that identity node
        assert len(optimized_model.graph.node[2].attribute[0].g.output) == 2
        assert optimized_model.graph.node[2].attribute[0].g.output[1].name == "_Y"

    def test_eliminate_identity_multiple_uses(self):  # type: () -> None
        identity = helper.make_node("Identity", ["X"], ["Y"])
        add = helper.make_node("Add", ["Z", "Y"], ["A"])
        mul = helper.make_node("Mul", ["A", "Y"], ["B"])
        graph = helper.make_graph(
            [identity, add, mul],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (5,)),
             helper.make_tensor_value_info("Z", TensorProto.FLOAT, (5,))],
            [helper.make_tensor_value_info("B", TensorProto.FLOAT, (5,))])
        optimized_model = self._optimized(graph, ["eliminate_identity"])

        for node in optimized_model.graph.node:
            assert node.op_type != "Identity"
        assert len(optimized_model.graph.node) == 2

    def test_nop_transpose(self):  # type: () -> None
        nodes = [helper.make_node("Transpose", ["X"], ["Y"], perm=[0, 1])]
        nodes.extend(self._make_fake_loop_op(
            [helper.make_node("Transpose", ["_Y"], ["_Y2"], perm=[0, 1])],
            [(TensorProto.FLOAT, (2, 3), "Y")],
            [(TensorProto.FLOAT, (2, 3), "Y2")]))
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (2, 3)),
             helper.make_tensor_value_info("Y2", TensorProto.FLOAT, (2, 3))])
        optimized_model = self._optimized(graph, ["eliminate_nop_transpose"])

        def check_transpose(node):  # type: (NodeProto) -> None
            assert node.op_type != "Transpose"
        self._visit_all_nodes_recursive(optimized_model.graph, check_transpose)
        # Use of the output from the Transpose node in the main graph should
        # have been replaced with the input to the identity node
        assert len(optimized_model.graph.output) == 2
        assert optimized_model.graph.output[0].name == "X"
        # Use of the output from the Transpose node in the loop graph should
        # have been replaced with the input to that identity node
        assert len(optimized_model.graph.node[2].attribute[0].g.output) == 2
        assert optimized_model.graph.node[2].attribute[0].g.output[1].name == "_Y"

    def test_nop_transpose_default(self):  # type: () -> None
        trans = helper.make_node("Transpose", ["X"], ["Y"])
        graph = helper.make_graph(
            [trans],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (3, 2))])
        optimized_model = self._optimized(graph, ["eliminate_nop_transpose"])

        assert len(list(optimized_model.graph.node)) == 1
        assert optimized_model.graph.node[0].op_type == "Transpose"

    def test_fuse_transpose(self):  # type: () -> None
        nodes = [helper.make_node("Transpose", ["X"], ["Y"], perm=[1, 0, 2]),
                 helper.make_node("Transpose", ["Y"], ["Z"], perm=[2, 0, 1]),
                 helper.make_node("Transpose", ["Z"], ["A"], perm=[2, 0, 1])]
        nodes.extend(self._make_fake_loop_op(
            [helper.make_node("Transpose", ["_X"], ["_Y2"], perm=[1, 0, 2]),
             helper.make_node("Transpose", ["_Y2"], ["_Y3"], perm=[2, 0, 1]),
             helper.make_node("Transpose", ["_Y3"], ["_Y4"], perm=[2, 0, 1])],
            [(TensorProto.FLOAT, (2, 3), "X")],
            [(TensorProto.FLOAT, (2, 3), "Y4")]))
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3, 4))],
            [helper.make_tensor_value_info("A", TensorProto.FLOAT, (4, 3, 2)),
             helper.make_tensor_value_info("Y4", TensorProto.FLOAT, (4, 3, 2))])
        optimized_model = self._optimized(graph, ["fuse_consecutive_transposes"])

        # Transpose, Constant (trip count), Constant (cond), Loop
        assert len(list(optimized_model.graph.node)) == 4
        # Transpose
        assert len(optimized_model.graph.node[3].attribute[0].g.node) == 1

    def test_fuse_transpose_default(self):  # type: () -> None
        trans1 = helper.make_node("Transpose", ["X"], ["Y"])
        trans2 = helper.make_node("Transpose", ["Y"], ["Z"])
        graph = helper.make_graph(
            [trans1, trans2],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3, 4))],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (2, 3, 4))])
        optimized_model = self._optimized(graph, ["fuse_consecutive_transposes"])

        assert len(list(optimized_model.graph.node)) == 0

    def test_fuse_transpose_default_no_fuse(self):  # type: () -> None
        trans1 = helper.make_node("Transpose", ["X"], ["Y"])
        trans2 = helper.make_node("Transpose", ["Y"], ["Z"], perm=[0, 1, 2])
        graph = helper.make_graph(
            [trans1, trans2],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3, 4))],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (4, 3, 2))])
        optimized_model = self._optimized(graph, ["fuse_consecutive_transposes"])

        assert len(list(optimized_model.graph.node)) == 2
        for node in optimized_model.graph.node:
            assert node.op_type == "Transpose"

    def test_fuse_transpose_into_gemm(self):  # type: () -> None
        nodes = [helper.make_node("Transpose", ["X"], ["A"], perm=[1, 0]),
                 helper.make_node("Transpose", ["Y"], ["B"], perm=[1, 0]),
                 helper.make_node("Gemm", ["A", "B", "C"], ["Z"])]
        nodes.extend(self._make_fake_loop_op(
            [helper.make_node("Transpose", ["_X"], ["_A"], perm=[1, 0]),
             helper.make_node("Transpose", ["_Y"], ["_B"], perm=[1, 0]),
             helper.make_node("Gemm", ["_A", "_B", "_C"], ["_Z2"])],
            [(TensorProto.FLOAT, (2, 3), "X"),
             (TensorProto.FLOAT, (5, 2), "Y"),
             (TensorProto.FLOAT, (3, 5), "C")],
            [(TensorProto.FLOAT, (2, 3), "Z2")]))
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (5, 2)),
             helper.make_tensor_value_info("C", TensorProto.FLOAT, (3, 5))],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (3, 5))])
        optimized_model = self._optimized(graph, ["fuse_transpose_into_gemm"])

        # Gemm, Constant (trip count), Constant (cond), Loop
        assert len(list(optimized_model.graph.node)) == 4
        assert optimized_model.graph.node[0].op_type == "Gemm"
        # Gemm
        assert len(optimized_model.graph.node[3].attribute[0].g.node) == 1
        assert optimized_model.graph.node[3].attribute[0].g.node[0].op_type == "Gemm"

    def test_fuse_add_bias_into_conv_use_weight_shape(self):  # type: () -> None
        nodes = [helper.make_node("Conv", ["X", "Y"], ["Z"]),
                 helper.make_node("Add", ["Z", "A"], ["B"])]
        nodes.extend(self._make_fake_loop_op(
            [helper.make_node("Conv", ["_X", "_Y"], ["_Z"]),
             helper.make_node("Add", ["_Z", "_A"], ["_B2"])],
            [(TensorProto.FLOAT, (1, 5, 3, 3), "X"),
             (TensorProto.FLOAT, (16, 5, 3, 3), "Y"),
             (TensorProto.FLOAT, (16, 1, 1), "A")],
            [(TensorProto.FLOAT, (1, 16, 3, 3), "B2")]))
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 5, 3, 3)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (16, 5, 3, 3)),
             helper.make_tensor_value_info("A", TensorProto.FLOAT, (16, 1, 1))],
            [helper.make_tensor_value_info("B", TensorProto.FLOAT, (1, 16, 3, 3))],
        )
        optimized_model = self._optimized(graph, ["fuse_add_bias_into_conv"])

        # Squeeze, Conv, Constant (trip count), Constant (condition), Loop
        assert len(list(optimized_model.graph.node)) == 5
        assert optimized_model.graph.node[0].op_type == 'Squeeze'
        assert optimized_model.graph.node[1].op_type == 'Conv'
        assert optimized_model.graph.output[0].name == 'Z'
        # Squeeze, Conv
        assert len(optimized_model.graph.node[4].attribute[0].g.node) == 2
        assert optimized_model.graph.node[4].attribute[0].g.node[0].op_type == 'Squeeze'
        assert optimized_model.graph.node[4].attribute[0].g.node[1].op_type == 'Conv'
        # Output 1 since 0 is 'cond'
        assert optimized_model.graph.node[4].attribute[0].g.output[1].name == '_Z'

    def test_fuse_add_bias_into_conv_use_weight_shape_with_tile(self):  # type: () -> None
        conv = helper.make_node("Conv", ["X", "Y"], ["Z"])
        add = helper.make_node("Add", ["Z", "A"], ["B"])
        graph = helper.make_graph(
            [conv, add],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 5, 3, 3)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (16, 5, 3, 3)),
             helper.make_tensor_value_info("A", TensorProto.FLOAT, (1,))],
            [helper.make_tensor_value_info("B", TensorProto.FLOAT, (1, 16, 3, 3))],
        )
        optimized_model = self._optimized(graph, ["fuse_add_bias_into_conv"])

        assert len(list(optimized_model.graph.node)) == 3
        assert len(optimized_model.graph.value_info) == 1
        assert optimized_model.graph.value_info[0].type.tensor_type.elem_type == TensorProto.INT64
        assert len(optimized_model.graph.value_info[0].type.tensor_type.shape.dim) == 1
        assert optimized_model.graph.node[0].op_type == 'Constant'
        assert optimized_model.graph.node[1].op_type == 'Tile'
        assert optimized_model.graph.node[2].op_type == 'Conv'
        assert optimized_model.graph.output[0].name == 'Z'

    def test_fuse_add_bias_into_conv_use_conv_shape(self):  # type: () -> None
        sub = helper.make_node("Sub", ["M", "N"], ["Y"])
        conv = helper.make_node("Conv", ["X", "Y"], ["Z"])
        add = helper.make_node("Add", ["Z", "A"], ["B"])
        graph = helper.make_graph(
            [sub, conv, add],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 5, 3, 3)),
             helper.make_tensor_value_info("M", TensorProto.FLOAT, (16, 5, 3, 3)),
             helper.make_tensor_value_info("N", TensorProto.FLOAT, (16, 5, 3, 3)),
             helper.make_tensor_value_info("A", TensorProto.FLOAT, (1, 16, 1, 1))],
            [helper.make_tensor_value_info("B", TensorProto.FLOAT, (1, 16, 3, 3))],
            value_info=[
                helper.make_tensor_value_info("Z", TensorProto.FLOAT, (1, 16, 3, 3))
            ],
        )
        optimized_model = self._optimized(graph, ["fuse_add_bias_into_conv"])

        assert len(optimized_model.graph.node) == 3
        assert optimized_model.graph.node[0].op_type == 'Sub'
        assert optimized_model.graph.node[1].op_type == 'Squeeze'
        assert optimized_model.graph.node[2].op_type == 'Conv'
        assert optimized_model.graph.output[0].name == 'Z'
        assert optimized_model.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
        assert len(optimized_model.graph.output[0].type.tensor_type.shape.dim) == 4

    def test_fuse_add_bias_into_conv_use_move_constant(self):  # type: () -> None
        conv = helper.make_node("Conv", ["X", "Y"], ["Z"])
        constant = helper.make_node("Constant", [], ["A"],
                                    value=helper.make_tensor(
                                        name="bias",
                                        data_type=TensorProto.FLOAT,
                                        dims=(16,),
                                        vals=np.random.randn(16).astype(np.float32).tolist()))
        add = helper.make_node("Add", ["Z", "A"], ["B"])
        graph = helper.make_graph(
            [conv, constant, add],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 5, 3, 3)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (16, 5, 3, 3))],
            [helper.make_tensor_value_info("B", TensorProto.FLOAT, (1, 16, 3, 3))],
            value_info=[
                helper.make_tensor_value_info("A", TensorProto.FLOAT, (16, 1, 1)),
            ]
        )
        optimized_model = self._optimized(graph, ["fuse_add_bias_into_conv"])

        assert len(optimized_model.graph.node) == 3
        assert optimized_model.graph.node[0].op_type == 'Constant'
        assert optimized_model.graph.node[1].op_type == 'Squeeze'
        assert optimized_model.graph.node[2].op_type == 'Conv'
        assert optimized_model.graph.output[0].name == 'Z'
        assert optimized_model.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
        assert len(optimized_model.graph.output[0].type.tensor_type.shape.dim) == 4

    def test_fuse_add_bias_into_conv_squeeze_1d_bias_no_fuse(self):  # type: () -> None
        conv = helper.make_node("Conv", ["X", "Y"], ["Z"])
        add = helper.make_node("Add", ["Z", "A"], ["B"])
        graph = helper.make_graph(
            [conv, add],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 5, 3, 3)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (16, 5, 3, 3)),
             helper.make_tensor_value_info("A", TensorProto.FLOAT, (3,))],
            [helper.make_tensor_value_info("B", TensorProto.FLOAT, (1, 16, 3, 3))],
            value_info=[
                helper.make_tensor_value_info("Z", TensorProto.FLOAT, (1, 16, 3, 3)),
            ]
        )
        optimized_model = self._optimized(graph, ["fuse_add_bias_into_conv"])

        assert len(list(optimized_model.graph.node)) == 2
        assert optimized_model.graph.node[0].op_type == 'Conv'
        assert optimized_model.graph.node[1].op_type == 'Add'

    def test_fuse_add_bias_into_conv_squeeze_3d_bias_no_fuse(self):  # type: () -> None
        conv = helper.make_node("Conv", ["X", "Y"], ["Z"])
        add = helper.make_node("Add", ["Z", "A"], ["B"])
        graph = helper.make_graph(
            [conv, add],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 5, 3, 3)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (16, 5, 3, 3)),
             helper.make_tensor_value_info("A", TensorProto.FLOAT, (16, 3, 3))],
            [helper.make_tensor_value_info("B", TensorProto.FLOAT, (1, 16, 3, 3))],
            value_info=[
                helper.make_tensor_value_info("Z", TensorProto.FLOAT, (1, 16, 3, 3)),
            ]
        )
        optimized_model = self._optimized(graph, ["fuse_add_bias_into_conv"])

        assert len(list(optimized_model.graph.node)) == 2
        assert optimized_model.graph.node[0].op_type == 'Conv'
        assert optimized_model.graph.node[1].op_type == 'Add'

    def test_fuse_add_bias_into_conv_squeeze_4d_bias_no_fuse(self):  # type: () -> None
        conv = helper.make_node("Conv", ["X", "Y"], ["Z"])
        add = helper.make_node("Add", ["Z", "A"], ["B"])
        graph = helper.make_graph(
            [conv, add],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 5, 3, 3)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (16, 5, 3, 3)),
             helper.make_tensor_value_info("A", TensorProto.FLOAT, (1, 16, 3, 3))],
            [helper.make_tensor_value_info("B", TensorProto.FLOAT, (1, 16, 3, 3))]
        )
        optimized_model = self._optimized(graph, ["fuse_add_bias_into_conv"])

        assert len(list(optimized_model.graph.node)) == 2
        assert optimized_model.graph.node[0].op_type == 'Conv'
        assert optimized_model.graph.node[1].op_type == 'Add'

    def test_preserve_value_info(self):  # type: () -> None
        trans1 = helper.make_node("Transpose", ["X"], ["Y"], perm=[1, 0, 2])
        trans2 = helper.make_node("Transpose", ["Y"], ["Z"], perm=[2, 0, 1])
        trans3 = helper.make_node("Transpose", ["Z"], ["A"], perm=[2, 0, 1])
        graph = helper.make_graph(
            [trans1, trans2, trans3],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3, 4))],
            [helper.make_tensor_value_info("A", TensorProto.FLOAT, (4, 3, 2))])
        vi = helper.make_tensor_value_info("Y", TensorProto.FLOAT, (3, 2, 4))

        graph.value_info.extend([vi])

        optimized_model = self._optimized(graph, ["nop"])

        assert list(optimized_model.graph.value_info) == [vi]
        assert len(list(optimized_model.graph.node)) == 3

    def test_split(self):  # type: () -> None
        node = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['X'],
            value=onnx.helper.make_tensor(
                name='X',
                data_type=TensorProto.FLOAT,
                dims=[1],
                vals=[5],
            ),
        )
        graph = helper.make_graph(
            [node],
            'test-optimize-split',
            [],
            [helper.make_tensor_value_info('X', TensorProto.FLOAT, (1,))])

        init_model = self._optimized(graph, ['split_init'])
        self.assertEqual(len(init_model.graph.node), 1)
        self.assertEqual(len(init_model.graph.output), 1)
        self.assertEqual(init_model.graph.node[0].op_type, 'Constant')

        predict_model = self._optimized(graph, ['split_predict'])
        self.assertEqual(len(predict_model.graph.node), 0)
        self.assertEqual(len(predict_model.graph.input), 1)
        self.assertEqual(predict_model.graph.input[0].name, 'X')

    def test_lift_lex_loop(self):  # type: () -> None
        nodes = [helper.make_node("Identity", ["X"], ["Y"])]
        nodes.extend(self._make_fake_loop_op(
            [helper.make_node("Identity", ["X"], ["_Y2"]),
             helper.make_node("Identity", ["Y"], ["_Y3"])],
            [],
            [(TensorProto.FLOAT, (5,), "Y2"),
             (TensorProto.FLOAT, (5,), "Y3")]))
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (5,))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (5,)),
             helper.make_tensor_value_info("Y2", TensorProto.FLOAT, (5,))])
        optimized_model = self._optimized(graph, ["lift_lexical_references"])
        assert len(optimized_model.graph.node) == 4
        # body_graph, __control_inputs
        assert len(optimized_model.graph.node[3].attribute) == 2
        assert optimized_model.graph.node[3].attribute[1].name == "__control_inputs"
        assert optimized_model.graph.node[3].attribute[1].strings[0] == b"X"
        assert optimized_model.graph.node[3].attribute[1].strings[1] == b"Y"

    def test_lift_lex_if(self):  # type: () -> None
        nodes = [helper.make_node("Identity", ["X"], ["Y"])]
        nodes.extend(self._make_fake_if_op(
            [helper.make_node("Identity", ["X"], ["_Y2"]),
             helper.make_node("Identity", ["Y"], ["_Y3"])],
            [helper.make_node("Identity", ["X"], ["_Y2"]),
             helper.make_node("Identity", ["X"], ["_Y3"])],
            [(TensorProto.FLOAT, (5,), "Y2"),
             (TensorProto.FLOAT, (5,), "Y3")]))
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (5,))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (5,)),
             helper.make_tensor_value_info("Y2", TensorProto.FLOAT, (5,))])
        # "If" node now diverges from ONNX schema. Disable checking.
        optimized_model = self._optimized(graph, ["lift_lexical_references"])

        # Identity, Constant (condition), If
        assert len(optimized_model.graph.node) == 3
        # else_branch, then_branch, __control_inputs
        assert len(optimized_model.graph.node[2].attribute) == 3
        assert optimized_model.graph.node[2].attribute[2].name == "__control_inputs"
        assert optimized_model.graph.node[2].attribute[2].strings[0] == b"X"
        assert optimized_model.graph.node[2].attribute[2].strings[1] == b"Y"


if __name__ == '__main__':
    unittest.main()
