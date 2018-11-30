from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from onnx import checker, helper, ModelProto, TensorProto, GraphProto, NodeProto
from typing import Sequence, Text, Tuple, List, Callable
from onnx import numpy_helper

import numpy as np  # type: ignore

import onnx.optimizer
import unittest


class TestOptimizer(unittest.TestCase):

    def _optimized(self, graph, opts, fixed_point=False):  # type: (GraphProto, Sequence[Text], bool) -> ModelProto
        orig_model = helper.make_model(graph, producer_name='onnx-test')
        optimized_model = onnx.optimizer.optimize(orig_model, opts, fixed_point)
        checker.check_model(optimized_model)
        return optimized_model

    # input_types and output_types are lists of triples of (name, type, shape)
    def _make_fake_loop_op(self,
                           body_nodes,   # type: Sequence[NodeProto]
                           input_types,  # type: Sequence[Tuple[TensorProto.DataType, Sequence[int], Text]]
                           output_types  # type: Sequence[Tuple[TensorProto.DataType, Sequence[int], Text]]
                           ):  # type: (...) -> List[NodeProto]
        zero = helper.make_tensor(
            "trip_count_value", TensorProto.INT32, (), [10])
        true = helper.make_tensor("condition", TensorProto.BOOL, (), [True])
        # lcd is a dummy loop-carried dependency that only exists because
        # right now the schema checker is broken and assumes a variadic
        # input needs at least one value.
        graph_inputs = [helper.make_tensor_value_info("i", TensorProto.INT32, ()),
                        helper.make_tensor_value_info("cond", TensorProto.BOOL, ())]
        for type, shape, name in input_types:
            graph_inputs.append(
                helper.make_tensor_value_info("_" + name, type, shape))
        graph_outputs = [helper.make_tensor_value_info(
            "cond", TensorProto.BOOL, ())]
        for type, shape, name in output_types:
            graph_outputs.append(
                helper.make_tensor_value_info("_" + name, type, shape))
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
                         true_nodes,   # type: Sequence[NodeProto]
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

    def test_get_available_passes(self):  # type: () -> None
        # FIXME does not garantees to be listing all
        graph = helper.make_graph([], "dummy_graph", [], [])
        list_of_passes = onnx.optimizer.get_available_passes()
        assert isinstance(list_of_passes, (list)) and len(list_of_passes) > 0
        for pass_name in list_of_passes:
            # If pass_name is invalid it throws a RuntimeError
            self._optimized(graph, [pass_name])

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

    def test_nop_pad(self):  # type: () -> None
        nodes = [helper.make_node("Pad", ["X"], ["Y"], pads=[0, 0])]
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (2, 3))])
        assert len(graph.node) == 1
        optimized_model = self._optimized(graph, ["eliminate_nop_pad"])

        def check_pad(node):  # type: (NodeProto) -> None
            assert node.op_type != "Pad"
        self._visit_all_nodes_recursive(optimized_model.graph, check_pad)
        assert len(optimized_model.graph.output) == 1
        assert optimized_model.graph.output[0].name == "X"
        assert len(optimized_model.graph.node) == 0

    def test_nop_pad_default(self):  # type: () -> None
        trans = helper.make_node("Pad", ["X"], ["Y"], pads=[0, 1])
        graph = helper.make_graph(
            [trans],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (2, 4))])
        optimized_model = self._optimized(graph, ["eliminate_nop_pad"])

        assert len(list(optimized_model.graph.node)) == 1
        assert optimized_model.graph.node[0].op_type == "Pad"

    def test_eliminate_unused_initializer(self):  # type: () -> None
        add = helper.make_node("Add", ["X", "Y"], ["Z"])
        graph = helper.make_graph(
            [add],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 2)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (1, 2))],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (1, 2))],
            [helper.make_tensor("A", TensorProto.FLOAT,
                                dims=(2, 3),
                                vals=np.random.randn(2, 3).astype(
                                    np.float32).tobytes(),
                                raw=True)])
        optimized_model = self._optimized(
            graph, ["eliminate_unused_initializer"])

        assert len(list(optimized_model.graph.initializer)) == 0

    def test_eliminate_unused_initializer_input(self):  # type: () -> None
        add = helper.make_node("Add", ["X", "Y"], ["Z"])
        graph = helper.make_graph(
            [add],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 2)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (1, 2)),
             helper.make_tensor_value_info("A", TensorProto.FLOAT, (2, 3))],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (1, 2))],
            [helper.make_tensor("A", TensorProto.FLOAT,
                                dims=(2, 3),
                                vals=np.random.randn(2, 3).astype(
                                    np.float32).tobytes(),
                                raw=True)])
        optimized_model = self._optimized(
            graph, ["eliminate_unused_initializer"])

        assert len(list(optimized_model.graph.initializer)) == 0
        assert len(optimized_model.graph.input) == 2

    def test_eliminate_unused_initializer_no_eliminate_used_default(self):  # type: () -> None
        add = helper.make_node("Add", ["X", "A"], ["Z"])
        graph = helper.make_graph(
            [add],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 2)),
             helper.make_tensor_value_info("A", TensorProto.FLOAT, (1, 2))],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (1, 2))],
            [helper.make_tensor("A", TensorProto.FLOAT,
                                dims=(1, 2),
                                vals=np.random.randn(1, 2).astype(
                                    np.float32).tobytes(),
                                raw=True)])
        optimized_model = self._optimized(
            graph, ["eliminate_unused_initializer"])

        assert len(list(optimized_model.graph.initializer)) == 1

    def test_eliminate_unused_initializer_no_eliminate_used(self):  # type: () -> None
        nodes = [helper.make_node("Add", ["X", "A"], ["Z"])]
        nodes.extend(self._make_fake_loop_op(
            [helper.make_node("Add", ["_X", "_A"], ["_Z2"])],
            [(TensorProto.FLOAT, (1, 2), "X"),
             (TensorProto.FLOAT, (1, 2), "A")],
            [(TensorProto.FLOAT, (1, 2), "Z2")]))
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 2)),
             helper.make_tensor_value_info("A", TensorProto.FLOAT, (1, 2))],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (1, 2))],
            [helper.make_tensor("A", TensorProto.FLOAT,
                                dims=(1, 2),
                                vals=np.random.randn(1, 2).astype(
                                    np.float32).tobytes(),
                                raw=True)])
        optimized_model = self._optimized(
            graph, ["eliminate_unused_initializer"])

        # Add, Constant (trip count), Constant (cond), Loop
        assert len(list(optimized_model.graph.node)) == 4
        assert optimized_model.graph.node[0].op_type == "Add"
        assert optimized_model.graph.output[0].name == "Z"
        # Add
        assert len(optimized_model.graph.node[3].attribute[0].g.node) == 1
        assert optimized_model.graph.node[3].attribute[0].g.node[0].op_type == 'Add'
        assert optimized_model.graph.node[3].attribute[0].g.output[1].name == '_Z2'

        assert len(list(optimized_model.graph.initializer)) == 1

    def test_eliminate_unused_initializer_no_eliminate_output(self):  # type: () -> None
        add = helper.make_node("Add", ["X", "Y"], ["Z"])
        graph = helper.make_graph(
            [add],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 2)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (1, 2)),
             helper.make_tensor_value_info("A", TensorProto.FLOAT, (2, 3))],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (1, 2)),
             helper.make_tensor_value_info("A", TensorProto.FLOAT, (2, 3))],
            [helper.make_tensor("A", TensorProto.FLOAT,
                                dims=(2, 3),
                                vals=np.random.randn(2, 3).astype(
                                    np.float32).tobytes(),
                                raw=True)])
        optimized_model = self._optimized(
            graph, ["eliminate_unused_initializer"])

        assert len(list(optimized_model.graph.initializer)) == 1
        assert "Z" in [o.name for o in optimized_model.graph.output]

    def test_extract_constant_to_initializer(self):  # type: () -> None
        conv = helper.make_node("Conv", ["X", "Y"], ["Z"])
        constant = helper.make_node("Constant", [], ["A"],
                                    value=helper.make_tensor(
                                        name="bias",
                                        data_type=TensorProto.FLOAT,
                                        dims=(16, 1, 1),
                                        vals=np.random.randn(16).astype(np.float32).tolist()))
        add = helper.make_node("Add", ["Z", "A"], ["B"])
        graph = helper.make_graph(
            [conv, constant, add],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 5, 3, 3)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (16, 5, 3, 3))],
            [helper.make_tensor_value_info(
                "B", TensorProto.FLOAT, (1, 16, 1, 1))],
        )
        optimized_model = self._optimized(
            graph, ["extract_constant_to_initializer"])
        self.assertEqual(
            set(vi.name for vi in optimized_model.graph.input),
            {'X', 'Y', 'A'})

        self.assertEqual(len(optimized_model.graph.initializer), 1)
        init = optimized_model.graph.initializer[0]
        self.assertEqual(init.name, 'A')
        self.assertEqual(init.dims, [16, 1, 1])
        self.assertEqual(init.data_type, TensorProto.FLOAT)

        self.assertEqual(
            [n.op_type for n in optimized_model.graph.node], ['Conv', 'Add'])

    def test_fuse_concats(self):  # type: () -> None
        nodes = [helper.make_node("Concat", ["A", "B", "C"], ["X"], axis=0),
                 helper.make_node("Concat", ["D", "E", "F"], ["Y"], axis=0),
                 helper.make_node("Concat", ["X", "Y"], ["Z"], axis=0)]
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("A", TensorProto.FLOAT, (2, 3, 4)),
            helper.make_tensor_value_info("B", TensorProto.FLOAT, (4, 3, 4)),
            helper.make_tensor_value_info("C", TensorProto.FLOAT, (2, 3, 4)),
            helper.make_tensor_value_info("D", TensorProto.FLOAT, (4, 3, 4)),
            helper.make_tensor_value_info("E", TensorProto.FLOAT, (2, 3, 4)),
            helper.make_tensor_value_info("F", TensorProto.FLOAT, (4, 3, 4))],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (18, 3, 4))])
        optimized_model = self._optimized(
            graph, ["fuse_consecutive_concats"], True)  # two passes are needed to simplify the graph to its simplest state.

        assert len(optimized_model.graph.node) == 1
        assert len(optimized_model.graph.node[0].input) == 6
        assert optimized_model.graph.node[0].op_type == "Concat"

    def test_fuse_concats_different_axis(self):  # type: () -> None
        nodes = [helper.make_node("Concat", ["A", "B", "C"], ["X"], axis=0),
                 helper.make_node("Concat", ["D", "E", "F"], ["Y"], axis=1),
                 helper.make_node("Concat", ["X", "Y"], ["Z"], axis=2)]
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("A", TensorProto.FLOAT, (2, 3, 4)),
            helper.make_tensor_value_info("B", TensorProto.FLOAT, (4, 3, 4)),
            helper.make_tensor_value_info("C", TensorProto.FLOAT, (2, 3, 4)),
            helper.make_tensor_value_info("D", TensorProto.FLOAT, (4, 3, 4)),
            helper.make_tensor_value_info("E", TensorProto.FLOAT, (4, 3, 4)),
            helper.make_tensor_value_info("F", TensorProto.FLOAT, (4, 3, 4))],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (18, 3, 4))])
        optimized_model = self._optimized(
            graph, ["fuse_consecutive_concats"], True)  # two passes are needed to simplify the graph to its simplest state.

        assert optimized_model.graph == graph

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
        optimized_model = self._optimized(
            graph, ["fuse_consecutive_transposes"])

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
        optimized_model = self._optimized(
            graph, ["fuse_consecutive_transposes"])

        assert len(list(optimized_model.graph.node)) == 0

    def test_fuse_transpose_default_no_fuse(self):  # type: () -> None
        trans1 = helper.make_node("Transpose", ["X"], ["Y"])
        trans2 = helper.make_node("Transpose", ["Y"], ["Z"], perm=[0, 1, 2])
        graph = helper.make_graph(
            [trans1, trans2],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3, 4))],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (4, 3, 2))])
        optimized_model = self._optimized(
            graph, ["fuse_consecutive_transposes"])

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
             helper.make_tensor_value_info(
                 "Y", TensorProto.FLOAT, (16, 5, 3, 3)),
             helper.make_tensor_value_info("A", TensorProto.FLOAT, (16, 1, 1))],
            [helper.make_tensor_value_info(
                "B", TensorProto.FLOAT, (1, 16, 1, 1))],
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
             helper.make_tensor_value_info(
                 "Y", TensorProto.FLOAT, (16, 5, 3, 3)),
             helper.make_tensor_value_info("A", TensorProto.FLOAT, (1,))],
            [helper.make_tensor_value_info(
                "B", TensorProto.FLOAT, (1, 16, 1, 1))],
        )
        optimized_model = self._optimized(graph, ["fuse_add_bias_into_conv"])

        assert len(list(optimized_model.graph.node)) == 3
        assert len(optimized_model.graph.value_info) == 1
        assert optimized_model.graph.value_info[0].type.tensor_type.elem_type == TensorProto.INT64
        assert len(
            optimized_model.graph.value_info[0].type.tensor_type.shape.dim) == 1
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
             helper.make_tensor_value_info(
                 "M", TensorProto.FLOAT, (16, 5, 3, 3)),
             helper.make_tensor_value_info(
                 "N", TensorProto.FLOAT, (16, 5, 3, 3)),
             helper.make_tensor_value_info("A", TensorProto.FLOAT, (1, 16, 1, 1))],
            [helper.make_tensor_value_info(
                "B", TensorProto.FLOAT, (1, 16, 1, 1))],
            value_info=[
                helper.make_tensor_value_info(
                    "Z", TensorProto.FLOAT, (1, 16, 1, 1))
            ],
        )
        optimized_model = self._optimized(graph, ["fuse_add_bias_into_conv"])

        assert len(optimized_model.graph.node) == 3
        assert optimized_model.graph.node[0].op_type == 'Sub'
        assert optimized_model.graph.node[1].op_type == 'Squeeze'
        assert optimized_model.graph.node[2].op_type == 'Conv'
        assert optimized_model.graph.output[0].name == 'Z'
        assert optimized_model.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
        assert len(
            optimized_model.graph.output[0].type.tensor_type.shape.dim) == 4

    def test_fuse_add_bias_into_conv_use_move_constant(self):  # type: () -> None
        conv = helper.make_node("Conv", ["X", "Y"], ["Z"])
        constant = helper.make_node("Constant", [], ["A"],
                                    value=helper.make_tensor(
                                        name="bias",
                                        data_type=TensorProto.FLOAT,
                                        dims=(16, 1, 1),
                                        vals=np.random.randn(16).astype(np.float32).tolist()))
        add = helper.make_node("Add", ["Z", "A"], ["B"])
        graph = helper.make_graph(
            [conv, constant, add],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 5, 3, 3)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (16, 5, 3, 3))],
            [helper.make_tensor_value_info(
                "B", TensorProto.FLOAT, (1, 16, 1, 1))],
            value_info=[
                helper.make_tensor_value_info(
                    "A", TensorProto.FLOAT, (16, 1, 1)),
            ]
        )
        optimized_model = self._optimized(graph, ["fuse_add_bias_into_conv"])

        assert len(optimized_model.graph.node) == 3
        assert optimized_model.graph.node[0].op_type == 'Constant'
        assert optimized_model.graph.node[1].op_type == 'Squeeze'
        assert optimized_model.graph.node[2].op_type == 'Conv'
        assert optimized_model.graph.output[0].name == 'Z'
        assert optimized_model.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
        assert len(
            optimized_model.graph.output[0].type.tensor_type.shape.dim) == 4

    def test_fuse_add_bias_into_conv_squeeze_1d_bias_no_fuse(self):  # type: () -> None
        conv = helper.make_node("Conv", ["X", "Y"], ["Z"])
        add = helper.make_node("Add", ["Z", "A"], ["B"])
        graph = helper.make_graph(
            [conv, add],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 5, 3, 3)),
             helper.make_tensor_value_info(
                 "Y", TensorProto.FLOAT, (16, 5, 3, 3)),
             helper.make_tensor_value_info("A", TensorProto.FLOAT, (3,))],
            [helper.make_tensor_value_info(
                "B", TensorProto.FLOAT, (1, 16, 1, 3))],
            value_info=[
                helper.make_tensor_value_info(
                    "Z", TensorProto.FLOAT, (1, 16, 1, 1)),
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
             helper.make_tensor_value_info(
                 "Y", TensorProto.FLOAT, (16, 5, 3, 3)),
             helper.make_tensor_value_info("A", TensorProto.FLOAT, (16, 3, 3))],
            [helper.make_tensor_value_info(
                "B", TensorProto.FLOAT, (1, 16, 3, 3))],
            value_info=[
                helper.make_tensor_value_info(
                    "Z", TensorProto.FLOAT, (1, 16, 1, 1)),
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
             helper.make_tensor_value_info(
                 "Y", TensorProto.FLOAT, (16, 5, 3, 3)),
             helper.make_tensor_value_info("A", TensorProto.FLOAT, (1, 16, 3, 3))],
            [helper.make_tensor_value_info(
                "B", TensorProto.FLOAT, (1, 16, 3, 3))]
        )
        optimized_model = self._optimized(graph, ["fuse_add_bias_into_conv"])

        assert len(list(optimized_model.graph.node)) == 2
        assert optimized_model.graph.node[0].op_type == 'Conv'
        assert optimized_model.graph.node[1].op_type == 'Add'

    def test_fuse_matmul_add_bias_into_gemm(self):  # type: () -> None
        matmul = helper.make_node("MatMul", ["X", "Y"], ["Z"])
        add = helper.make_node("Add", ["Z", "B"], ["A"])
        graph = helper.make_graph(
            [matmul, add],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (32, 10)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (10, 16)),
             helper.make_tensor_value_info("B", TensorProto.FLOAT, (16,))],
            [helper.make_tensor_value_info("A", TensorProto.FLOAT, (32, 16))]
        )
        optimized_model = self._optimized(graph, ["fuse_matmul_add_bias_into_gemm"])

        assert len(list(optimized_model.graph.node)) == 1
        assert optimized_model.graph.node[0].op_type == "Gemm"

    def test_fuse_matmul_add_bias_into_gemm_2d_bias(self):  # type: () -> None
        matmul = helper.make_node("MatMul", ["X", "Y"], ["Z"])
        add = helper.make_node("Add", ["Z", "B"], ["A"])
        graph = helper.make_graph(
            [matmul, add],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (32, 10)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (10, 16)),
             helper.make_tensor_value_info("B", TensorProto.FLOAT, (1, 16))],
            [helper.make_tensor_value_info("A", TensorProto.FLOAT, (32, 16))]
        )
        optimized_model = self._optimized(graph, ["fuse_matmul_add_bias_into_gemm"])

        assert len(list(optimized_model.graph.node)) == 1
        assert optimized_model.graph.node[0].op_type == "Gemm"

    def test_fuse_matmul_add_bias_into_gemm_2d_bias_same_shape(self):  # type: () -> None
        matmul = helper.make_node("MatMul", ["X", "Y"], ["Z"])
        add = helper.make_node("Add", ["Z", "B"], ["A"])
        graph = helper.make_graph(
            [matmul, add],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (32, 10)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (10, 16)),
             helper.make_tensor_value_info("B", TensorProto.FLOAT, (32, 16))],
            [helper.make_tensor_value_info("A", TensorProto.FLOAT, (32, 16))]
        )
        optimized_model = self._optimized(graph, ["fuse_matmul_add_bias_into_gemm"])

        assert len(list(optimized_model.graph.node)) == 1
        assert optimized_model.graph.node[0].op_type == "Gemm"

    def test_fuse_matmul_add_bias_into_gemm_2d_bias_bcast_no_fuse(self):  # type: () -> None
        matmul = helper.make_node("MatMul", ["X", "Y"], ["Z"])
        add = helper.make_node("Add", ["Z", "B"], ["A"])
        graph = helper.make_graph(
            [matmul, add],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 10)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (10, 16)),
             helper.make_tensor_value_info("B", TensorProto.FLOAT, (16, 16))],
            [helper.make_tensor_value_info("A", TensorProto.FLOAT, (16, 16))]
        )
        optimized_model = self._optimized(graph, ["fuse_matmul_add_bias_into_gemm"])

        assert optimized_model.graph == graph

    def test_fuse_matmul_add_bias_into_gemm_3d_matmul_no_fuse(self):  # type: () -> None
        matmul = helper.make_node("MatMul", ["X", "Y"], ["Z"])
        add = helper.make_node("Add", ["Z", "B"], ["A"])
        graph = helper.make_graph(
            [matmul, add],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3, 4)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (2, 4, 3)),
             helper.make_tensor_value_info("B", TensorProto.FLOAT, (3, 3))],
            [helper.make_tensor_value_info("A", TensorProto.FLOAT, (2, 3, 3))]
        )
        optimized_model = self._optimized(graph, ["fuse_matmul_add_bias_into_gemm"])

        assert optimized_model.graph == graph

    def test_fuse_matmul_add_bias_into_gemm_3d_bias_no_fuse(self):  # type: () -> None
        matmul = helper.make_node("MatMul", ["X", "Y"], ["Z"])
        add = helper.make_node("Add", ["Z", "B"], ["A"])
        graph = helper.make_graph(
            [matmul, add],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (32, 10)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (10, 16)),
             helper.make_tensor_value_info("B", TensorProto.FLOAT, (4, 1, 16))],
            [helper.make_tensor_value_info("A", TensorProto.FLOAT, (32, 16))]
        )
        optimized_model = self._optimized(graph, ["fuse_matmul_add_bias_into_gemm"])

        assert optimized_model.graph == graph

    def test_fuse_matmul_add_bias_into_gemm_multiple_use_no_fuse(self):  # type: () -> None
        matmul = helper.make_node("MatMul", ["X", "Y"], ["Z"])
        identity = helper.make_node("Identity", ["Z"], ["A1"])
        add = helper.make_node("Add", ["Z", "B"], ["A2"])
        graph = helper.make_graph(
            [matmul, add, identity],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (32, 10)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (10, 16)),
             helper.make_tensor_value_info("B", TensorProto.FLOAT, (1, 16))],
            [helper.make_tensor_value_info("A1", TensorProto.FLOAT, (32, 16)),
             helper.make_tensor_value_info("A2", TensorProto.FLOAT, (32, 16))]
        )
        optimized_model = self._optimized(graph, ["fuse_matmul_add_bias_into_gemm"])

        assert optimized_model.graph == graph

    def test_fuse_pad_into_conv(self):  # type: () -> None
        pad = helper.make_node(
            "Pad",
            ["X"],
            ["P"],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, 1, 1]
        )
        conv = helper.make_node("Conv", ["P", "Y"], ["Z"])
        graph = helper.make_graph(
            [pad, conv],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 5, 2, 2)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (16, 5, 3, 3))],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (1, 16, 1, 1))]
        )
        optimized_model = self._optimized(graph, ["fuse_pad_into_conv"])

        assert len(list(optimized_model.graph.node)) == 1
        assert optimized_model.graph.node[0].op_type == "Conv"
        assert optimized_model.graph.node[0].attribute[0].name == "pads"
        assert list(optimized_model.graph.node[0].attribute[0].ints) == [0, 0, 1, 1]

    def test_fuse_pad_into_con_1d(self):  # type: () -> None
        pad = helper.make_node(
            "Pad",
            ["X"],
            ["P"],
            mode="constant",
            pads=[0, 0, 1, 0, 0, 1]
        )
        conv = helper.make_node("Conv", ["P", "Y"], ["Z"])
        graph = helper.make_graph(
            [pad, conv],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 5, 30)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (16, 5, 32))],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (1, 16, 1))]
        )
        optimized_model = self._optimized(graph, ["fuse_pad_into_conv"])

        assert len(list(optimized_model.graph.node)) == 1
        assert optimized_model.graph.node[0].op_type == "Conv"
        assert optimized_model.graph.node[0].attribute[0].name == "pads"
        assert list(optimized_model.graph.node[0].attribute[0].ints) == [1, 1]

    def test_fuse_pad_into_conv_existing_conv_pad(self):  # type: () -> None
        pad = helper.make_node(
            "Pad",
            ["X"],
            ["P"],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, 1, 1]
        )
        conv = helper.make_node(
            "Conv",
            ["P", "Y"],
            ["Z"],
            pads=[1, 1, 0, 0]
        )
        graph = helper.make_graph(
            [pad, conv],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 5, 2, 2)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (16, 5, 4, 4))],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (1, 16, 1, 1))]
        )
        optimized_model = self._optimized(graph, ["fuse_pad_into_conv"])

        assert len(list(optimized_model.graph.node)) == 1
        assert optimized_model.graph.node[0].op_type == "Conv"
        assert optimized_model.graph.node[0].attribute[0].name == "pads"
        assert list(optimized_model.graph.node[0].attribute[0].ints) == [1, 1, 1, 1]

    def test_fuse_pad_into_conv_pad_feature_no_fuse(self):  # type: () -> None
        pad = helper.make_node(
            "Pad",
            ["X"],
            ["P"],
            mode="constant",
            pads=[0, 1, 0, 0, 0, 0, 0, 0]
        )
        conv = helper.make_node("Conv", ["P", "Y"], ["Z"])
        graph = helper.make_graph(
            [pad, conv],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 4, 3, 3)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (16, 5, 3, 3))],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (1, 16, 1, 1))]
        )
        optimized_model = self._optimized(graph, ["fuse_pad_into_conv"])

        assert optimized_model.graph == graph

    def test_fuse_pad_into_conv_negative_pad_no_fuse(self):  # type: () -> None
        pad = helper.make_node(
            "Pad",
            ["X"],
            ["P"],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, -1, -1]
        )
        conv = helper.make_node("Conv", ["P", "Y"], ["Z"])
        graph = helper.make_graph(
            [pad, conv],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 5, 4, 4)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (16, 5, 3, 3))],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (1, 16, 1, 1))]
        )
        optimized_model = self._optimized(graph, ["fuse_pad_into_conv"])

        assert optimized_model.graph == graph

    def test_fuse_pad_into_conv_reflection_pad_no_fuse(self):  # type: () -> None
        pad = helper.make_node(
            "Pad",
            ["X"],
            ["P"],
            mode="reflect",
            pads=[0, 0, 0, 0, 0, 0, 1, 1]
        )
        conv = helper.make_node("Conv", ["P", "Y"], ["Z"])
        graph = helper.make_graph(
            [pad, conv],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 5, 2, 2)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (16, 5, 3, 3))],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (1, 16, 1, 1))]
        )
        optimized_model = self._optimized(graph, ["fuse_pad_into_conv"])

        assert optimized_model.graph == graph

    def test_fuse_consecutive_squeezes(self):  # type: () -> None
        nodes = [helper.make_node("Squeeze", ["X"], ["Y"], axes=[0, 4, 5]),
                 helper.make_node("Squeeze", ["Y"], ["Z"], axes=[0, 3])]
        nodes.extend(self._make_fake_loop_op(
            [helper.make_node("Squeeze", ["_X"], ["_Y"], axes=[0, 4, 5]),
             helper.make_node("Squeeze", ["_Y"], ["_Z2"], axes=[0, 3])],
            [(TensorProto.FLOAT, (1, 1, 2, 3, 1, 1, 1, 1, 8, 9), "X")],
            [(TensorProto.FLOAT, (2, 3, 1, 8, 9), "Z2")]))

        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info(
                "X", TensorProto.FLOAT, (1, 1, 2, 3, 1, 1, 1, 1, 8, 9))],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (2, 3, 1, 8, 9))])
        optimized_model = self._optimized(graph, ["fuse_consecutive_squeezes"])

        # Squeeze, Constant (trip count), Constant (cond), Loop
        assert optimized_model.graph.node[0].op_type == "Squeeze"
        assert list(optimized_model.graph.node[0].attribute[0].ints) == [
            0, 1, 4, 5, 6]
        assert len(list(optimized_model.graph.node)) == 4

    def test_fuse_consecutive_squeezes_default(self):  # type: () -> None
        squeeze1 = helper.make_node("Squeeze", ["X"], ["Y"], axes=[0, 4, 5])
        squeeze2 = helper.make_node("Squeeze", ["Y"], ["Z"], axes=[0, 3])
        squeeze3 = helper.make_node("Squeeze", ["Z"], ["A"], axes=[2])
        nodes = [squeeze1, squeeze2, squeeze3]
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info(
                "X", TensorProto.FLOAT, (1, 1, 2, 3, 1, 1, 1, 1, 8, 9))],
            [helper.make_tensor_value_info("A", TensorProto.FLOAT, (2, 3, 8, 9))])
        optimized_model = self._optimized(graph, ["fuse_consecutive_squeezes"])

        assert optimized_model.graph.node[0].op_type == "Squeeze"
        assert list(optimized_model.graph.node[0].attribute[0].ints) == [
            0, 1, 4, 5, 6, 7]
        assert len(list(optimized_model.graph.node)) == 1

    def test_fuse_consecutive_squeezes_random(self):  # type: () -> None
        x_shape = [1, 1, 1, 3, 4, 1, 6, 1, 1, 9]
        s1_one_indices = [i for i, a in enumerate(x_shape) if a == 1]
        s1_axes = np.random.choice(s1_one_indices, size=np.random.randint(low=1, high=len(s1_one_indices) - 1),
                                   replace=False)
        s2_x_shape = [a for i, a in enumerate(x_shape) if i not in s1_axes]
        s2_one_indices = [i for i, a in enumerate(s2_x_shape) if a == 1]
        s2_axes = s2_one_indices

        squeeze1 = helper.make_node("Squeeze", ["X"], ["Y"], axes=s1_axes)
        squeeze2 = helper.make_node("Squeeze", ["Y"], ["Z"], axes=s2_axes)
        nodes = [squeeze1, squeeze2]
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, x_shape)],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (3, 4, 6, 9))])
        optimized_model = self._optimized(graph, ["fuse_consecutive_squeezes"])

        assert optimized_model.graph.node[0].op_type == "Squeeze"
        assert list(optimized_model.graph.node[0].attribute[0].ints) == [
            0, 1, 2, 5, 7, 8]
        assert len(list(optimized_model.graph.node)) == 1

    def test_fuse_consecutive_squeezes_multi_uses(self):  # type: () -> None
        squeeze1 = helper.make_node("Squeeze", ["X"], ["Y"], axes=[0, 4, 5])
        add = helper.make_node("Add", ["Y", "A"], ["Z2"])
        squeeze2 = helper.make_node("Squeeze", ["Y"], ["Z"], axes=[0, 3])
        graph = helper.make_graph(
            [squeeze1, add, squeeze2],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 1, 2, 3, 1, 1, 1, 1, 8, 9)),
             helper.make_tensor_value_info("A", TensorProto.FLOAT, (1,))],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (2, 3, 1, 8, 9)),
             helper.make_tensor_value_info("Z2", TensorProto.FLOAT, (1, 2, 3, 1, 1, 8, 9))])
        optimized_model = self._optimized(graph, ["fuse_consecutive_squeezes"])

        assert optimized_model.graph.node[0].op_type == "Squeeze"
        assert list(optimized_model.graph.node[0].attribute[0].ints) == [
            0, 4, 5]
        assert optimized_model.graph.node[2].op_type == "Squeeze"
        assert optimized_model.graph.node[2].input == ["X"]
        assert list(optimized_model.graph.node[2].attribute[0].ints) == [
            0, 1, 4, 5, 6]
        assert len(list(optimized_model.graph.node)) == 3

    def test_fuse_consecutive_softmax_log_axis(self):  # type: () -> None
        for axis in range(3):
            softmax = helper.make_node("Softmax", ["X"], ["Y"], axis=axis)
            log = helper.make_node("Log", ["Y"], ["Z"])
            graph = helper.make_graph(
                [softmax, log],
                "test",
                [helper.make_tensor_value_info(
                    "X", TensorProto.FLOAT, (5, 7, 11))],
                [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (5, 7, 11))])
            optimized_model = self._optimized(
                graph, ["fuse_consecutive_log_softmax"])

            assert optimized_model.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
            assert len(optimized_model.graph.output) == 1
            assert len(optimized_model.graph.node) == 1
            assert optimized_model.graph.node[0].op_type == "LogSoftmax"
            assert optimized_model.graph.node[0].attribute[0].name == "axis"
            assert optimized_model.graph.node[0].attribute[0].i == axis

    def test_fuse_consecutive_softmax_log_side_effect(self):  # type: () -> None
        softmax = helper.make_node("Softmax", ["X"], ["Y"], axis=2)
        log = helper.make_node("Log", ["Y"], ["Z"])
        graph = helper.make_graph(
            [softmax, log],
            "test",
            [helper.make_tensor_value_info(
                "X", TensorProto.FLOAT, (5, 7, 11))],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (5, 7, 11)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (5, 7, 11))])
        optimized_model = self._optimized(
            graph, ["fuse_consecutive_log_softmax"])

        assert graph == optimized_model.graph

    def test_fuse_consecutive_softmax_log_multiple_out(self):  # type: () -> None
        softmax = helper.make_node("Softmax", ["X"], ["Y"], axis=2)
        log = helper.make_node("Log", ["Y"], ["Z"])
        exp = helper.make_node("Exp", ["Z"], ["Z1"])
        graph = helper.make_graph(
            [softmax, log, exp],
            "test",
            [helper.make_tensor_value_info(
                "X", TensorProto.FLOAT, (5, 7, 11))],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (5, 7, 11)),
             helper.make_tensor_value_info("Z1", TensorProto.FLOAT, (5, 7, 11))])
        optimized_model = self._optimized(
            graph, ["fuse_consecutive_log_softmax"])

        assert len(optimized_model.graph.output) == 2
        assert len(optimized_model.graph.node) == 2
        assert optimized_model.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
        assert optimized_model.graph.output[1].type.tensor_type.elem_type == TensorProto.FLOAT
        assert optimized_model.graph.node[0].op_type == "LogSoftmax"
        assert optimized_model.graph.node[0].attribute[0].name == "axis"
        assert optimized_model.graph.node[0].attribute[0].i == 2
        assert optimized_model.graph.node[1].op_type == "Exp"

    def test_preserve_value_info(self):  # type: () -> None
        trans1 = helper.make_node("Transpose", ["X"], ["Y"], perm=[1, 0, 2])
        trans2 = helper.make_node("Transpose", ["Y"], ["Z"], perm=[2, 0, 1])
        trans3 = helper.make_node("Transpose", ["Z"], ["A"], perm=[2, 0, 1])
        graph = helper.make_graph(
            [trans1, trans2, trans3],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3, 4))],
            [helper.make_tensor_value_info("A", TensorProto.FLOAT, (2, 4, 3))])
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

    def test_fuse_bn_into_conv_simple(self):  # type: () -> None
        for (tensor_type, np_type) in [(TensorProto.FLOAT, np.float32), (TensorProto.DOUBLE, np.float64)]:
            conv = helper.make_node("Conv", ["X", "W", "B"], ["Y"])
            bn = helper.make_node("BatchNormalization", [
                                  "Y", "scale", "b", "mean", "var"], ["Z"])

            W = np.random.randn(3, 2, 5, 5).astype(np_type) + 2
            B = np.random.randn(3,).astype(np_type) + 2
            scale = np.random.randn(3,).astype(np_type) + 2
            b = np.random.randn(3,).astype(np_type) + 2
            mean = np.random.randn(3,).astype(np_type) + 2
            var = np.abs(np.random.randn(3,).astype(np_type)) + 2

            initializers = [
                helper.make_tensor(name, tensor_type,
                                   npa.shape, npa.tobytes(), raw=True)
                for name, npa in [('W', W), ('B', B), ('scale', scale), ('b', b), ('mean', mean), ('var', var)]
            ]
            graph = helper.make_graph(
                [conv, bn],
                "test",
                [helper.make_tensor_value_info("X", tensor_type, (5, 2, 28, 28)),
                 helper.make_tensor_value_info("W", tensor_type, (3, 2, 5, 5)),
                 helper.make_tensor_value_info("B", tensor_type, (3,)),
                 helper.make_tensor_value_info("scale", tensor_type, (3,)),
                 helper.make_tensor_value_info("b", tensor_type, (3,)),
                 helper.make_tensor_value_info("mean", tensor_type, (3,)),
                 helper.make_tensor_value_info("var", tensor_type, (3,))],
                [helper.make_tensor_value_info(
                    "Z", tensor_type, (5, 3, 24, 24))],
                initializer=initializers,
                value_info=[
                    helper.make_tensor_value_info(
                        "Y", tensor_type, (5, 3, 24, 24))
                ]
            )
            optimized_model = self._optimized(graph, ["fuse_bn_into_conv"])

            self.assertEqual(len(optimized_model.graph.node), 1)
            self.assertEqual(optimized_model.graph.node[0].op_type, 'Conv')
            self.assertEqual(len(optimized_model.graph.initializer), 2)
            new_W = numpy_helper.to_array(optimized_model.graph.initializer[0])
            new_b = numpy_helper.to_array(optimized_model.graph.initializer[1])

            f = scale / np.sqrt(var + 1e-5)
            np.testing.assert_almost_equal((B - mean) * f + b, new_b)
            np.testing.assert_almost_equal(
                W * f[:, np.newaxis, np.newaxis, np.newaxis], new_W)

    def _internal_test_deadend_elimination(self, fixed):  # type: (bool) -> None
        softmax = helper.make_node("Softmax", ["X"], ["Y"], axis=2)
        log = helper.make_node("Log", ["Y"], ["Z"])
        exp = helper.make_node("Exp", ["Z"], ["Z1"])
        exp1 = helper.make_node("Log", ["Z"], ["Z2"])
        exp2 = helper.make_node("Sqrt", ["Z1"], ["Z3"])
        graph = helper.make_graph(
            [softmax, log, exp, exp1, exp2],
            "test",
            [helper.make_tensor_value_info(
                "X", TensorProto.FLOAT, (5, 7, 11))],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (5, 7, 11))])
        optimized_model = self._optimized(
            graph, ["eliminate_deadend"], fixed)
        assert len(optimized_model.graph.output) == 1
        assert len(optimized_model.graph.node) == 2
        assert optimized_model.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
        assert optimized_model.graph.node[0].op_type == "Softmax"
        assert optimized_model.graph.node[0].attribute[0].name == "axis"
        assert optimized_model.graph.node[0].attribute[0].i == 2
        assert optimized_model.graph.node[1].op_type == "Log"

    def test_deadend_elimination_simple(self):  # type: () -> None
        self._internal_test_deadend_elimination(False)

    def test_deadend_elimination_simple_fixed(self):  # type: () -> None
        self._internal_test_deadend_elimination(True)

    def test_eliminate_nop_monotone_argmax_basic_no_node_axis(self):  # type: () -> None
        for node_name in ["Log", "Exp", "Sqrt"]:
            for axis in range(3):
                node = helper.make_node(node_name, ["X"], ["Y"])
                argmax = helper.make_node("ArgMax", ["Y"], ["Z"], axis=axis)
                graph = helper.make_graph(
                    [node, argmax],
                    "test",
                    [helper.make_tensor_value_info(
                        "X", TensorProto.FLOAT, (5, 7, 11))],
                    [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (5, 7, 11))])
                optimized_model = self._optimized(
                    graph, ["eliminate_nop_monotone_argmax"])
                assert len(optimized_model.graph.output) == 1
                assert len(optimized_model.graph.node) == 1
                assert optimized_model.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
                assert optimized_model.graph.node[0].op_type == "ArgMax"
                assert optimized_model.graph.node[0].attribute[0].name == "axis"
                assert optimized_model.graph.node[0].attribute[0].i == axis

    def test_eliminate_nop_monotone_argmax_basic_with_node_axis(self):  # type: () -> None
        for node_name in ["Softmax", "LogSoftmax"]:
            for axis_n in range(3):
                for axis_max in range(3):
                    node = helper.make_node(node_name, ["X"], ["Y"], axis=axis_n)
                    argmax = helper.make_node("ArgMax", ["Y"], ["Z"], axis=axis_max)
                    graph = helper.make_graph(
                        [node, argmax],
                        "test",
                        [helper.make_tensor_value_info(
                            "X", TensorProto.FLOAT, (5, 7, 11))],
                        [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (5, 7, 11))])
                    optimized_model = self._optimized(
                        graph, ["eliminate_nop_monotone_argmax"])
                    if axis_max == axis_n:
                        assert len(optimized_model.graph.output) == 1
                        assert len(optimized_model.graph.node) == 1
                        assert optimized_model.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
                        assert optimized_model.graph.node[0].op_type == "ArgMax"
                        assert optimized_model.graph.node[0].attribute[0].name == "axis"
                        assert optimized_model.graph.node[0].attribute[0].i == axis_max
                    else:
                        assert optimized_model.graph == graph

    def test_eliminate_nop_monotone_argmax_multiple_out(self):  # type: () -> None
        for node_name in ["Log", "Exp", "Sqrt"]:
            for axis in range(3):
                node = helper.make_node(node_name, ["X"], ["Y"])
                node2 = helper.make_node(node_name, ["Y"], ["Z1"])
                argmax = helper.make_node("ArgMax", ["Y"], ["Z"], axis=axis)
                graph = helper.make_graph(
                    [node, node2, argmax],
                    "test",
                    [helper.make_tensor_value_info(
                        "X", TensorProto.FLOAT, (5, 7, 11))],
                    [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (5, 7, 11)),
                     helper.make_tensor_value_info("Z1", TensorProto.FLOAT, (5, 7, 11))])
                optimized_model = self._optimized(
                    graph, ["eliminate_nop_monotone_argmax"])
                assert optimized_model.graph == graph

    def test_eliminate_nop_monotone_argmax_consecutive(self):  # type: () -> None
        def _assertion(graph, optimized_model, axis_aligned, true_axis):  # type: (GraphProto, ModelProto, bool, int) -> None
            if axis_aligned:
                assert len(optimized_model.graph.output) == 1
                assert len(optimized_model.graph.node) == 1
                assert optimized_model.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
                assert optimized_model.graph.node[0].op_type == "ArgMax"
                assert optimized_model.graph.node[0].attribute[0].name == "axis"
                assert optimized_model.graph.node[0].attribute[0].i == true_axis
            else:
                assert optimized_model.graph == graph
        # no axis X no axis test
        for node_name_0 in ["Log", "Exp", "Sqrt"]:
            for node_name_1 in ["Log", "Exp", "Sqrt"]:
                for axis in range(3):
                    node = helper.make_node(node_name_0, ["X"], ["Y"])
                    node2 = helper.make_node(node_name_1, ["Y"], ["Y1"])
                    argmax = helper.make_node("ArgMax", ["Y1"], ["Z"], axis=axis)
                    graph = helper.make_graph(
                        [node, node2, argmax],
                        "test",
                        [helper.make_tensor_value_info(
                            "X", TensorProto.FLOAT, (5, 7, 11))],
                        [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (5, 7, 11))])
                    optimized_model = self._optimized(
                        graph, ["eliminate_nop_monotone_argmax"], True)
                    _assertion(graph, optimized_model, True, axis)
        # no axis X axis test
        for node_name_0 in ["Log", "Exp", "Sqrt"]:
            for node_name_1 in ["Softmax", "LogSoftmax"]:
                for axis_0 in range(3):
                    for axis_1 in range(3):
                        node = helper.make_node(node_name_0, ["X"], ["Y"])
                        node2 = helper.make_node(node_name_1, ["Y"], ["Y1"], axis=axis_0)
                        argmax = helper.make_node("ArgMax", ["Y1"], ["Z"], axis=axis_1)
                        graph = helper.make_graph(
                            [node, node2, argmax],
                            "test",
                            [helper.make_tensor_value_info(
                                "X", TensorProto.FLOAT, (5, 7, 11))],
                            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (5, 7, 11))])
                        optimized_model = self._optimized(
                            graph, ["eliminate_nop_monotone_argmax"], True)
                        _assertion(graph, optimized_model, axis_0 == axis_1, axis_1)
        # axis X axis test
        for node_name_0 in ["Softmax", "LogSoftmax"]:
            for node_name_1 in ["Softmax", "LogSoftmax"]:
                for axis_0 in range(3):
                    for axis_1 in range(3):
                        for axis_2 in range(3):
                            node = helper.make_node(node_name_0, ["X"], ["Y"], axis=axis_0)
                            node2 = helper.make_node(node_name_1, ["Y"], ["Y1"], axis=axis_1)
                            argmax = helper.make_node("ArgMax", ["Y1"], ["Z"], axis=axis_2)
                            graph = helper.make_graph(
                                [node, node2, argmax],
                                "test",
                                [helper.make_tensor_value_info(
                                    "X", TensorProto.FLOAT, (5, 7, 11))],
                                [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (5, 7, 11))])
                            optimized_model = self._optimized(
                                graph, ["eliminate_nop_monotone_argmax"], True)
                            if axis_0 == axis_1:  # we can reduce both of the monotonic ops
                                _assertion(graph, optimized_model, axis_1 == axis_2, axis_2)
                            elif axis_1 == axis_2:  # we can reduce one of the monotonic ops
                                assert len(optimized_model.graph.output) == 1
                                assert len(optimized_model.graph.node) == 2
                                assert optimized_model.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
                                assert optimized_model.graph.node[-1].op_type == "ArgMax"
                                assert optimized_model.graph.node[-1].attribute[0].name == "axis"
                                assert optimized_model.graph.node[-1].attribute[0].i == axis_2
                            else:  # we can't reduce anything
                                assert optimized_model.graph == graph

    def test_eliminate_nop_dropout(self):  # type: () -> None
        for ratio in [0.0, 0.5]:
            node = helper.make_node("Dropout", ["X"], ["Y"], ratio=ratio)
            node1 = helper.make_node("Log", ["Y"], ["Z"])
            graph = helper.make_graph(
                [node, node1],
                "test",
                [helper.make_tensor_value_info(
                    "X", TensorProto.FLOAT, (5, 7))],
                [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (5, 7))])
            optimized_model = self._optimized(
                graph, ["eliminate_nop_dropout"], False)

            if ratio > 0.0:
                assert optimized_model.graph == graph
            else:
                assert len(optimized_model.graph.output) == 1
                assert len(optimized_model.graph.node) == 1
                assert optimized_model.graph.node[0].op_type == "Log"

    def test_fuse_reduction_unsqueeze(self):  # type: () -> None
        def _calculate_post_transform_shape(input_shape, reduction_axes, unsqueeze_axes, keepdim):  # type: (Tuple[int, ...], List[int], List[int], bool) -> Tuple[int, ...]
            post_reduce_shape = None
            if keepdim:
                post_reduce_shape = tuple([(x if i not in reduction_axes else 1) for i, x in enumerate(input_shape)])
            else:
                post_reduce_shape = tuple([x for i, x in enumerate(input_shape) if i not in reduction_axes])
            post_unsqueeze_shape = list(post_reduce_shape)
            for ax in unsqueeze_axes:
                post_unsqueeze_shape.insert(ax, 1)
            return tuple(post_unsqueeze_shape)

        for reduction in ["ReduceL1", "ReduceL2", "ReduceLogSum",
                          "ReduceLogSumExp", "ReduceMax", "ReduceMean",
                          "ReduceMin", "ReduceProd", "ReduceSum", "ReduceSumSquare"]:
            for axes1 in [[1], [1, 2], [2]]:
                for axes2 in [[1], [1, 2], [2]]:
                    for keepdim in [False, True]:
                        input_shape = (5, 7, 9)
                        output_shape = _calculate_post_transform_shape(input_shape, axes1, axes2, keepdim)  # type: Tuple[int, ...]
                        node = helper.make_node(reduction, ["X"], ["Y"], axes=axes1, keepdims=keepdim)
                        node1 = helper.make_node("Unsqueeze", ["Y"], ["Z"], axes=axes2)
                        graph = helper.make_graph(
                            [node, node1],
                            "test",
                            [helper.make_tensor_value_info(
                                "X", TensorProto.FLOAT, input_shape)],
                            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, output_shape)])
                        optimized_model = self._optimized(
                            graph, ["fuse_consecutive_reduce_unsqueeze"], False)

                        if keepdim or axes1 != axes2:
                            assert optimized_model.graph == graph
                        else:
                            assert len(optimized_model.graph.output) == 1
                            assert len(optimized_model.graph.node) == 1
                            assert optimized_model.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
                            assert optimized_model.graph.node[-1].op_type == reduction
                            assert optimized_model.graph.node[-1].attribute[0].name == "axes"
                            assert optimized_model.graph.node[-1].attribute[0].ints == axes1
                            optimized_output_shape = tuple(x.dim_value for x in optimized_model.graph.output[0].type.tensor_type.shape.dim)
                            assert optimized_output_shape == output_shape


if __name__ == '__main__':
    unittest.main()
