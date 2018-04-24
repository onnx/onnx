from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from onnx import checker, helper, TensorProto

import numpy as np  # type: ignore

import onnx.optimizer
import unittest


class TestOptimizer(unittest.TestCase):

    def _optimized(self, graph, opts):
        orig_model = helper.make_model(graph, producer_name='onnx-test')
        optimized_model = onnx.optimizer.optimize(orig_model, opts)
        checker.check_model(optimized_model)
        return optimized_model

    # input_types and output_types are lists of triples of (name, type, shape)
    def _make_fake_loop_op(self, body_nodes, input_types, output_types):
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
        loop_outputs = [name for _, _, name in output_types]
        retval_nodes = [
            helper.make_node("Constant", [], ["trip_count"], value=zero),
            helper.make_node("Constant", [], ["condition"], value=true),
            helper.make_node("Loop", loop_inputs, loop_outputs, body=body_graph)
        ]
        return retval_nodes

    # fn is a function that takes a single node as argument
    def _visit_all_nodes_recursive(self, graph, fn):
        for node in graph.node:
            fn(node)
            for attr in node.attribute:
                if attr.g is not None:
                    self._visit_all_nodes_recursive(attr.g, fn)
                if len(attr.graphs):
                    for gr in attr.graphs:
                        self._visit_all_nodes_recursive(gr, fn)

    def test_eliminate_identity_single_use(self):
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
        def check_identity(node):
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

    def test_eliminate_identity_multiple_uses(self):
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

    def test_nop_transpose(self):
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

        def check_transpose(node):
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

    def test_nop_transpose_default(self):
        trans = helper.make_node("Transpose", ["X"], ["Y"])
        graph = helper.make_graph(
            [trans],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (3, 2))])
        optimized_model = self._optimized(graph, ["eliminate_nop_transpose"])

        assert len(list(optimized_model.graph.node)) == 1
        assert optimized_model.graph.node[0].op_type == "Transpose"

    def test_fuse_transpose(self):
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

    def test_fuse_transpose_default(self):
        trans1 = helper.make_node("Transpose", ["X"], ["Y"])
        trans2 = helper.make_node("Transpose", ["Y"], ["Z"])
        graph = helper.make_graph(
            [trans1, trans2],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3, 4))],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (2, 3, 4))])
        optimized_model = self._optimized(graph, ["fuse_consecutive_transposes"])

        assert len(list(optimized_model.graph.node)) == 0

    def test_fuse_transpose_default_no_fuse(self):
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

    def test_fuse_transpose_into_gemm(self):
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

    def test_fuse_add_bias_into_conv_use_weight_shape(self):
        nodes = [helper.make_node("Conv", ["X", "Y"], ["Z"]),
                 helper.make_node("Add", ["Z", "A"], ["B"], broadcast=1, axis=1)]
        nodes.extend(self._make_fake_loop_op(
            [helper.make_node("Conv", ["_X", "_Y"], ["_Z"]),
             helper.make_node("Add", ["_Z", "_A"], ["_B2"], broadcast=1, axis=1)],
            [(TensorProto.FLOAT, (1, 5, 3, 3), "X"),
             (TensorProto.FLOAT, (16, 5, 3, 3), "Y"),
             (TensorProto.FLOAT, (16,), "A")],
            [(TensorProto.FLOAT, (1, 16, 3, 3), "B2")]))
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 5, 3, 3)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (16, 5, 3, 3)),
             helper.make_tensor_value_info("A", TensorProto.FLOAT, (16,))],
            [helper.make_tensor_value_info("B", TensorProto.FLOAT, (1, 16, 3, 3))],
        )
        optimized_model = self._optimized(graph, ["fuse_add_bias_into_conv"])

        # Conv, Constant (trip count), Constant (condition), Loop
        assert len(list(optimized_model.graph.node)) == 4
        assert optimized_model.graph.node[0].op_type == 'Conv'
        assert optimized_model.graph.output[0].name == 'Z'
        # Conv
        assert len(optimized_model.graph.node[3].attribute[0].g.node) == 1
        assert optimized_model.graph.node[3].attribute[0].g.node[0].op_type == 'Conv'
        # Output 1 since 0 is 'cond'
        assert optimized_model.graph.node[3].attribute[0].g.output[1].name == '_Z'

    def test_fuse_add_bias_into_conv_use_weight_shape_with_tile(self):
        conv = helper.make_node("Conv", ["X", "Y"], ["Z"])
        add = helper.make_node("Add", ["Z", "A"], ["B"], broadcast=1, axis=1)
        graph = helper.make_graph(
            [conv, add],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 5, 3, 3)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (16, 5, 3, 3)),
             helper.make_tensor_value_info("A", TensorProto.FLOAT, (1,))],
            [helper.make_tensor_value_info("B", TensorProto.FLOAT, (1, 16, 3, 3))],
        )
        optimized_model = self._optimized(graph, ["fuse_add_bias_into_conv"])

        assert len(list(optimized_model.graph.node)) == 4
        assert len(optimized_model.graph.value_info) == 2
        assert optimized_model.graph.value_info[0].type.tensor_type.elem_type == TensorProto.INT64
        assert optimized_model.graph.value_info[1].type.tensor_type.elem_type == TensorProto.INT64
        assert len(optimized_model.graph.value_info[0].type.tensor_type.shape.dim) == 1
        assert len(optimized_model.graph.value_info[1].type.tensor_type.shape.dim) == 1
        assert optimized_model.graph.node[0].op_type == 'Constant'
        assert optimized_model.graph.node[1].op_type == 'Constant'
        assert optimized_model.graph.node[2].op_type == 'Tile'
        assert optimized_model.graph.node[3].op_type == 'Conv'
        assert optimized_model.graph.output[0].name == 'Z'

    def test_fuse_add_bias_into_conv_use_conv_shape(self):
        sub = helper.make_node("Sub", ["M", "N"], ["Y"])
        conv = helper.make_node("Conv", ["X", "Y"], ["Z"])
        add = helper.make_node("Add", ["Z", "A"], ["B"], broadcast=1, axis=1)
        graph = helper.make_graph(
            [sub, conv, add],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 5, 3, 3)),
             helper.make_tensor_value_info("M", TensorProto.FLOAT, (16, 5, 3, 3)),
             helper.make_tensor_value_info("N", TensorProto.FLOAT, (16, 5, 3, 3)),
             helper.make_tensor_value_info("A", TensorProto.FLOAT, (16,))],
            [helper.make_tensor_value_info("B", TensorProto.FLOAT, (1, 16, 3, 3))],
            value_info=[
                helper.make_tensor_value_info("Z", TensorProto.FLOAT, (1, 16, 3, 3))
            ],
        )
        optimized_model = self._optimized(graph, ["fuse_add_bias_into_conv"])

        assert len(optimized_model.graph.node) == 2
        assert optimized_model.graph.node[0].op_type == 'Sub'
        assert optimized_model.graph.node[1].op_type == 'Conv'
        assert optimized_model.graph.output[0].name == 'Z'
        assert optimized_model.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
        assert len(optimized_model.graph.output[0].type.tensor_type.shape.dim) == 4

    def test_fuse_add_bias_into_conv_use_move_constant(self):
        conv = helper.make_node("Conv", ["X", "Y"], ["Z"])
        constant = helper.make_node("Constant", [], ["A"],
                                    value=helper.make_tensor(
                                        name="bias",
                                        data_type=TensorProto.FLOAT,
                                        dims=(16,),
                                        vals=np.random.randn(16).astype(np.float32).tolist()))
        add = helper.make_node("Add", ["Z", "A"], ["B"], broadcast=1, axis=1)
        graph = helper.make_graph(
            [conv, constant, add],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 5, 3, 3)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (16, 5, 3, 3))],
            [helper.make_tensor_value_info("B", TensorProto.FLOAT, (1, 16, 3, 3))],
            value_info=[
                helper.make_tensor_value_info("A", TensorProto.FLOAT, (16,)),
            ]
        )
        optimized_model = self._optimized(graph, ["fuse_add_bias_into_conv"])

        assert len(optimized_model.graph.node) == 2
        assert optimized_model.graph.node[0].op_type == 'Constant'
        assert optimized_model.graph.node[1].op_type == 'Conv'
        assert optimized_model.graph.output[0].name == 'Z'
        assert optimized_model.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
        assert len(optimized_model.graph.output[0].type.tensor_type.shape.dim) == 4

    def test_fuse_add_bias_into_conv_squeeze_1d_bias_no_fuse(self):
        conv = helper.make_node("Conv", ["X", "Y"], ["Z"])
        add = helper.make_node("Add", ["Z", "A"], ["B"], broadcast=1, axis=3)
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

    def test_fuse_add_bias_into_conv_squeeze_3d_bias_no_fuse(self):
        conv = helper.make_node("Conv", ["X", "Y"], ["Z"])
        add = helper.make_node("Add", ["Z", "A"], ["B"], broadcast=1, axis=1)
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

    def test_fuse_add_bias_into_conv_squeeze_4d_bias_no_fuse(self):
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

    def test_preserve_value_info(self):
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

    def test_split(self):
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

    def test_speculate_split(self):
        def make_if_op(true_nodes, true_outputs, false_nodes, false_outputs, if_outputs):
            true = helper.make_tensor("condition", TensorProto.BOOL, (), [True])
            true_outputs = [helper.make_tensor_value_info(name, TensorProto.FLOAT, (1,)) for name in true_outputs]
            false_outputs = [helper.make_tensor_value_info(name, TensorProto.FLOAT, (1,)) for name in false_outputs]
            true_graph = helper.make_graph(true_nodes, "true_graph", [], true_outputs)
            false_graph = helper.make_graph(false_nodes, "false_graph", [], false_outputs)
            if_inputs = ["condition"]
            retval_nodes = [
                helper.make_node("Constant", [], ["condition"], value=true),
                helper.make_node("If", if_inputs, if_outputs, then_branch=true_graph,
                                 else_branch=false_graph)
            ]
            return retval_nodes

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

        capture_node = onnx.helper.make_node('Captured', [], ['X'])
        transpose_node = onnx.helper.make_node(
            'Transpose',
            inputs=['X'],
            outputs=['transposed']
        )

        if_stmt = make_if_op(
            [capture_node, transpose_node], ['transposed'],
            [capture_node], ['X'],
            ['output'])

        graph = helper.make_graph(
            [node] + if_stmt,
            'test-optimize-split',
            [],
            [helper.make_tensor_value_info('output', TensorProto.FLOAT, (1,))])

        def opt(graph, opts):
            orig_model = helper.make_model(graph, producer_name='onnx-test')
            return onnx.optimizer.optimize(orig_model, opts).graph

        def pretty(g):
            def ftensor(xs):
                return ', '.join([x.name for x in xs])

            def fio(xs):
                return ', '.join(xs)

            def fnode(n):
                text = [
                    '  {} = {}({})'.format(fio(n.output), n.op_type, fio(n.input))
                ]
                for attr in n.attribute:
                    if attr.type == onnx.AttributeProto.GRAPH:
                        text.append('    ' + attr.name + ':')
                        text += ['      ' + x for x in pretty(attr.g).split('\n')]
                return '\n'.join(text)

            template = """
graph({}) {{
{}
    return {}
}}
            """

            return template.format(
                ftensor(g.input),
                '\n'.join(fnode(n) for n in g.node),
                ftensor(g.output))

        init = pretty(opt(graph, ['split_init']))
        split = pretty(opt(graph, ['split_predict']))
        assert('transposed = Transpose(X)' in init)
        assert('transposed = Captured()' in split)


if __name__ == '__main__':
    unittest.main()
