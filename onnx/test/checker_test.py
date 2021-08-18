# SPDX-License-Identifier: Apache-2.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import unittest

from typing import Sequence, Text
import numpy as np  # type: ignore

from onnx import checker, helper, shape_inference
from onnx import TensorProto, GraphProto, SparseTensorProto
import onnx.onnx_cpp2py_export.checker as C
import onnx.defs


class TestChecker(unittest.TestCase):
    @property
    def _sample_float_tensor(self):  # type: () -> TensorProto
        np_array = np.random.randn(2, 3).astype(np.float32)
        return helper.make_tensor(
            name='test',
            data_type=TensorProto.FLOAT,
            dims=(2, 3),
            vals=np_array.reshape(6).tolist()
        )

    def make_sparse(self,
                    shape,  # type: Sequence[int]
                    values,  # type: Sequence[int]
                    indices_shape,  # type: Sequence[int]
                    indices,  # type: Sequence[int]
                    name='spval'  # type: Text
                    ):  # type: (...) -> SparseTensorProto
        sparse = SparseTensorProto()
        sparse.dims.extend(shape)
        nnz = len(values)

        sparse.values.CopyFrom(helper.make_tensor(name, TensorProto.INT64, (nnz,), values))
        sparse.indices.CopyFrom(helper.make_tensor('spind', TensorProto.INT64, indices_shape, indices))
        return sparse

    def test_check_node(self):  # type: () -> None
        node = helper.make_node(
            "Relu", ["X"], ["Y"], name="test")

        checker.check_node(node)

    def test_check_node_input_marked_optional(self):  # type: () -> None
        # GivenTensorFill's input is marked optional, hence it is used in this test.
        node = helper.make_node(
            "GivenTensorFill", [], ["Y"], name="test")
        checker.check_node(node)

        # Explicitly pass the empty string as optional
        node = helper.make_node(
            "GivenTensorFill", [""], ["Y"], name="test")

        # Input of RELU is not optional
        node = helper.make_node(
            "Relu", [""], ["Y"], name="test")
        self.assertRaises(checker.ValidationError, checker.check_node, node)

    def test_check_graph_ir_version_3(self):  # type: () -> None
        ctx = C.CheckerContext()
        ctx.ir_version = 3
        ctx.opset_imports = {'': onnx.defs.onnx_opset_version()}

        def check_ir_version_3(g):   # type: (GraphProto) -> None
            checker.check_graph(g, ctx)

        node = helper.make_node(
            "Relu", ["X"], ["Y"], name="test")
        graph = helper.make_graph(
            [node],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])])
        check_ir_version_3(graph)

        graph.initializer.extend([self._sample_float_tensor])

        graph.initializer[0].name = 'no-exist'

        self.assertRaises(checker.ValidationError, check_ir_version_3, graph)

        graph.initializer[0].name = 'X'
        check_ir_version_3(graph)

    def test_check_graph(self):  # type: () -> None
        node = helper.make_node(
            "Relu", ["X"], ["Y"], name="test")
        graph = helper.make_graph(
            [node],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])])
        checker.check_graph(graph)

        graph.initializer.extend([self._sample_float_tensor])

        graph.initializer[0].name = 'no-exist'
        checker.check_graph(graph)

        graph.initializer[0].name = 'X'
        checker.check_graph(graph)

    def test_check_graph_empty_initializer_name(self):  # type: () -> None
        node = helper.make_node(
            "Relu", ["X"], ["Y"], name="test")
        graph = helper.make_graph(
            [node],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])])
        checker.check_graph(graph)

        # Supply no name for the initializer
        graph.initializer.extend([self._sample_float_tensor])
        graph.initializer[0].name = ''
        self.assertRaises(checker.ValidationError, checker.check_graph, graph)

    def test_check_graph_empty_sparse_initializer_name(self):  # type: () -> None
        node = helper.make_node(
            "Relu", ["X"], ["Y"], name="test")
        graph = helper.make_graph(
            [node],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])])
        checker.check_graph(graph)

        # Supply no name for the sparse_initializer
        sparse = self.make_sparse([100], [13, 17, 19], [3], [9, 27, 81], '')
        graph.sparse_initializer.extend([sparse])
        self.assertRaises(checker.ValidationError, checker.check_graph, graph)

    def test_check_graph_duplicate_init_names(self):  # type: () -> None
        node = helper.make_node(
            "Relu", ["X"], ["Y"], name="test")
        graph = helper.make_graph(
            [node],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])])
        checker.check_graph(graph)

        graph.initializer.extend([self._sample_float_tensor])
        graph.initializer[0].name = 'X'

        # Add sparse initializer with the same name as above
        sparse = self.make_sparse([100], [13, 17, 19], [3], [9, 27, 81], 'X')
        graph.sparse_initializer.extend([sparse])
        self.assertRaises(checker.ValidationError, checker.check_graph, graph)

    def test_check_graph_optional_input(self):  # type: () -> None
        # GivenTensorFill's input is marked optional, hence it is used in this test.
        node = helper.make_node(
            "GivenTensorFill", [""], ["Y"], name="test")
        graph = helper.make_graph(
            [node],
            "test",
            [],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])])
        checker.check_graph(graph)

    def test_check_graph_ssa(self):  # type: () -> None
        relu1 = helper.make_node(
            "Relu", ["X"], ["Z"], name="relu1")
        relu2 = helper.make_node(
            "Relu", ["Y"], ["Z"], name="relu2")

        graph = helper.make_graph(
            [relu1, relu2],
            "test",
            inputs=[
                helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2]),
                helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])
            ],
            outputs=[
                helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 2])
            ]
        )
        self.assertRaises(checker.ValidationError, checker.check_graph, graph)

    def test_check_graph_topologically_sorted(self):  # type: () -> None
        n1 = helper.make_node(
            "Scale", ["X"], ["Y"], scale=2., name="n1")
        n2 = helper.make_node(
            "Scale", ["Y"], ["Z"], scale=3., name="n2")

        graph = helper.make_graph(
            [n2, n1],
            "test",
            inputs=[
                helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])
            ],
            outputs=[
                helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 2])
            ]
        )
        self.assertRaises(checker.ValidationError, checker.check_graph, graph)

    def test_check_model(self):  # type: () -> None
        node = helper.make_node(
            "Relu", ["X"], ["Y"], name="test")
        graph = helper.make_graph(
            [node],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])])
        model = helper.make_model(graph, producer_name='test')

        checker.check_model(model)

    def test_check_serialized_model(self):  # type: () -> None
        node = helper.make_node(
            "Relu", ["X"], ["Y"], name="test")
        graph = helper.make_graph(
            [node],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])])
        model = helper.make_model(graph, producer_name='test')

        checker.check_model(model.SerializeToString())

    def test_check_old_model(self):  # type: () -> None
        node = helper.make_node(
            "Pad", ["X"], ["Y"], paddings=(0, 0, 0, 0))
        graph = helper.make_graph(
            [node],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])])
        onnx_id = helper.make_opsetid("", 1)
        model = helper.make_model(graph, producer_name='test', opset_imports=[onnx_id])

        checker.check_model(model)

    def test_check_tensor(self):  # type: () -> None
        tensor = self._sample_float_tensor
        checker.check_tensor(tensor)

        tensor.raw_data = np.random.randn(2, 3).astype(np.float32).tobytes()
        self.assertRaises(checker.ValidationError, checker.check_tensor, tensor)

    def test_check_string_tensor(self):  # type: () -> None
        tensor = TensorProto()
        tensor.data_type = TensorProto.STRING
        tensor.dims.append(1)
        tensor.string_data.append('Test'.encode('utf-8'))
        checker.check_tensor(tensor)

        del tensor.string_data[:]
        tensor.raw_data = 'Test'.encode('utf-8')
        # string data should not be stored in raw_data field
        self.assertRaises(checker.ValidationError, checker.check_tensor, tensor)

    def test_check_tensor_mismatched_field(self):  # type: () -> None
        tensor = self._sample_float_tensor
        tensor.data_type = TensorProto.INT32
        self.assertRaises(checker.ValidationError, checker.check_tensor, tensor)

    def test_nested_graph(self):  # type: () -> None
        n1 = helper.make_node(
            "Scale", ["X"], ["Y"], scale=2., name="n1")
        n2 = helper.make_node(
            "Scale", ["Y"], ["Z"], scale=3., name="n2")

        graph = helper.make_graph(
            [n1, n2],
            "nested",
            inputs=[
                helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])
            ],
            outputs=[
                helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 2])
            ]
        )

        i1 = helper.make_node(
            "If", ["cond"], ["Z"], then_branch=graph, else_branch=graph)

        graph = helper.make_graph(
            [i1],
            "test",
            inputs=[
                helper.make_tensor_value_info("cond", TensorProto.BOOL, [1]),
                helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])
            ],
            outputs=[helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 2])],
        )

        checker.check_graph(graph)
        #self.assertRaises(checker.ValidationError, checker.check_graph, graph)

    def test_nested_graph_without_subgraph_input_shape(self):  # type: () -> None
        n1 = helper.make_node(
            "Scale", ["X"], ["Y"], scale=2., name="n1")
        n2 = helper.make_node(
            "Scale", ["Y"], ["Z"], scale=3., name="n2")

        input_x = onnx.ValueInfoProto()
        input_x.name = "X"
        graph = helper.make_graph(
            [n1, n2],
            "nested",
            inputs=[
                input_x
            ],
            outputs=[
                helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 2])
            ]
        )

        i1 = helper.make_node(
            "If", ["cond"], ["Z"], then_branch=graph, else_branch=graph)

        graph = helper.make_graph(
            [i1],
            "test",
            inputs=[
                helper.make_tensor_value_info("cond", TensorProto.BOOL, [1]),
                helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])
            ],
            outputs=[helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 2])],
        )

        checker.check_graph(graph)

    @property
    def _sample_0_elem_tensor(self):  # type: () -> TensorProto
        np_array = np.random.randn(0, 3).astype(np.float32)
        return helper.make_tensor(
            name='test',
            data_type=TensorProto.FLOAT,
            dims=(0, 3),
            vals=np_array.reshape(0).tolist()
        )

    def test_check_tensor_zero_elem(self):  # type: () -> None
        tensor = self._sample_0_elem_tensor
        checker.check_tensor(tensor)

    def test_check_removed_experimental_op(self):  # type: () -> None
        node = helper.make_node(
            "ConstantFill", [], ["Y"], name="test", shape=[1, 2])
        checker.check_node(node)

    def test_skip_schema_check_on_non_standard_domain(self):  # type: () -> None
        node = helper.make_node(
            "NonExistOp", ["X"], ["Y"], name="test", domain="test.domain")
        graph = helper.make_graph(
            [node],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])])
        onnx_id = helper.make_opsetid("test.domain", 1)
        model = helper.make_model(graph, producer_name='test',
                                  opset_imports=[onnx_id])
        checker.check_model(model)

    def test_check_sparse_tensor(self):  # type: () -> None
        sparse = self.make_sparse([100], [13, 17, 19], [3], [9, 27, 81])
        checker.check_sparse_tensor(sparse)

    def test_check_sparse_tensor_invalid_index(self):  # type: () -> None
        # index value 181 is out-of-range
        sparse = self.make_sparse([100], [13, 17, 19], [3], [9, 27, 181])
        self.assertRaises(checker.ValidationError, checker.check_sparse_tensor, sparse)

    def test_check_sparse_tensor_unordered(self):  # type: () -> None
        # index values are not in sorted order
        sparse = self.make_sparse([100], [13, 17, 19], [3], [27, 9, 81])
        self.assertRaises(checker.ValidationError, checker.check_sparse_tensor, sparse)

    def test_check_sparse_tensor_coo_format(self):  # type: () -> None
        sparse = self.make_sparse([10, 10], [13, 17, 19], [3, 2], [0, 9, 2, 7, 8, 1])
        checker.check_sparse_tensor(sparse)

    def test_check_sparse_tensor_coo_format_invalid_index(self):  # type: () -> None
        sparse = self.make_sparse([10, 10], [13, 17, 19], [3, 2], [0, 9, 0, 27, 8, 1])
        self.assertRaises(checker.ValidationError, checker.check_sparse_tensor, sparse)

    def test_check_sparse_tensor_coo_format_invalid_shape(self):  # type: () -> None
        sparse = self.make_sparse([10, 10], [13, 17, 19], [2, 3], [0, 9, 2, 7, 8, 1])
        self.assertRaises(checker.ValidationError, checker.check_sparse_tensor, sparse)

    def test_check_sparse_tensor_coo_format_invalid_dim2(self):  # type: () -> None
        sparse = self.make_sparse([10, 10], [13, 17, 19], [3, 1], [0, 1, 2])
        self.assertRaises(checker.ValidationError, checker.check_sparse_tensor, sparse)

    def test_check_sparse_matmul(self):  # type: () -> None
        M = 5
        N = 10
        # Create ValueInfoProto for input X of shape [N]
        X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [N])
        # Create a [M,N] sparse-matrix constant C
        sparse_tensor = self.make_sparse([M, N], [2, 3, 1], [3], [3, 11, 37])
        node1 = helper.make_node('Constant', [], ['C'], sparse_value=sparse_tensor)
        # Create ValueInfoProto for output Y of shape [M]
        Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [M])
        # Compute Y = C X
        node2 = helper.make_node('MatMul', ['C', 'X'], ['Y'])
        # create graph
        graph = helper.make_graph([node1, node2], "sparse_matmul", [X], [Y])
        # check graph
        checker.check_graph(graph)

    def test_check_model_unsupported_input_type(self):  # type: () -> None
        N = 10
        X = helper.make_tensor_value_info('X', TensorProto.BOOL, [N])
        Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [N])
        Z = helper.make_tensor_value_info('Z', TensorProto.FLOAT, [N])
        onnx_id = helper.make_opsetid("", 6)
        node = helper.make_node('Add', ['X', 'Y'], ['Z'])
        graph = helper.make_graph([node], "test_add_input", [X, Y], [Z])
        model = helper.make_model(graph, producer_name='test', opset_imports=[onnx_id])
        self.assertRaises(shape_inference.InferenceError, checker.check_model, model, True)

    def test_check_model_inconsistent_type(self):  # type: () -> None
        N = 10
        X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [N])
        Y = helper.make_tensor_value_info('Y', TensorProto.INT32, [N])
        Z = helper.make_tensor_value_info('Z', TensorProto.FLOAT, [N])
        onnx_id = helper.make_opsetid("", 6)
        node = helper.make_node('Add', ['X', 'Y'], ['Z'])
        graph = helper.make_graph([node], "test_add_input", [X, Y], [Z])
        model = helper.make_model(graph, producer_name='test', opset_imports=[onnx_id])
        self.assertRaises(shape_inference.InferenceError, checker.check_model, model, True)

    def test_check_model_unsupported_output_type(self):  # type: () -> None
        N = 10
        X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [N])
        Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [N])
        Z = helper.make_tensor_value_info('Z', TensorProto.BOOL, [N])
        onnx_id = helper.make_opsetid("", 6)
        node = helper.make_node('Add', ['X', 'Y'], ['Z'])
        graph = helper.make_graph([node], "test_add_input", [X, Y], [Z])
        model = helper.make_model(graph, producer_name='test', opset_imports=[onnx_id])
        self.assertRaises(shape_inference.InferenceError, checker.check_model, model, True)


if __name__ == '__main__':
    unittest.main()
