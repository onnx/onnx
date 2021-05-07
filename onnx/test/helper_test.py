# SPDX-License-Identifier: Apache-2.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import random

import numpy as np  # type: ignore

from onnx import helper, defs, numpy_helper, checker
from onnx import AttributeProto, TensorProto, GraphProto, ModelProto, OptionalProto, TypeProto, SequenceProto
from typing import Text, Any, List, Tuple

import unittest


class TestHelperAttributeFunctions(unittest.TestCase):

    def test_attr_float(self):  # type: () -> None
        # float
        attr = helper.make_attribute("float", 1.)
        self.assertEqual(attr.name, "float")
        self.assertEqual(attr.f, 1.)
        checker.check_attribute(attr)
        # float with scientific
        attr = helper.make_attribute("float", 1e10)
        self.assertEqual(attr.name, "float")
        self.assertEqual(attr.f, 1e10)
        checker.check_attribute(attr)

    def test_attr_int(self):  # type: () -> None
        # integer
        attr = helper.make_attribute("int", 3)
        self.assertEqual(attr.name, "int")
        self.assertEqual(attr.i, 3)
        checker.check_attribute(attr)
        # long integer
        attr = helper.make_attribute("int", 5)
        self.assertEqual(attr.name, "int")
        self.assertEqual(attr.i, 5)
        checker.check_attribute(attr)
        # octinteger
        attr = helper.make_attribute("int", 0o1701)
        self.assertEqual(attr.name, "int")
        self.assertEqual(attr.i, 0o1701)
        checker.check_attribute(attr)
        # hexinteger
        attr = helper.make_attribute("int", 0x1701)
        self.assertEqual(attr.name, "int")
        self.assertEqual(attr.i, 0x1701)
        checker.check_attribute(attr)

    def test_attr_doc_string(self):  # type: () -> None
        attr = helper.make_attribute("a", "value")
        self.assertEqual(attr.name, "a")
        self.assertEqual(attr.doc_string, "")
        attr = helper.make_attribute("a", "value", "doc")
        self.assertEqual(attr.name, "a")
        self.assertEqual(attr.doc_string, "doc")

    def test_attr_string(self):  # type: () -> None
        # bytes
        attr = helper.make_attribute("str", b"test")
        self.assertEqual(attr.name, "str")
        self.assertEqual(attr.s, b"test")
        checker.check_attribute(attr)
        # unspecified
        attr = helper.make_attribute("str", "test")
        self.assertEqual(attr.name, "str")
        self.assertEqual(attr.s, b"test")
        checker.check_attribute(attr)
        # unicode
        attr = helper.make_attribute("str", u"test")
        self.assertEqual(attr.name, "str")
        self.assertEqual(attr.s, b"test")
        checker.check_attribute(attr)
        # empty str
        attr = helper.make_attribute("str", "")
        self.assertEqual(attr.name, "str")
        self.assertEqual(helper.get_attribute_value(attr), b"")
        checker.check_attribute(attr)

    def test_attr_repeated_float(self):  # type: () -> None
        attr = helper.make_attribute("floats", [1.0, 2.0])
        self.assertEqual(attr.name, "floats")
        self.assertEqual(list(attr.floats), [1.0, 2.0])
        checker.check_attribute(attr)

    def test_attr_repeated_int(self):  # type: () -> None
        attr = helper.make_attribute("ints", [1, 2])
        self.assertEqual(attr.name, "ints")
        self.assertEqual(list(attr.ints), [1, 2])
        checker.check_attribute(attr)

    def test_attr_repeated_mixed_floats_and_ints(self):  # type: () -> None
        attr = helper.make_attribute("mixed", [1, 2, 3.0, 4.5])
        self.assertEqual(attr.name, "mixed")
        self.assertEqual(list(attr.floats), [1.0, 2.0, 3.0, 4.5])
        checker.check_attribute(attr)

    def test_attr_repeated_str(self):  # type: () -> None
        attr = helper.make_attribute("strings", ["str1", "str2"])
        self.assertEqual(attr.name, "strings")
        self.assertEqual(list(attr.strings), [b"str1", b"str2"])
        checker.check_attribute(attr)

    def test_attr_repeated_tensor_proto(self):  # type: () -> None
        tensors = [
            helper.make_tensor(
                name='a',
                data_type=TensorProto.FLOAT,
                dims=(1,),
                vals=np.ones(1).tolist()
            ),
            helper.make_tensor(
                name='b',
                data_type=TensorProto.FLOAT,
                dims=(1,),
                vals=np.ones(1).tolist()
            )]
        attr = helper.make_attribute("tensors", tensors)
        self.assertEqual(attr.name, "tensors")
        self.assertEqual(list(attr.tensors), tensors)
        checker.check_attribute(attr)

    def test_attr_repeated_graph_proto(self):  # type: () -> None
        graphs = [GraphProto(), GraphProto()]
        graphs[0].name = "a"
        graphs[1].name = "b"
        attr = helper.make_attribute("graphs", graphs)
        self.assertEqual(attr.name, "graphs")
        self.assertEqual(list(attr.graphs), graphs)
        checker.check_attribute(attr)

    def test_attr_type_proto(self):  # type: () -> None
        # type_proto
        type = TypeProto()
        attr = helper.make_attribute("type_proto", type)
        self.assertEqual(attr.name, "type_proto")
        self.assertEqual(attr.tp, type)
        self.assertEqual(attr.type, AttributeProto.TYPE_PROTO)
        # type_protos
        types = [TypeProto(), TypeProto()]
        attr = helper.make_attribute("type_protos", types)

        self.assertEqual(attr.name, "type_protos")
        self.assertEqual(list(attr.type_protos), types)
        self.assertEqual(attr.type, AttributeProto.TYPE_PROTOS)

    def test_is_attr_legal(self):  # type: () -> None
        # no name, no field
        attr = AttributeProto()
        self.assertRaises(checker.ValidationError, checker.check_attribute, attr)
        # name, but no field
        attr = AttributeProto()
        attr.name = "test"
        self.assertRaises(checker.ValidationError, checker.check_attribute, attr)
        # name, with two fields
        attr = AttributeProto()
        attr.name = "test"
        attr.f = 1.0
        attr.i = 2
        self.assertRaises(checker.ValidationError, checker.check_attribute, attr)

    def test_is_attr_legal_verbose(self):  # type: () -> None

        def _set(attr, type, var, value):  # type: (AttributeProto, AttributeProto.AttributeType, Text, Any) -> None
            setattr(attr, var, value)
            setattr(attr, 'type', type)

        def _extend(attr, type, var, value):  # type: (AttributeProto, AttributeProto.AttributeType, List[Any], Any) -> None
            var.extend(value)
            setattr(attr, 'type', type)

        SET_ATTR = [
            (lambda attr: _set(attr, AttributeProto.FLOAT, "f", 1.0)),
            (lambda attr: _set(attr, AttributeProto.INT, "i", 1)),
            (lambda attr: _set(attr, AttributeProto.STRING, "s", b"str")),
            (lambda attr: _extend(attr, AttributeProto.FLOATS, attr.floats, [1.0, 2.0])),
            (lambda attr: _extend(attr, AttributeProto.INTS, attr.ints, [1, 2])),
            (lambda attr: _extend(attr, AttributeProto.STRINGS, attr.strings, [b"a", b"b"])),
        ]
        # Randomly set one field, and the result should be legal.
        for _i in range(100):
            attr = AttributeProto()
            attr.name = "test"
            random.choice(SET_ATTR)(attr)
            checker.check_attribute(attr)
        # Randomly set two fields, and then ensure helper function catches it.
        for _i in range(100):
            attr = AttributeProto()
            attr.name = "test"
            for func in random.sample(SET_ATTR, 2):
                func(attr)
            self.assertRaises(checker.ValidationError,
                              checker.check_attribute,
                              attr)


class TestHelperNodeFunctions(unittest.TestCase):

    def test_node_no_arg(self):  # type: () -> None
        self.assertTrue(defs.has("Relu"))
        node_def = helper.make_node(
            "Relu", ["X"], ["Y"], name="test")
        self.assertEqual(node_def.op_type, "Relu")
        self.assertEqual(node_def.name, "test")
        self.assertEqual(list(node_def.input), ["X"])
        self.assertEqual(list(node_def.output), ["Y"])

    def test_attr_doc_string(self):  # type: () -> None
        node_def = helper.make_node(
            "Relu", ["X"], ["Y"], name="test", doc_string="doc")
        self.assertEqual(node_def.doc_string, "doc")

    def test_node_with_arg(self):  # type: () -> None
        self.assertTrue(defs.has("Relu"))
        # Note: Relu actually does not need an arg, but let's
        # test it.
        node_def = helper.make_node(
            "Relu", ["X"], ["Y"],
            arg_value=1)
        self.assertEqual(node_def.op_type, "Relu")
        self.assertEqual(list(node_def.input), ["X"])
        self.assertEqual(list(node_def.output), ["Y"])
        self.assertEqual(len(node_def.attribute), 1)
        self.assertEqual(
            node_def.attribute[0],
            helper.make_attribute("arg_value", 1))

    def test_node_domain(self):  # type: () -> None
        node_def = helper.make_node(
            "Relu", ["X"], ["Y"], name="test", doc_string="doc", domain="test.domain")
        self.assertEqual(node_def.domain, "test.domain")

    def test_graph(self):  # type: () -> None
        node_def1 = helper.make_node(
            "Relu", ["X"], ["Y"])
        node_def2 = helper.make_node(
            "Add", ["X", "Y"], ["Z"])
        value_info = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])]
        graph = helper.make_graph(
            [node_def1, node_def2],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 2])],
            doc_string=None,
            value_info=value_info,
        )
        self.assertEqual(graph.name, "test")
        self.assertEqual(len(graph.node), 2)
        self.assertEqual(graph.node[0], node_def1)
        self.assertEqual(graph.node[1], node_def2)
        self.assertEqual(graph.doc_string, "")
        self.assertEqual(graph.value_info[0], value_info[0])

    def test_graph_docstring(self):  # type: () -> None
        graph = helper.make_graph([], "my graph", [], [], None, "my docs")
        self.assertEqual(graph.name, "my graph")
        self.assertEqual(graph.doc_string, "my docs")

    def test_model(self):  # type: () -> None
        node_def = helper.make_node(
            "Relu", ["X"], ["Y"])
        graph_def = helper.make_graph(
            [node_def],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])])
        self.assertRaises(AttributeError, helper.make_model, graph_def, xxx=1)
        model_def = helper.make_model(graph_def, producer_name='test')
        self.assertEqual(model_def.producer_name, 'test')

    def test_model_docstring(self):  # type: () -> None
        graph = helper.make_graph([], "my graph", [], [])
        model_def = helper.make_model(graph, doc_string='test')
        # models may have their own documentation, but don't have a name
        # their name is the domain-qualified name of the underlying graph.
        self.assertFalse(hasattr(model_def, "name"))
        self.assertEqual(model_def.doc_string, 'test')

    def test_model_metadata_props(self):  # type: () -> None
        graph = helper.make_graph([], "my graph", [], [])
        model_def = helper.make_model(graph, doc_string='test')
        helper.set_model_props(model_def, {'Title': 'my graph', 'Keywords': 'test;graph'})
        checker.check_model(model_def)
        helper.set_model_props(model_def, {'Title': 'my graph', 'Keywords': 'test;graph'})
        checker.check_model(model_def)  # helper replaces, so no dupe

        dupe = model_def.metadata_props.add()
        dupe.key = 'Title'
        dupe.value = 'Other'
        self.assertRaises(checker.ValidationError, checker.check_model, model_def)

    def test_model_irversion(self):  # type: () -> None
        def mk_model(opset_versions):  # type: (List[Tuple[Text, int]]) -> ModelProto
            graph = helper.make_graph([], "my graph", [], [])
            return helper.make_model_gen_version(graph, opset_imports=[helper.make_opsetid(*pair) for pair in opset_versions])

        def test(opset_versions, ir_version):  # type: (List[Tuple[Text, int]], int) -> None
            model = mk_model(opset_versions)
            self.assertEqual(model.ir_version, ir_version)
        # opset version 9 requires minimum ir_version 4
        test([("", 9)], 4)
        test([("", 10)], 5)
        test([("", 11)], 6)
        test([("", 12)], 7)
        test([("", 13)], 7)
        test([("", 14)], 7)
        # standard opset can be referred to using empty-string or "ai.onnx"
        test([("ai.onnx", 9)], 4)
        test([("ai.onnx.ml", 2)], 6)
        test([("ai.onnx.training", 1)], 7)
        # helper should pick *max* IR version required from all opsets specified.
        test([("", 10), ("ai.onnx.ml", 2)], 6)
        self.assertRaises(ValueError, mk_model, [("", 100)])


class TestHelperTensorFunctions(unittest.TestCase):

    def test_make_tensor(self):  # type: () -> None
        np_array = np.random.randn(2, 3).astype(np.float32)

        tensor = helper.make_tensor(
            name='test',
            data_type=TensorProto.FLOAT,
            dims=(2, 3),
            vals=np_array.reshape(6).tolist()
        )
        self.assertEqual(tensor.name, 'test')
        np.testing.assert_equal(np_array, numpy_helper.to_array(tensor))

        # use raw_data field to store the data
        tensor = helper.make_tensor(
            name='test',
            data_type=TensorProto.FLOAT,
            dims=(2, 3),
            vals=np_array.reshape(6).tobytes(),
            raw=True,
        )
        np.testing.assert_equal(np_array, numpy_helper.to_array(tensor))

        string_list = list(s.encode('utf-8') for s in ['Amy', 'Billy', 'Cindy', 'David'])
        tensor = helper.make_tensor(
            name='test',
            data_type=TensorProto.STRING,
            dims=(2, 2),
            vals=string_list,
            raw=False
        )
        self.assertEqual(string_list, list(tensor.string_data))

    def test_make_sparse_tensor(self):  # type: () -> None
        values = [1.1, 2.2, 3.3, 4.4, 5.5]
        values_tensor = helper.make_tensor(
            name='test',
            data_type=TensorProto.FLOAT,
            dims=(5, ),
            vals=values
        )
        indices = [1, 3, 5, 7, 9]
        indices_tensor = helper.make_tensor(
            name='test_indices',
            data_type=TensorProto.INT64,
            dims=(5, ),
            vals=indices
        )
        dense_shape = [10]
        sparse = helper.make_sparse_tensor(values_tensor, indices_tensor, dense_shape)
        self.assertEqual(sparse.values, values_tensor)
        self.assertEqual(sparse.indices, indices_tensor)
        self.assertEqual(sparse.dims, dense_shape)

    def test_make_tensor_value_info(self):  # type: () -> None
        vi = helper.make_tensor_value_info('X', TensorProto.FLOAT, (2, 4))
        checker.check_value_info(vi)

        # scalar value
        vi = helper.make_tensor_value_info('Y', TensorProto.FLOAT, ())
        checker.check_value_info(vi)


class TestHelperOptionalFunctions(unittest.TestCase):
    def test_make_optional(self):  # type: () -> None
        values = [1.1, 2.2, 3.3, 4.4, 5.5]
        values_tensor = helper.make_tensor(
            name='test',
            data_type=TensorProto.FLOAT,
            dims=(5,),
            vals=values
        )
        optional = helper.make_optional(
            name='test',
            elem_type=OptionalProto.TENSOR,
            value=values_tensor
        )
        self.assertEqual(optional.name, 'test')
        self.assertEqual(optional.elem_type, OptionalProto.TENSOR)
        self.assertEqual(optional.tensor_value, values_tensor)

        # Test None
        optional_none = helper.make_optional(
            name='test',
            elem_type=OptionalProto.UNDEFINED,
            value=None
        )
        self.assertEqual(optional_none.name, 'test')
        self.assertEqual(optional_none.elem_type, OptionalProto.UNDEFINED)
        self.assertFalse(optional_none.HasField('tensor_value'))

    def test_make_optional_value_info(self):  # type: () -> None
        optional_val_info = helper.make_optional_value_info(
            name='test',
            elem_type=TensorProto.FLOAT,
            shape=[5])

        self.assertEqual(optional_val_info.name, 'test')
        self.assertTrue(optional_val_info.type.optional_type)
        self.assertEqual(optional_val_info.type.optional_type.elem_type.tensor_type.elem_type, OptionalProto.TENSOR)

        # # Test Sequence
        optional_val_info = helper.make_optional_value_info(
            name='test',
            elem_type=SequenceProto.SEQUENCE,
            shape=[5])

        self.assertEqual(optional_val_info.name, 'test')
        self.assertTrue(optional_val_info.type.optional_type)
        sequence_value_info = helper.make_sequence_value_info(
            name='test',
            elem_type=SequenceProto.SEQUENCE,
            shape=[5])
        self.assertEqual(optional_val_info.type.optional_type.elem_type.sequence_type, sequence_value_info.type.sequence_type)


class TestPrintableGraph(unittest.TestCase):

    def test_initializer_with_matching_graph_input(self):  # type: () -> None
        add = helper.make_node("Add", ["X", "Y_Initializer"], ["Z"])
        value_info = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])]

        graph = helper.make_graph(
            [add],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1]),
             helper.make_tensor_value_info("Y_Initializer", TensorProto.FLOAT, [1])],  # inputs
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1])],  # outputs
            [helper.make_tensor("Y_Initializer", TensorProto.FLOAT, [1], [1])],  # initializers
            doc_string=None,
            value_info=value_info
        )

        graph_str = helper.printable_graph(graph)
        self.assertTrue(''') optional inputs with matching initializers (
  %Y_Initializer[FLOAT, 1]''' in graph_str, graph_str)

    def test_initializer_no_matching_graph_input(self):  # type: () -> None
        add = helper.make_node("Add", ["X", "Y_Initializer"], ["Z"])
        value_info = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])]

        graph = helper.make_graph(
            [add],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1])],  # inputs
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1])],  # outputs
            [helper.make_tensor("Y_Initializer", TensorProto.FLOAT, [1], [1])],  # initializers
            doc_string=None,
            value_info=value_info
        )

        graph_str = helper.printable_graph(graph)
        self.assertTrue(''') initializers (
  %Y_Initializer[FLOAT, 1]''' in graph_str, graph_str)


if __name__ == '__main__':
    unittest.main()
