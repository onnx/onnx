from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import random

import numpy as np

from onnx import helper, defs, numpy_helper, checker
from onnx import AttributeProto, TensorProto, GraphProto

import unittest


class TestHelperAttributeFunctions(unittest.TestCase):

    def test_attr_float(self):
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

    def test_attr_int(self):
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

    def test_attr_doc_string(self):
        attr = helper.make_attribute("a", "value")
        self.assertEqual(attr.name, "a")
        self.assertEqual(attr.doc_string, "")
        attr = helper.make_attribute("a", "value", "doc")
        self.assertEqual(attr.name, "a")
        self.assertEqual(attr.doc_string, "doc")

    def test_attr_string(self):
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

    def test_attr_repeated_float(self):
        attr = helper.make_attribute("floats", [1.0, 2.0])
        self.assertEqual(attr.name, "floats")
        self.assertEqual(list(attr.floats), [1.0, 2.0])
        checker.check_attribute(attr)

    def test_attr_repeated_int(self):
        attr = helper.make_attribute("ints", [1, 2])
        self.assertEqual(attr.name, "ints")
        self.assertEqual(list(attr.ints), [1, 2])
        checker.check_attribute(attr)

    def test_attr_repeated_str(self):
        attr = helper.make_attribute("strings", ["str1", "str2"])
        self.assertEqual(attr.name, "strings")
        self.assertEqual(list(attr.strings), [b"str1", b"str2"])
        checker.check_attribute(attr)

    def test_attr_repeated_tensor_proto(self):
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

    def test_attr_repeated_graph_proto(self):
        graphs = [GraphProto(), GraphProto()]
        graphs[0].name = "a"
        graphs[1].name = "b"
        attr = helper.make_attribute("graphs", graphs)
        self.assertEqual(attr.name, "graphs")
        self.assertEqual(list(attr.graphs), graphs)
        checker.check_attribute(attr)

    def test_is_attr_legal(self):
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

    def test_is_attr_legal_verbose(self):

        SET_ATTR = [
            (lambda attr: setattr(attr, "f", 1.0) or
             setattr(attr, 'type', AttributeProto.FLOAT)),
            (lambda attr: setattr(attr, "i", 1) or
             setattr(attr, 'type', AttributeProto.INT)),
            (lambda attr: setattr(attr, "s", b"str") or
             setattr(attr, 'type', AttributeProto.STRING)),
            (lambda attr: attr.floats.extend([1.0, 2.0]) or
             setattr(attr, 'type', AttributeProto.FLOATS)),
            (lambda attr: attr.ints.extend([1, 2]) or
             setattr(attr, 'type', AttributeProto.INTS)),
            (lambda attr: attr.strings.extend([b"a", b"b"]) or
             setattr(attr, 'type', AttributeProto.STRINGS)),
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

    def test_node_no_arg(self):
        self.assertTrue(defs.has("Relu"))
        node_def = helper.make_node(
            "Relu", ["X"], ["Y"], name="test")
        self.assertEqual(node_def.op_type, "Relu")
        self.assertEqual(node_def.name, "test")
        self.assertEqual(list(node_def.input), ["X"])
        self.assertEqual(list(node_def.output), ["Y"])

    def test_attr_doc_string(self):
        node_def = helper.make_node(
            "Relu", ["X"], ["Y"], name="test", doc_string="doc")
        self.assertEqual(node_def.doc_string, "doc")

    def test_node_with_arg(self):
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

    def test_graph(self):
        node_def = helper.make_node(
            "Relu", ["X"], ["Y"])
        graph = helper.make_graph(
            [node_def],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])])
        self.assertEqual(len(graph.node), 1)
        self.assertEqual(graph.node[0], node_def)
        self.assertEqual(graph.doc_string, "")

    def test_graph_docstring(self):
        graph = helper.make_graph([], "my graph", [], [], None, "my docs")
        self.assertEqual(graph.name, "my graph")
        self.assertEqual(graph.doc_string, "my docs")

    def test_model(self):
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

    def test_model_docstring(self):
        graph = helper.make_graph([], "my graph", [], [])
        model_def = helper.make_model(graph, doc_string='test')
        # models may have their own documentation, but don't have a name
        # their name is the domain-qualified name of the underlying graph.
        self.assertFalse(hasattr(model_def, "name"))
        self.assertEqual(model_def.doc_string, 'test')

    def test_model_metadata_props(self):
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


class TestHelperTensorFunctions(unittest.TestCase):

    def test_make_tensor(self):
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

    def test_make_tensor_value_info(self):
        vi = helper.make_tensor_value_info('X', TensorProto.FLOAT, (2, 4))
        checker.check_value_info(vi)

        # scalar value
        vi = helper.make_tensor_value_info('Y', TensorProto.FLOAT, ())
        checker.check_value_info(vi)


if __name__ == '__main__':
    unittest.main()
