from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import random

from onnx import helper, defs
from onnx.onnx_pb2 import AttributeProto, TensorProto, GraphProto,ModelProto, IR_VERSION

import unittest


class TestHelperAttributeFunctions(unittest.TestCase):

    def test_attr_float(self):
        # float
        attr = helper.make_attribute("float", 1.)
        self.assertEqual(attr.name, "float")
        self.assertEqual(attr.f, 1.)
        self.assertTrue(helper.is_attribute_legal(attr))
        # float with scientific
        attr = helper.make_attribute("float", 1e10)
        self.assertEqual(attr.name, "float")
        self.assertEqual(attr.f, 1e10)
        self.assertTrue(helper.is_attribute_legal(attr))

    def test_attr_int(self):
        # integer
        attr = helper.make_attribute("int", 3)
        self.assertEqual(attr.name, "int")
        self.assertEqual(attr.i, 3)
        self.assertTrue(helper.is_attribute_legal(attr))
        # long integer
        attr = helper.make_attribute("int", 5)
        self.assertEqual(attr.name, "int")
        self.assertEqual(attr.i, 5)
        self.assertTrue(helper.is_attribute_legal(attr))
        # octinteger
        attr = helper.make_attribute("int", 0o1701)
        self.assertEqual(attr.name, "int")
        self.assertEqual(attr.i, 0o1701)
        self.assertTrue(helper.is_attribute_legal(attr))
        # hexinteger
        attr = helper.make_attribute("int", 0x1701)
        self.assertEqual(attr.name, "int")
        self.assertEqual(attr.i, 0x1701)
        self.assertTrue(helper.is_attribute_legal(attr))

    def test_attr_string(self):
        # bytes
        attr = helper.make_attribute("str", b"test")
        self.assertEqual(attr.name, "str")
        self.assertEqual(attr.s, b"test")
        self.assertTrue(helper.is_attribute_legal(attr))
        # unspecified
        attr = helper.make_attribute("str", "test")
        self.assertEqual(attr.name, "str")
        self.assertEqual(attr.s, b"test")
        self.assertTrue(helper.is_attribute_legal(attr))
        # unicode
        attr = helper.make_attribute("str", u"test")
        self.assertEqual(attr.name, "str")
        self.assertEqual(attr.s, b"test")
        self.assertTrue(helper.is_attribute_legal(attr))

    def test_attr_repeated_float(self):
        attr = helper.make_attribute("floats", [1.0, 2.0])
        self.assertEqual(attr.name, "floats")
        self.assertEqual(list(attr.floats), [1.0, 2.0])
        self.assertTrue(helper.is_attribute_legal(attr))

    def test_attr_repeated_int(self):
        attr = helper.make_attribute("ints", [1, 2])
        self.assertEqual(attr.name, "ints")
        self.assertEqual(list(attr.ints), [1, 2])
        self.assertTrue(helper.is_attribute_legal(attr))

    def test_attr_repeated_str(self):
        attr = helper.make_attribute("strings", ["str1", "str2"])
        self.assertEqual(attr.name, "strings")
        self.assertEqual(list(attr.strings), [b"str1", b"str2"])
        self.assertTrue(helper.is_attribute_legal(attr))

    def test_attr_repeated_tensor_proto(self):
        tensors = [TensorProto(), TensorProto()]
        tensors[0].name = "a"
        tensors[1].name = "b"
        attr = helper.make_attribute("tensors", tensors)
        self.assertEqual(attr.name, "tensors")
        self.assertEqual(list(attr.tensors), tensors)
        self.assertTrue(helper.is_attribute_legal(attr))

    def test_attr_repeated_tensor_proto(self):
        graphs = [GraphProto(), GraphProto()]
        graphs[0].name = "a"
        graphs[1].name = "b"
        attr = helper.make_attribute("graphs", graphs)
        self.assertEqual(attr.name, "graphs")
        self.assertEqual(list(attr.graphs), graphs)
        self.assertTrue(helper.is_attribute_legal(attr))

    def test_is_attr_legal(self):
        # no name, no field
        attr = AttributeProto()
        self.assertFalse(helper.is_attribute_legal(attr))
        # name, but no field
        attr = AttributeProto()
        attr.name = "test"
        self.assertFalse(helper.is_attribute_legal(attr))
        # name, with two fields
        attr = AttributeProto()
        attr.name = "test"
        attr.f = 1.0
        attr.i = 2
        self.assertFalse(helper.is_attribute_legal(attr))

    def test_is_attr_legal_verbose(self):

        ATTR_FUNCTIONS = [
            (lambda attr: setattr(attr, "f", 1.0)),
            (lambda attr: setattr(attr, "i", 1)),
            (lambda attr: setattr(attr, "s", b"str")),
            (lambda attr: attr.floats.extend([1.0, 2.0])),
            (lambda attr: attr.ints.extend([1, 2])),
            (lambda attr: attr.strings.extend([b"a", b"b"])),
            (lambda attr: attr.tensors.extend([TensorProto(), TensorProto()])),
            (lambda attr: attr.graphs.extend([GraphProto(), GraphProto()])),
        ]
        # Randomly set one field, and the result should be legal.
        for i in range(100):
            attr = AttributeProto()
            attr.name = "test"
            random.choice(ATTR_FUNCTIONS)(attr)
            self.assertTrue(helper.is_attribute_legal(attr))
        # Randomly set two fields, and then ensure helper function catches it.
        for i in range(100):
            attr = AttributeProto()
            attr.name = "test"
            for func in random.sample(ATTR_FUNCTIONS, 2):
                func(attr)
            self.assertFalse(helper.is_attribute_legal(attr))









class TestHelperNodeFunctions(unittest.TestCase):

    def test_node_no_arg(self):
        self.assertTrue(defs.has("Relu"))
        node_def = helper.make_node(
            "Relu", ["X"], ["Y"], name="test")
        self.assertEqual(node_def.op_type, "Relu")
        self.assertEqual(node_def.name, "test")
        self.assertEqual(list(node_def.input), ["X"])
        self.assertEqual(list(node_def.output), ["Y"])

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
            ["X"],
            ["Y"])
        self.assertEqual(len(graph.node), 1)
        self.assertEqual(graph.node[0], node_def)

if __name__ == '__main__':
    unittest.main()
