# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import random
import unittest
from typing import Any

import numpy as np

import onnx.onnx2.helper as oh
from onnx import onnx2
from onnx.onnx2 import pychecker


class TestOnnx2Helper(unittest.TestCase):
    def test_make_operatorsetid(self):
        op = oh.make_operatorsetid("", 19)
        self.assertEqual(op.domain, "")
        self.assertEqual(op.version, 19)
        op = oh.make_operatorsetid("ai.onnx.ml", 5)
        self.assertEqual(op.domain, "ai.onnx.ml")
        self.assertEqual(op.version, 5)
        s = str(op)
        self.assertIn('domain: "ai.onnx.ml",', s)

    def test_make_tensor_type_proto(self) -> None:
        proto = oh.make_tensor_type_proto(elem_type=2, shape=None)
        self.assertEqual(proto.tensor_type.elem_type, 2)
        self.assertTrue(proto.tensor_type.shape)
        self.assertFalse(proto.sequence_type)
        s = str(proto)
        self.assertIn("elem_type: 2,", s)

    def test_make_optional_value_info(self) -> None:
        tensor_type_proto = oh.make_tensor_type_proto(elem_type=2, shape=[5])
        tensor_val_into = oh.make_value_info(name="test", type_proto=tensor_type_proto)
        optional_type_proto = oh.make_optional_type_proto(tensor_type_proto)
        optional_val_info = oh.make_value_info(
            name="test", type_proto=optional_type_proto
        )

        self.assertEqual(optional_val_info.name, "test")
        self.assertTrue(optional_val_info.type.optional_type)
        self.assertEqual(
            optional_val_info.type.optional_type.elem_type, tensor_val_into.type
        )

        # Test Sequence
        sequence_type_proto = oh.make_sequence_type_proto(tensor_type_proto)
        optional_type_proto = oh.make_optional_type_proto(sequence_type_proto)
        optional_val_info = oh.make_value_info(
            name="test", type_proto=optional_type_proto
        )

        self.assertEqual(optional_val_info.name, "test")
        self.assertTrue(optional_val_info.type.optional_type)
        sequence_value_info = oh.make_value_info(
            name="test", type_proto=tensor_type_proto
        )
        self.assertEqual(
            optional_val_info.type.optional_type.elem_type.sequence_type.elem_type,
            sequence_value_info.type,
        )

    def test_make_sequence_value_info(self) -> None:
        tensor_type_proto = oh.make_tensor_type_proto(elem_type=2, shape=None)
        sequence_type_proto = oh.make_sequence_type_proto(tensor_type_proto)
        sequence_val_info = oh.make_value_info(
            name="test", type_proto=sequence_type_proto
        )
        sequence_val_info_prim = oh.make_tensor_sequence_value_info(
            name="test", elem_type=2, shape=None
        )

        self.assertEqual(sequence_val_info, sequence_val_info_prim)

    def test_attr_float(self) -> None:
        # float
        attr = oh.make_attribute("float", 1.0)
        self.assertEqual(attr.name, "float")
        self.assertEqual(attr.f, 1.0)
        pychecker.check_attribute(attr)
        # float with scientific
        attr = oh.make_attribute("float", 1e10)
        self.assertEqual(attr.name, "float")
        self.assertEqual(attr.f, 1e10)
        pychecker.check_attribute(attr)

    def test_attr_int(self) -> None:
        # integer
        attr = oh.make_attribute("int", 3)
        self.assertEqual(attr.name, "int")
        self.assertEqual(attr.i, 3)
        pychecker.check_attribute(attr)
        # long integer
        attr = oh.make_attribute("int", 5)
        self.assertEqual(attr.name, "int")
        self.assertEqual(attr.i, 5)
        pychecker.check_attribute(attr)
        # octinteger
        attr = oh.make_attribute("int", 0o1701)
        self.assertEqual(attr.name, "int")
        self.assertEqual(attr.i, 0o1701)
        pychecker.check_attribute(attr)
        # hexinteger
        attr = oh.make_attribute("int", 0x1701)
        self.assertEqual(attr.name, "int")
        self.assertEqual(attr.i, 0x1701)
        pychecker.check_attribute(attr)

    def test_attr_doc_string(self) -> None:
        attr = oh.make_attribute("a", "value")
        self.assertEqual(attr.name, "a")
        self.assertEqual(attr.doc_string, "")
        attr = oh.make_attribute("a", "value", "doc")
        self.assertEqual(attr.name, "a")
        self.assertEqual(attr.doc_string, "doc")

    def test_attr_string(self) -> None:
        # bytes
        attr = oh.make_attribute("str", b"test")
        self.assertEqual(attr.name, "str")
        self.assertEqual(attr.s, b"test")
        pychecker.check_attribute(attr)
        # unspecified
        attr = oh.make_attribute("str", "test")
        self.assertEqual(attr.name, "str")
        self.assertEqual(attr.s, b"test")
        pychecker.check_attribute(attr)
        # unicode
        attr = oh.make_attribute("str", "test")
        self.assertEqual(attr.name, "str")
        self.assertEqual(attr.s, b"test")
        pychecker.check_attribute(attr)

    def test_attr_repeated_float(self) -> None:
        attr = oh.make_attribute("floats", [1.0, 2.0])
        self.assertEqual(attr.name, "floats")
        self.assertEqual(list(attr.floats), [1.0, 2.0])
        pychecker.check_attribute(attr)

    def test_attr_repeated_int(self) -> None:
        attr = oh.make_attribute("ints", [1, 2])
        self.assertEqual(attr.name, "ints")
        self.assertEqual(list(attr.ints), [1, 2])
        pychecker.check_attribute(attr)

    def test_attr_repeated_mixed_floats_and_ints(self) -> None:
        attr = oh.make_attribute("mixed", [1, 2, 3.0, 4.5])
        self.assertEqual(attr.name, "mixed")
        self.assertEqual(list(attr.floats), [1.0, 2.0, 3.0, 4.5])
        pychecker.check_attribute(attr)

    def test_attr_repeated_str(self) -> None:
        attr = oh.make_attribute("strings", ["str1", "str2"])
        self.assertEqual(attr.name, "strings")
        self.assertEqual(list(attr.strings), [b"str1", b"str2"])
        pychecker.check_attribute(attr)

    def test_attr_repeated_tensor_proto(self) -> None:
        tensors = [
            oh.make_tensor(
                name="a", data_type=onnx2.TensorProto.FLOAT, dims=(1,), vals=np.ones(1)
            ),
            oh.make_tensor(
                name="b", data_type=onnx2.TensorProto.FLOAT, dims=(1,), vals=np.ones(1)
            ),
        ]
        attr = oh.make_attribute("tensors", tensors)
        attr_tensors = list(attr.tensors)
        self.assertIsInstance(attr_tensors, list)
        self.assertIsInstance(tensors, list)
        self.assertEqual(attr.name, "tensors")
        self.assertEqual(tensors, attr_tensors)
        pychecker.check_attribute(attr)

    def test_attr_sparse_tensor_proto(self) -> None:
        dense_shape = [3, 3]
        sparse_values = [1.764052391052246, 0.40015721321105957, 0.978738009929657]
        values_tensor = oh.make_tensor(
            name="sparse_values",
            data_type=onnx2.TensorProto.FLOAT,
            dims=[len(sparse_values)],
            vals=np.array(sparse_values).astype(np.float32),
            raw=False,
        )

        linear_indices = [2, 3, 5]
        indices_tensor = oh.make_tensor(
            name="indices",
            data_type=onnx2.TensorProto.INT64,
            dims=[len(linear_indices)],
            vals=np.array(linear_indices).astype(np.int64),
            raw=False,
        )
        sparse_tensor = oh.make_sparse_tensor(
            values_tensor, indices_tensor, dense_shape
        )

        attr = oh.make_attribute("sparse_attr", sparse_tensor)
        self.assertEqual(attr.name, "sparse_attr")
        pychecker.check_sparse_tensor(oh.get_attribute_value(attr))
        pychecker.check_attribute(attr)

    def test_attr_sparse_tensor_repeated_protos(self) -> None:
        dense_shape = [3, 3]
        sparse_values = [1.764052391052246, 0.40015721321105957, 0.978738009929657]
        values_tensor = oh.make_tensor(
            name="sparse_values",
            data_type=onnx2.TensorProto.FLOAT,
            dims=[len(sparse_values)],
            vals=np.array(sparse_values).astype(np.float32),
            raw=False,
        )

        linear_indices = [2, 3, 5]
        indices_tensor = oh.make_tensor(
            name="indices",
            data_type=onnx2.TensorProto.INT64,
            dims=[len(linear_indices)],
            vals=np.array(linear_indices).astype(np.int64),
            raw=False,
        )
        sparse_tensor = oh.make_sparse_tensor(
            values_tensor, indices_tensor, dense_shape
        )

        repeated_sparse = [sparse_tensor, sparse_tensor]
        attr = oh.make_attribute("sparse_attrs", repeated_sparse)
        self.assertEqual(attr.name, "sparse_attrs")
        pychecker.check_attribute(attr)
        for s in oh.get_attribute_value(attr):
            pychecker.check_sparse_tensor(s)

    @unittest.skipIf(True, "not yet implemented")
    def test_attr_repeated_graph_proto(self) -> None:
        graphs = [onnx2.GraphProto(), onnx2.GraphProto()]
        graphs[0].name = "a"
        graphs[1].name = "b"
        attr = oh.make_attribute("graphs", graphs)
        self.assertEqual(attr.name, "graphs")
        self.assertEqual(list(attr.graphs), graphs)
        pychecker.check_attribute(attr)

    def test_attr_empty_list(self) -> None:
        attr = oh.make_attribute("empty", [], attr_type=onnx2.AttributeProto.STRINGS)
        self.assertEqual(int(attr.type), onnx2.AttributeProto.STRINGS)
        self.assertEqual(len(attr.strings), 0)
        self.assertRaises(ValueError, oh.make_attribute, "empty", [])

    def test_attr_mismatch(self) -> None:
        with self.assertRaisesRegex(TypeError, "Inferred attribute type 'FLOAT'"):
            oh.make_attribute("test", 6.4, attr_type=onnx2.AttributeProto.STRING)

    def test_is_attr_legal(self) -> None:
        # no name, no field
        attr = onnx2.AttributeProto()
        self.assertRaises(pychecker.ValidationError, pychecker.check_attribute, attr)
        # name, but no field
        attr = onnx2.AttributeProto()
        attr.name = "test"
        self.assertRaises(pychecker.ValidationError, pychecker.check_attribute, attr)
        # name, with two fields
        attr = onnx2.AttributeProto()
        attr.name = "test"
        attr.f = 1.0
        attr.i = 2
        self.assertRaises(pychecker.ValidationError, pychecker.check_attribute, attr)

    def test_is_attr_legal_verbose(self) -> None:
        def _set(
            attr: onnx2.AttributeProto,
            type_: onnx2.AttributeProto.AttributeType,
            var: str,
            value: Any,
        ) -> None:
            setattr(attr, var, value)
            attr.type = type_

        def _extend(
            attr: onnx2.AttributeProto,
            type_: onnx2.AttributeProto.AttributeType,
            var: list[Any],
            value: Any,
        ) -> None:
            var.extend(value)
            attr.type = type_

        SET_ATTR = [
            (lambda attr: _set(attr, onnx2.AttributeProto.FLOAT, "f", 1.0)),
            (lambda attr: _set(attr, onnx2.AttributeProto.INT, "i", 1)),
            (lambda attr: _set(attr, onnx2.AttributeProto.STRING, "s", b"str")),
            (
                lambda attr: _extend(
                    attr, onnx2.AttributeProto.FLOATS, attr.floats, [1.0, 2.0]
                )
            ),
            (lambda attr: _extend(attr, onnx2.AttributeProto.INTS, attr.ints, [1, 2])),
            (
                lambda attr: _extend(
                    attr, onnx2.AttributeProto.STRINGS, attr.strings, [b"a", b"b"]
                )
            ),
        ]
        # Randomly set one field, and the result should be legal.
        for _i in range(100):
            attr = onnx2.AttributeProto()
            attr.name = "test"
            random.choice(SET_ATTR)(attr)
            pychecker.check_attribute(attr)
        # Randomly set two fields, and then ensure helper function catches it.
        for _i in range(100):
            attr = onnx2.AttributeProto()
            attr.name = "test"
            for func in random.sample(SET_ATTR, 2):
                func(attr)
            self.assertRaises(
                pychecker.ValidationError, pychecker.check_attribute, attr
            )

    def test_node_no_arg(self) -> None:
        node_def = oh.make_node("Relu", ["X"], ["Y"], name="test")
        self.assertEqual(node_def.op_type, "Relu")
        self.assertEqual(node_def.name, "test")
        self.assertEqual(list(node_def.input), ["X"])
        self.assertEqual(list(node_def.output), ["Y"])

    def test_node_with_arg(self) -> None:
        node_def = oh.make_node("Relu", ["X"], ["Y"], arg_value=1)
        self.assertEqual(node_def.op_type, "Relu")
        self.assertEqual(list(node_def.input), ["X"])
        self.assertEqual(list(node_def.output), ["Y"])
        self.assertEqual(len(node_def.attribute), 1)
        self.assertEqual(node_def.attribute[0], oh.make_attribute("arg_value", 1))

    def test_node_domain(self) -> None:
        node_def = oh.make_node(
            "Relu", ["X"], ["Y"], name="test", doc_string="doc", domain="test.domain"
        )
        self.assertEqual(node_def.domain, "test.domain")

    def test_graph(self) -> None:
        node_def1 = oh.make_node("Relu", ["X"], ["Y"])
        node_def2 = oh.make_node("Add", ["X", "Y"], ["Z"])
        value_info = [oh.make_tensor_value_info("Y", onnx2.TensorProto.FLOAT, [1, 2])]
        graph = oh.make_graph(
            [node_def1, node_def2],
            "test",
            [oh.make_tensor_value_info("X", onnx2.TensorProto.FLOAT, [1, 2])],
            [oh.make_tensor_value_info("Z", onnx2.TensorProto.FLOAT, [1, 2])],
            doc_string=None,
            value_info=value_info,
        )
        self.assertEqual(graph.name, "test")
        self.assertEqual(len(graph.node), 2)
        self.assertEqual(graph.node[0], node_def1)
        self.assertEqual(graph.node[1], node_def2)
        self.assertEqual(graph.doc_string, "")
        self.assertEqual(graph.value_info[0], value_info[0])

    def test_graph_docstring(self) -> None:
        graph = oh.make_graph([], "my graph", [], [], None, "my docs")
        self.assertEqual(graph.name, "my graph")
        self.assertEqual(graph.doc_string, "my docs")

    def test_model(self) -> None:
        node_def = oh.make_node("Relu", ["X"], ["Y"])
        graph_def = oh.make_graph(
            [node_def],
            "test",
            [oh.make_tensor_value_info("X", onnx2.TensorProto.FLOAT, [1, 2])],
            [oh.make_tensor_value_info("Y", onnx2.TensorProto.FLOAT, [1, 2])],
        )
        self.assertRaises((AttributeError, TypeError), oh.make_model, graph_def, xxx=1)
        model_def = oh.make_model(graph_def, producer_name="test")
        self.assertEqual(model_def.producer_name, "test")

    def test_model_docstring(self) -> None:
        graph = oh.make_graph([], "my graph", [], [])
        model_def = oh.make_model(graph, doc_string="test")
        # models may have their own documentation, but don't have a name
        # their name is the domain-qualified name of the underlying graph.
        self.assertFalse(hasattr(model_def, "name"))
        self.assertEqual(model_def.doc_string, "test")

    def test_model_metadata_props(self) -> None:
        graph = oh.make_graph([], "my graph", [], [])
        model_def = oh.make_model(graph, doc_string="test")
        oh.set_model_props(model_def, {"Title": "my graph", "Keywords": "test;graph"})
        pychecker.check_model(model_def)
        oh.set_model_props(model_def, {"Title": "my graph", "Keywords": "test;graph"})
        pychecker.check_model(model_def)  # helper replaces, so no dupe

        dupe = model_def.metadata_props.add()
        dupe.key = "Title"
        dupe.value = "Other"
        self.assertRaises(pychecker.ValidationError, pychecker.check_model, model_def)


if __name__ == "__main__":
    unittest.main(verbosity=2)
