# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Pure-Python printer for the ONNX text representation.

This mirrors the C++ ``ProtoPrinter`` in ``onnx/defs/printer.cc`` and produces
byte-identical output. The text syntax is experimental and may change; it is the
counterpart of :mod:`onnx.parser`.
"""

from __future__ import annotations

import io
import struct
from typing import TYPE_CHECKING

import onnx

if TYPE_CHECKING:
    from collections.abc import Callable

# Indentation step, matching std::setw increments in onnx/defs/printer.cc.
_INDENT_STEP = 3


def _g(value: float) -> str:
    """Format a float as C++ ``ostream`` does by default (printf %g, precision 6)."""
    return f"{value:g}"


# Mirror PrimitiveTypeNameMap in onnx/defs/parser.h. The empty string maps to
# value 0 (UNDEFINED); any unknown value prints as "undefined".
_PRIMITIVE_TYPE_NAME: dict[int, str] = {
    onnx.TensorProto.FLOAT: "float",
    onnx.TensorProto.UINT8: "uint8",
    onnx.TensorProto.INT8: "int8",
    onnx.TensorProto.UINT16: "uint16",
    onnx.TensorProto.INT16: "int16",
    onnx.TensorProto.INT32: "int32",
    onnx.TensorProto.INT64: "int64",
    onnx.TensorProto.STRING: "string",
    onnx.TensorProto.BOOL: "bool",
    onnx.TensorProto.FLOAT16: "float16",
    onnx.TensorProto.DOUBLE: "double",
    onnx.TensorProto.UINT32: "uint32",
    onnx.TensorProto.UINT64: "uint64",
    onnx.TensorProto.COMPLEX64: "complex64",
    onnx.TensorProto.COMPLEX128: "complex128",
    onnx.TensorProto.BFLOAT16: "bfloat16",
    onnx.TensorProto.FLOAT8E4M3FN: "float8e4m3fn",
    onnx.TensorProto.FLOAT8E4M3FNUZ: "float8e4m3fnuz",
    onnx.TensorProto.FLOAT8E5M2: "float8e5m2",
    onnx.TensorProto.FLOAT8E5M2FNUZ: "float8e5m2fnuz",
    onnx.TensorProto.FLOAT8E8M0: "float8e8m0",
    onnx.TensorProto.UINT4: "uint4",
    onnx.TensorProto.INT4: "int4",
    onnx.TensorProto.FLOAT4E2M1: "float4e2m1",
    onnx.TensorProto.UINT2: "uint2",
    onnx.TensorProto.INT2: "int2",
}

# Mirror AttributeTypeNameMap in onnx/defs/parser.h.
_ATTRIBUTE_TYPE_NAME: dict[int, str] = {
    onnx.AttributeProto.FLOAT: "float",
    onnx.AttributeProto.INT: "int",
    onnx.AttributeProto.STRING: "string",
    onnx.AttributeProto.TENSOR: "tensor",
    onnx.AttributeProto.GRAPH: "graph",
    onnx.AttributeProto.SPARSE_TENSOR: "sparse_tensor",
    onnx.AttributeProto.TYPE_PROTO: "type_proto",
    onnx.AttributeProto.FLOATS: "floats",
    onnx.AttributeProto.INTS: "ints",
    onnx.AttributeProto.STRINGS: "strings",
    onnx.AttributeProto.TENSORS: "tensors",
    onnx.AttributeProto.GRAPHS: "graphs",
    onnx.AttributeProto.SPARSE_TENSORS: "sparse_tensors",
    onnx.AttributeProto.TYPE_PROTOS: "type_protos",
}

# struct code and element formatter for the raw_data types handled by ParseData.
_RAW_DATA_FORMAT: dict[int, tuple[str, Callable[[float], str]]] = {
    onnx.TensorProto.INT32: ("i", str),
    onnx.TensorProto.INT64: ("q", str),
    onnx.TensorProto.FLOAT: ("f", _g),
    onnx.TensorProto.DOUBLE: ("d", _g),
}


def _primitive_type_name(elem_type: int) -> str:
    return _PRIMITIVE_TYPE_NAME.get(elem_type, "undefined")


def _attribute_type_name(attr_type: int) -> str:
    return _ATTRIBUTE_TYPE_NAME.get(attr_type, "undefined")


def _is_valid_identifier(s: str) -> bool:
    if not s:
        return False
    if not (s[0].isascii() and (s[0].isalpha() or s[0] == "_")):
        return False
    return all(c.isascii() and (c.isalnum() or c == "_") for c in s)


class _ProtoPrinter:
    def __init__(self) -> None:
        self._out = io.StringIO()
        self._indent_level = _INDENT_STEP

    def text(self) -> str:
        return self._out.getvalue()

    def _write(self, s: str) -> None:
        self._out.write(s)

    def _indent(self) -> None:
        self._indent_level += _INDENT_STEP

    def _outdent(self) -> None:
        self._indent_level -= _INDENT_STEP

    def _write_quoted(self, s: str) -> None:
        self._write('"')
        for ch in s:
            if ch in ("\\", '"'):
                self._write("\\")
            self._write(ch)
        self._write('"')

    def _write_id(self, s: str) -> None:
        if _is_valid_identifier(s):
            self._write(s)
        else:
            self._write_quoted(s)

    def print_dimension(self, dim: onnx.TensorShapeProto.Dimension) -> None:
        if dim.HasField("dim_value"):
            self._write(str(dim.dim_value))
        elif dim.HasField("dim_param"):
            if _is_valid_identifier(dim.dim_param):
                self._write(dim.dim_param)
            else:
                self._write_quoted(dim.dim_param)
        else:
            self._write("?")

    def print_shape(self, shape: onnx.TensorShapeProto) -> None:
        self._write("[")
        for i, dim in enumerate(shape.dim):
            if i:
                self._write(",")
            self.print_dimension(dim)
        self._write("]")

    def print_tensor_type(self, tensor_type: onnx.TypeProto.Tensor) -> None:
        self._write(_primitive_type_name(tensor_type.elem_type))
        if tensor_type.HasField("shape"):
            if len(tensor_type.shape.dim) > 0:
                self.print_shape(tensor_type.shape)
        else:
            self._write("[]")

    def print_sequence_type(self, seq_type: onnx.TypeProto.Sequence) -> None:
        self._write("seq(")
        self.print_type(seq_type.elem_type)
        self._write(")")

    def print_map_type(self, map_type: onnx.TypeProto.Map) -> None:
        self._write("map(" + _primitive_type_name(map_type.key_type) + ", ")
        self.print_type(map_type.value_type)
        self._write(")")

    def print_optional_type(self, opt_type: onnx.TypeProto.Optional) -> None:
        self._write("optional(")
        self.print_type(opt_type.elem_type)
        self._write(")")

    def print_sparse_tensor_type(
        self, sparse_type: onnx.TypeProto.SparseTensor
    ) -> None:
        self._write("sparse_tensor(" + _primitive_type_name(sparse_type.elem_type))
        if sparse_type.HasField("shape"):
            if len(sparse_type.shape.dim) > 0:
                self.print_shape(sparse_type.shape)
        else:
            self._write("[]")
        self._write(")")

    def print_type(self, type_proto: onnx.TypeProto) -> None:
        if type_proto.HasField("tensor_type"):
            self.print_tensor_type(type_proto.tensor_type)
        elif type_proto.HasField("sequence_type"):
            self.print_sequence_type(type_proto.sequence_type)
        elif type_proto.HasField("map_type"):
            self.print_map_type(type_proto.map_type)
        elif type_proto.HasField("optional_type"):
            self.print_optional_type(type_proto.optional_type)
        elif type_proto.HasField("sparse_tensor_type"):
            self.print_sparse_tensor_type(type_proto.sparse_tensor_type)

    def print_tensor(
        self, tensor: onnx.TensorProto, is_initializer: bool = False
    ) -> None:
        self._write(_primitive_type_name(tensor.data_type))
        if len(tensor.dims) > 0:
            self._write("[" + ",".join(str(d) for d in tensor.dims) + "]")

        if tensor.name:
            self._write(" ")
            self._write_id(tensor.name)
        if is_initializer:
            self._write(" = ")
        # TODO(ONNX): does not yet handle all types
        if (
            tensor.HasField("data_location")
            and tensor.data_location == onnx.TensorProto.EXTERNAL
        ):
            self._write_string_string_entries(tensor.external_data)
        elif tensor.HasField("raw_data"):
            fmt = _RAW_DATA_FORMAT.get(tensor.data_type)
            if fmt is None:
                self._write("...")  # ParseData not instantiated for other types.
            else:
                code, fmt_elem = fmt
                values = struct.unpack(
                    "<" + code * (len(tensor.raw_data) // struct.calcsize(code)),
                    tensor.raw_data,
                )
                self._write(" {" + ",".join(fmt_elem(v) for v in values) + "}")
        else:
            self._write_tensor_typed_data(tensor)

    def _write_tensor_typed_data(self, tensor: onnx.TensorProto) -> None:
        T = onnx.TensorProto
        match tensor.data_type:
            case T.INT8 | T.INT16 | T.INT32 | T.UINT8 | T.UINT16 | T.BOOL:
                self._write(" {" + ",".join(str(v) for v in tensor.int32_data) + "}")
            case T.INT64:
                self._write(" {" + ",".join(str(v) for v in tensor.int64_data) + "}")
            case T.UINT32 | T.UINT64:
                self._write(" {" + ",".join(str(v) for v in tensor.uint64_data) + "}")
            case T.FLOAT:
                self._write(" {" + ",".join(_g(v) for v in tensor.float_data) + "}")
            case T.DOUBLE:
                self._write(" {" + ",".join(_g(v) for v in tensor.double_data) + "}")
            case T.STRING:
                self._write("{")
                sep = ""
                for elt in tensor.string_data:
                    self._write(sep)
                    self._write_quoted(elt.decode("utf-8", "surrogateescape"))
                    sep = ", "
                self._write("}")

    def print_value_info(self, value_info: onnx.ValueInfoProto) -> None:
        self.print_type(value_info.type)
        self._write(" ")
        self._write_id(value_info.name)

    def print_value_info_list(self, vilist) -> None:
        self._write("(")
        for i, vi in enumerate(vilist):
            if i:
                self._write(", ")
            self.print_value_info(vi)
        self._write(")")

    def print_attribute(self, attr: onnx.AttributeProto) -> None:
        A = onnx.AttributeProto
        # Special case of attr-ref:
        if attr.HasField("ref_attr_name"):
            self._write(
                attr.name
                + ": "
                + _attribute_type_name(attr.type)
                + " = @"
                + attr.ref_attr_name
            )
            return
        # General case:
        self._write(attr.name + ": " + _attribute_type_name(attr.type) + " = ")
        match attr.type:
            case A.INT:
                self._write(str(attr.i))
            case A.INTS:
                self._write("[" + ", ".join(str(v) for v in attr.ints) + "]")
            case A.FLOAT:
                self._write(_g(attr.f))
            case A.FLOATS:
                self._write("[" + ", ".join(_g(v) for v in attr.floats) + "]")
            case A.STRING:
                self._write_quoted(attr.s.decode("utf-8", "surrogateescape"))
            case A.STRINGS:
                self._write("[")
                sep = ""
                for elt in attr.strings:
                    self._write(sep)
                    self._write_quoted(elt.decode("utf-8", "surrogateescape"))
                    sep = ", "
                self._write("]")
            case A.GRAPH:
                self._indent()
                self.print_graph(attr.g)
                self._outdent()
            case A.GRAPHS:
                self._indent()
                self._write("[")
                for i, g in enumerate(attr.graphs):
                    if i:
                        self._write(", ")
                    self.print_graph(g)
                self._write("]")
                self._outdent()
            case A.TENSOR:
                self.print_tensor(attr.t)
            case A.TENSORS:
                self._write("[")
                for i, t in enumerate(attr.tensors):
                    if i:
                        self._write(", ")
                    self.print_tensor(t)
                self._write("]")
            case A.TYPE_PROTO:
                self.print_type(attr.tp)
            case A.TYPE_PROTOS:
                self._write("[")
                for i, tp in enumerate(attr.type_protos):
                    if i:
                        self._write(", ")
                    self.print_type(tp)
                self._write("]")

    def print_attr_list(self, attrlist) -> None:
        self._write(" <")
        for i, attr in enumerate(attrlist):
            if i:
                self._write(", ")
            self.print_attribute(attr)
        self._write(">")

    def print_node(self, node: onnx.NodeProto) -> None:
        self._write(" ".rjust(self._indent_level))
        if node.HasField("name"):
            self._write("[")
            self._write_id(node.name)
            self._write("] ")
        # outputs: printIdSet("", ", ", "")
        self._write_id_set("", ", ", "", node.output)
        self._write(" = ")
        if node.domain:
            self._write(node.domain + ".")
        self._write(node.op_type)
        if node.overload:
            self._write(":" + node.overload)
        has_subgraph = any(
            attr.HasField("g") or len(attr.graphs) > 0 for attr in node.attribute
        )
        if (not has_subgraph) and len(node.attribute) > 0:
            self.print_attr_list(node.attribute)
        self._write_id_set(" (", ", ", ")", node.input)
        if has_subgraph and len(node.attribute) > 0:
            self.print_attr_list(node.attribute)
        self._write("\n")

    def _write_id_set(self, open_s: str, sep_s: str, close_s: str, coll) -> None:
        self._write(open_s)
        sep = ""
        for elt in coll:
            self._write(sep)
            self._write_id(elt)
            sep = sep_s
        self._write(close_s)

    def print_node_list(self, nodelist) -> None:
        self._write("{\n")
        for node in nodelist:
            self.print_node(node)
        if self._indent_level > _INDENT_STEP:
            self._write("   ".rjust(self._indent_level - _INDENT_STEP))
        self._write("}")

    def print_graph(self, graph: onnx.GraphProto) -> None:
        self._write_id(graph.name)
        self._write(" ")
        self.print_value_info_list(graph.input)
        self._write(" => ")
        self.print_value_info_list(graph.output)
        self._write(" ")
        if len(graph.initializer) > 0 or len(graph.value_info) > 0:
            self._write("\n" + " ".rjust(self._indent_level) + "<")
            sep = ""
            for init in graph.initializer:
                self._write(sep)
                self.print_tensor(init, is_initializer=True)
                sep = ", "
            for vi in graph.value_info:
                self._write(sep)
                self.print_value_info(vi)
                sep = ", "
            self._write(">\n")
        self.print_node_list(graph.node)

    def _write_opset_id(self, opset: onnx.OperatorSetIdProto) -> None:
        self._write_quoted(opset.domain)
        self._write(" : " + str(opset.version))

    def _write_opset_id_list(self, opsets) -> None:
        self._write("[")
        for i, opset in enumerate(opsets):
            if i:
                self._write(", ")
            self._write_opset_id(opset)
        self._write("]")

    def _write_string_string_entries(self, entries) -> None:
        self._write("[")
        for i, entry in enumerate(entries):
            if i:
                self._write(", ")
            self._write_quoted(entry.key)
            self._write(": ")
            self._write_quoted(entry.value)
        self._write("]")

    def _key_value_pair(self, key: str, value: str, addsep: bool = True) -> None:
        if addsep:
            self._write(",\n")
        self._write(" ".rjust(self._indent_level) + key + ": " + value)

    def _key_value_pair_quoted(self, key: str, value: str) -> None:
        self._write(",\n")
        self._write(" ".rjust(self._indent_level) + key + ": ")
        self._write_quoted(value)

    def print_model(self, model: onnx.ModelProto) -> None:
        self._write("<\n")
        self._key_value_pair("ir_version", str(model.ir_version), addsep=False)
        self._write(",\n" + " ".rjust(self._indent_level) + "opset_import: ")
        self._write_opset_id_list(model.opset_import)
        if model.HasField("producer_name"):
            self._key_value_pair_quoted("producer_name", model.producer_name)
        if model.HasField("producer_version"):
            self._key_value_pair_quoted("producer_version", model.producer_version)
        if model.HasField("domain"):
            self._key_value_pair_quoted("domain", model.domain)
        if model.HasField("model_version"):
            self._key_value_pair("model_version", str(model.model_version))
        if model.HasField("doc_string"):
            self._key_value_pair_quoted("doc_string", model.doc_string)
        if len(model.metadata_props) > 0:
            self._write(",\n" + " ".rjust(self._indent_level) + "metadata_props: ")
            self._write_string_string_entries(model.metadata_props)
        self._write("\n>\n")

        self.print_graph(model.graph)
        for fn in model.functions:
            self._write("\n")
            self.print_function(fn)

    def print_function(self, fn: onnx.FunctionProto) -> None:
        self._write("<\n")
        self._write("  domain: ")
        self._write_quoted(fn.domain)
        self._write(",\n")
        if fn.overload:
            self._write("  overload: ")
            self._write_quoted(fn.overload)
            self._write(",\n")
        self._write("  opset_import: ")
        self._write_opset_id_list_no_space(fn.opset_import)
        self._write("\n>\n")
        self._write_id(fn.name)
        self._write(" ")
        if len(fn.attribute) > 0:
            self._write("<" + ",".join(fn.attribute) + ">")
        self._write_id_set("(", ", ", ")", fn.input)
        self._write(" => ")
        self._write_id_set("(", ", ", ")", fn.output)
        self._write("\n")
        self.print_node_list(fn.node)

    def _write_opset_id_list_no_space(self, opsets) -> None:
        # FunctionProto opset_import uses "[",",","]" (no space separator).
        self._write("[")
        for i, opset in enumerate(opsets):
            if i:
                self._write(",")
            self._write_opset_id(opset)
        self._write("]")


def to_text(
    proto: onnx.ModelProto | onnx.FunctionProto | onnx.GraphProto | onnx.NodeProto,
) -> str:
    printer = _ProtoPrinter()
    if isinstance(proto, onnx.ModelProto):
        printer.print_model(proto)
    elif isinstance(proto, onnx.FunctionProto):
        printer.print_function(proto)
    elif isinstance(proto, onnx.GraphProto):
        printer.print_graph(proto)
    elif isinstance(proto, onnx.NodeProto):
        printer.print_node(proto)
    else:
        raise TypeError("Unsupported argument type.")
    return printer.text()
