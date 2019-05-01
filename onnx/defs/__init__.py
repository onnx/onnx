from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from onnx import AttributeProto, FunctionProto
import onnx.onnx_cpp2py_export.defs as C

from collections import defaultdict
from typing import List, Dict

ONNX_DOMAIN = ""
ONNX_ML_DOMAIN = 'ai.onnx.ml'


has = C.has_schema
get_schema = C.get_schema
get_all_schemas = C.get_all_schemas
get_all_schemas_with_history = C.get_all_schemas_with_history


def onnx_opset_version():  # type: () -> int
    return C.schema_version_map()[ONNX_DOMAIN][1]


@property  # type: ignore
def _Function_proto(self):  # type: ignore
    func_proto = FunctionProto()
    func_proto.ParseFromString(self._function_body)
    return func_proto


OpSchema = C.OpSchema  # type: ignore
C.OpSchema.function_body = _Function_proto  # type: ignore


@property  # type: ignore
def _Attribute_default_value(self):  # type: ignore
    attr = AttributeProto()
    attr.ParseFromString(self._default_value)
    return attr


OpSchema.Attribute.default_value = _Attribute_default_value  # type: ignore


def get_function_ops():  # type: () -> List[OpSchema]
    schemas = C.get_all_schemas()
    return [schema for schema in schemas if schema.has_function]  # type: ignore
