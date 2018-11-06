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


OpSchema = C.OpSchema


@property  # type: ignore
def _Attribute_default_value(self):  # type: ignore
    attr = AttributeProto()
    attr.ParseFromString(self._default_value)
    return attr


OpSchema.Attribute.default_value = _Attribute_default_value  # type: ignore


def get_functions(domain=ONNX_DOMAIN):  # type: ignore
    function_map = defaultdict(list)  # type: Dict[int, List[FunctionProto]]
    function_byte_map = C.get_all_functions(domain)  # type: ignore
    for function_name, raw_functions in function_byte_map.items():
        for function_bytes in raw_functions:
            function_proto = FunctionProto()
            function_proto.ParseFromString(function_bytes)
            function_map[function_name].append(function_proto)
    return function_map
