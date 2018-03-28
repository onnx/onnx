from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import onnx
import onnx.onnx_cpp2py_export.defs as C


ONNX_DOMAIN = ""


def has(op_type):
    return C.has_schema(op_type)


def get_schema(op_type):
    return C.get_schema(op_type)


def get_all_schemas():
    return C.get_all_schemas()


def get_all_schemas_with_history():
    return C.get_all_schemas_with_history()


def onnx_opset_version():
    return C.schema_version_map()[ONNX_DOMAIN][1]


OpSchema = C.OpSchema


@property
def _Attribute_default_value(self):
    attr = onnx.AttributeProto()
    attr.ParseFromString(self._default_value)
    return attr


OpSchema.Attribute.default_value = _Attribute_default_value
