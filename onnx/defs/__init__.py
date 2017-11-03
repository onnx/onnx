from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import onnx.onnx_cpp2py_export.defs as C


def has(op_type):
    return C.has_schema(op_type)


def get_schema(op_type):
    return C.get_schema(op_type)


def get_all_schemas():
    return C.get_all_schemas()


OpSchema = C.OpSchema
