"""onnx optimizer

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import onnx.onnx_cpp2py_export.optimizer as C

optimize = C.optimize
