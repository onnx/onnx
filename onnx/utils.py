from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import onnx.checker
import onnx.helper
import onnx.optimizer
import onnx.shape_inference

from onnx import ModelProto


def polish_model(model):  # type: (ModelProto) -> ModelProto
    '''
        This function combines several useful utility functions together.
    '''
    onnx.checker.check_model(model)
    onnx.helper.strip_doc_string(model)
    model = onnx.shape_inference.infer_shapes(model)
    model = onnx.optimizer.optimize(model)
    onnx.checker.check_model(model)
    return model
