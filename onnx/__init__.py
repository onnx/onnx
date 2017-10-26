from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .onnx_ml_pb2 import *

import sys

def load(obj):
    '''
    Loads a binary protobuf that stores onnx graph

    @params
    Takes a file-like object (has "read" function)
    or a string containing a file name
    @return ONNX ModelProto object
    '''
    model = ModelProto()
    if hasattr(obj, 'read') and callable(obj.read):
        model.ParseFromString(obj.read())
    else:
        with open(obj, 'rb') as f:
            model.ParseFromString(f.read())
    return model
