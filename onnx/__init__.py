from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .onnx_pb2 import ModelProto

import sys

def load(obj):
    '''
    Loads a binary protobuf that stores onnx graph

    @params
    Takes a file-like object (has to implement fileno that returns a file descriptor)
    or a string containing a file name
    @return ONNX ModelProto object
    '''
    model = ModelProto()
    if isinstance(obj, str) or (sys.version_info[0] == 2 and
                                isinstance(obj, unicode_literals.unicode_or_str)):
        with open(obj, 'rb') as f:
            model.ParseFromString(f.read())
    else:
        model.ParseFromString(obj.read())
    return model
