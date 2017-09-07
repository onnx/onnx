from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .onnx_pb2 import *
from . import checker, helper
from . import defs

import sys

def load(obj):
    '''
    Loads a binary protobuf that stores onnx graph

    @params
    Takes a file-like object (has to implement fileno that returns a file descriptor)
    or a string containing a file name
    @return ONNX GraphProto object
    '''
    graph = GraphProto()
    if isinstance(obj, str) or (sys.version_info[0] == 2 and isinstance(obj, unicode)):
        with open(obj, 'rb') as f:
            graph.ParseFromString(f.read())
    else:
        graph.ParseFromString(obj.read())
    return graph
