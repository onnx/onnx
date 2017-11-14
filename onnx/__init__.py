from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .onnx_pb2 import *  # noqa
from .version import version as __version__  # noqa

# Import common subpackages so they're available when you 'import onnx'
import onnx.helper  # noqa
import onnx.checker  # noqa
import onnx.defs  # noqa

import google.protobuf.message


def load(obj):
    '''
    Loads a binary protobuf that stores onnx model

    @params
    Takes a file-like object (has "read" function)
    or a string containing a file name
    @return ONNX ModelProto object
    '''
    model = ModelProto()
    if hasattr(obj, 'read') and callable(obj.read):
        s = obj.read()
    else:
        with open(obj, 'rb') as f:
            s = f.read()
    decoded = model.ParseFromString(s)
    if decoded != len(s):
        raise google.protobuf.message.DecodeError(
            "Protobuf decoding consumed too few bytes: {} out of {} {}".format(
                decoded, len(s), model))
    return model


def load_from_string(obj):
    '''
    Loads a binary string that stores onnx model

    @params
    Takes a string object containing protobuf
    @return ONNX ModelProto object
    '''
    model = ModelProto()
    model.ParseFromString(obj)
    return model
