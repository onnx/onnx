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
    Loads a binary protobuf that stores onnx graph

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

    # google protobuf python implementation ParseFromString returns
    # None, cpp implementation returns #bytes decoded
    if decoded is not None and decoded != len(s):
        raise google.protobuf.message.DecodeError(
            "Protobuf decoding consumed too few bytes: {} out of {}".format(
                decoded, len(s)))
    return model
