from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .onnx_pb import *  # noqa
from .version import version as __version__  # noqa

# Import common subpackages so they're available when you 'import onnx'
import onnx.helper  # noqa
import onnx.checker  # noqa
import onnx.defs  # noqa

import google.protobuf.message  # type: ignore


# f should be either readable or a file path
def _load_bytes(f):
    if hasattr(f, 'read') and callable(f.read):
        s = f.read()
    else:
        with open(f, 'rb') as readable:
            s = readable.read()
    return s


# str should be bytes,
# f should be either writable or a file path
def _save_bytes(str, f):
    if hasattr(f, 'write') and callable(f.write):
        f.write(str)
    else:
        with open(f, 'wb') as writable:
            writable.write(str)


def _serialize(proto):
    '''
    Serialize a in-memory proto to bytes

    @params
    proto is a in-memory proto, such as a ModelProto, TensorProto, etc

    @return
    Serialized proto in bytes
    '''
    if isinstance(proto, bytes):
        return proto
    elif hasattr(proto, 'SerializeToString') and callable(proto.SerializeToString):
        return proto.SerializeToString()
    else:
        raise ValueError('No SerializeToString method is detected. '
                         'neither proto is a str.\ntype is {}'.format(type(proto)))


def _deserialize(s, proto):
    '''
    Parse bytes into a in-memory proto

    @params
    s is bytes containing serialized proto
    proto is a in-memory proto object

    @return
    The proto instance filled in by s
    '''
    if not isinstance(s, bytes):
        raise ValueError('Parameter s must be bytes, but got type: {}'.format(type(s)))

    if not (hasattr(proto, 'ParseFromString') and callable(proto.ParseFromString)):
        raise ValueError('No ParseFromString method is detected. '
                         '\ntype is {}'.format(type(proto)))

    decoded = proto.ParseFromString(s)
    if decoded is not None and decoded != len(s):
        raise google.protobuf.message.DecodeError(
            "Protobuf decoding consumed too few bytes: {} out of {}".format(
                decoded, len(s)))
    return proto


def load_model(f, format=None):
    '''
    Loads a serialized ModelProto into memory

    @params
    f can be a file-like object (has "read" function) or a string containing a file name
    format is for future use

    @return
    Loaded in-memory ModelProto
    '''
    s = _load_bytes(f)
    return load_model_from_string(s, format=format)


def load_tensor(f, format=None):
    '''
    Loads a serialized TensorProto into memory

    @params
    f can be a file-like object (has "read" function) or a string containing a file name
    format is for future use

    @return
    Loaded in-memory TensorProto
    '''
    s = _load_bytes(f)
    return load_tensor_from_string(s, format=format)


def load_model_from_string(s, format=None):
    '''
    Loads a binary string (bytes) that contains serialized ModelProto

    @params
    s is a string, which contains serialized ModelProto
    format is for future use

    @return
    Loaded in-memory ModelProto
    '''
    return _deserialize(s, ModelProto())


def load_tensor_from_string(s, format=None):
    '''
    Loads a binary string (bytes) that contains serialized TensorProto

    @params
    s is a string, which contains serialized TensorProto
    format is for future use

    @return
    Loaded in-memory TensorProto
    '''
    return _deserialize(s, TensorProto())


def save_model(proto, f, format=None):
    '''
    Saves the ModelProto to the specified path.

    @params
    proto should be a in-memory ModelProto
    f can be a file-like object (has "write" function) or a string containing a file name
    format is for future use
    '''
    s = _serialize(proto)
    _save_bytes(s, f)


def save_tensor(proto, f):
    '''
    Saves the TensorProto to the specified path.

    @params
    proto should be a in-memory TensorProto
    f can be a file-like object (has "write" function) or a string containing a file name
    format is for future use
    '''
    s = _serialize(proto)
    _save_bytes(s, f)


# For backward compatibility
load = load_model
load_from_string = load_model_from_string
save = save_model
