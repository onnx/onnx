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


def _load_string(obj):
    if hasattr(obj, 'read') and callable(obj.read):
        s = obj.read()
    else:
        with open(obj, 'rb') as f:
            s = f.read()
    return s


def _save_string(str, f):
    if hasattr(f, 'write') and callable(f.write):
        f.write(str)
    else:
        with open(f, 'wb') as writable:
            writable.write(str)


def serialize(proto):
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
                         'neither obj is a str.\ntype is {}'.format(type(proto)))


def deserialize(s, obj):
    '''
    Parse bytes into a in-memory proto

    @params
    s is bytes containing serialized proto
    obj can be a class object, such as ModelProto, also can be a proto instance to be filled in

    @return
    The proto instance filled in by s
    '''
    if not isinstance(s, bytes):
        raise ValueError('Parameter s must be bytes, but got type: {}'.format(type(s)))

    if callable(obj):
        obj = obj()
    if not (hasattr(obj, 'ParseFromString') and callable(obj.ParseFromString)):
        raise ValueError('No ParseFromString method is detected. '
                         '\ntype is {}'.format(type(obj)))

    decoded = obj.ParseFromString(s)
    if decoded is not None and decoded != len(s):
        raise google.protobuf.message.DecodeError(
            "Protobuf decoding consumed too few bytes: {} out of {}".format(
                decoded, len(s)))
    return obj


def load(obj, cls=None):
    '''
    Loads a binary protobuf into memory

    @params
    obj can be a file-like object (has "read" function) or a string containing a file name
    cls is a class object, such as ModelProto, TensorProto, etc, if not specified,
    load as ModelProto

    @return
    Loaded in-memory proto object
    '''
    s = _load_string(obj)
    return load_from_string(s, cls)


def load_from_string(s, cls=None):
    '''
    Loads a binary string that contains serialized proto

    @params
    s is a string, which contains serialized proto
    cls is a class object, such as ModelProto, TensorProto, etc, if not specified,
    load as ModelProto

    @return
    Loaded in-memory proto object
    '''
    if not cls:
        cls = ModelProto
    return deserialize(s, cls)


def save(proto, f):
    '''
    Saves the proto to the specified path.

    @params
    proto can be either a in-memory proto (e.g., ModelProto) or serialized proto
    f can be a file-like object (has "write" function) or a string containing a file name
    '''
    s = serialize(proto)
    _save_string(s, f)
