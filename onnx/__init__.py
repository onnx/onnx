from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

from onnx import numpy_helper
from onnx import external_data_helper

from .onnx_pb import *  # noqa
from .onnx_operators_pb import * # noqa
from .version import version as __version__  # noqa

# Import common subpackages so they're available when you 'import onnx'
import onnx.helper  # noqa
import onnx.checker  # noqa
import onnx.defs  # noqa

import google.protobuf.message

from typing import Union, Text, IO, Optional, cast, TypeVar, Any


# f should be either readable or a file path
def _load_bytes(f):  # type: (Union[IO[bytes], Text]) -> bytes
    if hasattr(f, 'read') and callable(cast(IO[bytes], f).read):
        s = cast(IO[bytes], f).read()
    else:
        with open(cast(Text, f), 'rb') as readable:
            s = readable.read()
    return s


# str should be bytes,
# f should be either writable or a file path
def _save_bytes(str, f):  # type: (bytes, Union[IO[bytes], Text]) -> None
    if hasattr(f, 'write') and callable(cast(IO[bytes], f).write):
        cast(IO[bytes], f).write(str)
    else:
        with open(cast(Text, f), 'wb') as writable:
            writable.write(str)


def _serialize(proto):  # type: (Union[bytes, google.protobuf.message.Message]) -> bytes
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
        result = proto.SerializeToString()
        return result
    else:
        raise ValueError('No SerializeToString method is detected. '
                         'neither proto is a str.\ntype is {}'.format(type(proto)))


_Proto = TypeVar('_Proto', bound=google.protobuf.message.Message)


def _deserialize(s, proto):  # type: (bytes, _Proto) -> _Proto
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

    decoded = cast(Optional[int], proto.ParseFromString(s))
    if decoded is not None and decoded != len(s):
        raise google.protobuf.message.DecodeError(
            "Protobuf decoding consumed too few bytes: {} out of {}".format(
                decoded, len(s)))
    return proto


def load_model(f, format=None):  # type: (Union[IO[bytes], Text], Optional[Any]) -> ModelProto
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


def load_tensor(f, format=None):  # type: (Union[IO[bytes], Text], Optional[Any]) -> TensorProto
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


def load_model_from_string(s, format=None):  # type: (bytes, Optional[Any]) -> ModelProto
    '''
    Loads a binary string (bytes) that contains serialized ModelProto

    @params
    s is a string, which contains serialized ModelProto
    format is for future use

    @return
    Loaded in-memory ModelProto
    '''
    return _deserialize(s, ModelProto())


def load_tensor_from_string(s, format=None):  # type: (bytes, Optional[Any]) -> TensorProto
    '''
    Loads a binary string (bytes) that contains serialized TensorProto

    @params
    s is a string, which contains serialized TensorProto
    format is for future use

    @return
    Loaded in-memory TensorProto
    '''
    return _deserialize(s, TensorProto())


def save_model(proto, f, format=None):  # type: (Union[ModelProto, bytes], Union[IO[bytes], Text], Optional[Any]) -> None
    '''
    Saves the ModelProto to the specified path.

    @params
    proto should be a in-memory ModelProto
    f can be a file-like object (has "write" function) or a string containing a file name
    format is for future use
    '''
    s = _serialize(proto)
    _save_bytes(s, f)


def save_tensor(proto, f):  # type: (TensorProto, Union[IO[bytes], Text]) -> None
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


def load_from_disk(onnx_filename, lazy_loading=True):
    """Load binary protobuf file with an ONNX model.

    :param onnx_filename: Path to file containing an ONNX model.
    :param lazy_loading: By default tensor values are loaded from external data
            files only when accessed using `numpy_helper.to_array`.
            Set this to False to load all external data values into memory.
    :return: loaded ONNX model
    """
    with open(onnx_filename, 'rb') as f:
        onnx_string = f.read()
    onnx_model_proto = load_from_string(onnx_string)

    external_data_helper.set_external_data_runtime_values(
        onnx_model_proto, onnx_filename)

    if not lazy_loading:
        for tensor in external_data_helper.get_all_tensors(onnx_model_proto):
            numpy_helper.to_array(tensor)

    return onnx_model_proto


def save_to_disk(onnx_model_proto, filename):
    """Save ONNX model to files on disk.

    External data is written to additional files relative to the directory
    in which the ONNX file is written.

    :param onnx_model_proto: ONNX Protocol Buffers model
    :param filename: path to the output file
    """
    dirname = os.path.dirname(filename)

    for tensor in external_data_helper.get_all_tensors(onnx_model_proto):
        if tensor.HasField("external_data"):
            if tensor.external_data.startswith('runtime://'):
                persistence_val = external_data_helper.runtime_to_persistence(
                    tensor.external_data)
                tensor.external_data = persistence_val

            data_filename = external_data_helper.persistence_to_filename(
                tensor.external_data)
            external_data_filepath = os.path.join(dirname, data_filename)

            tensor_value = numpy_helper.to_array(tensor)

            # Write external data file
            with open(external_data_filepath, 'wb') as data_file:
                data_file.write(tensor_value.tobytes())

            # Clear tensor data fields
            for data_field in ['double_data', 'float_data', 'int32_data',
                               'int64_data', 'raw_data',
                               'string_data', 'uint64_data']:
                tensor.ClearField(data_field)

    with open(filename, 'wb') as f:
        f.write(onnx_model_proto.SerializeToString())
