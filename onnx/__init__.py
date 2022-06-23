# SPDX-License-Identifier: Apache-2.0

import os

from .onnx_cpp2py_export import ONNX_ML
from onnx.external_data_helper import load_external_data_for_model, write_external_data_tensors, convert_model_to_external_data
from .onnx_pb import *  # noqa
from .onnx_operators_pb import * # noqa
from .onnx_data_pb import * # noqa
from .version import version as __version__  # noqa

# Import common subpackages so they're available when you 'import onnx'
import onnx.checker  # noqa
import onnx.defs  # noqa
import onnx.helper  # noqa
import onnx.utils  # noqa
import onnx.compose  # noqa

import google.protobuf.message

from typing import Union, IO, Optional, cast, TypeVar, Any


# f should be either readable or a file path
def _load_bytes(f: Union[IO[bytes], str]) -> bytes:
    if hasattr(f, 'read') and callable(cast(IO[bytes], f).read):
        s = cast(IO[bytes], f).read()
    else:
        with open(cast(str, f), 'rb') as readable:
            s = readable.read()
    return s


# content should be bytes,
# f should be either writable or a file path
def _save_bytes(content: bytes, f: Union[IO[bytes], str]) -> None:
    if hasattr(f, 'write') and callable(cast(IO[bytes], f).write):
        cast(IO[bytes], f).write(content)
    else:
        with open(cast(str, f), 'wb') as writable:
            writable.write(content)


# f should be either a readable file or a file path
def _get_file_path(f: Union[IO[bytes], str]) -> Optional[str]:
    if isinstance(f, str):
        return os.path.abspath(f)
    if hasattr(f, 'name'):
        return os.path.abspath(f.name)
    return None


def _serialize(proto: Union[bytes, google.protobuf.message.Message]) -> bytes:
    '''
    Serialize a in-memory proto to bytes

    Arguments:
        proto: a in-memory proto, such as a ModelProto, TensorProto, etc

    Returns:
        Serialized proto in bytes
    '''
    if isinstance(proto, bytes):
        return proto
    elif hasattr(proto, 'SerializeToString') and callable(proto.SerializeToString):
        result = proto.SerializeToString()
        return result
    else:
        raise TypeError('No SerializeToString method is detected. '
                         'neither proto is a str.\ntype is {}'.format(type(proto)))


_Proto = TypeVar('_Proto', bound=google.protobuf.message.Message)


def _deserialize(s: bytes, proto: _Proto) -> _Proto:
    '''
    Parse bytes into a in-memory proto

    Arguments:
        s: bytes containing serialized proto
        proto: a in-memory proto object

    Returns:
        The proto instance filled in by s
    '''
    if not isinstance(s, bytes):
        raise ValueError(f'Parameter s must be bytes, but got type: {type(s)}')

    if not (hasattr(proto, 'ParseFromString') and callable(proto.ParseFromString)):
        raise ValueError('No ParseFromString method is detected. '
                         '\ntype is {}'.format(type(proto)))

    decoded = cast(Optional[int], proto.ParseFromString(s))
    if decoded is not None and decoded != len(s):
        raise google.protobuf.message.DecodeError(
            "Protobuf decoding consumed too few bytes: {} out of {}".format(
                decoded, len(s)))
    return proto


def load_model(f: Union[IO[bytes], str], format: Optional[Any] = None, load_external_data: bool = True) -> ModelProto:
    '''
    Loads a serialized ModelProto into memory
    load_external_data is true if the external data under the same directory of the model and load the external data
    If not, users need to call load_external_data_for_model with directory to load

    Arguments:
        f: can be a file-like object (has "read" function) or a string containing a file name
        format: for future use

    Returns:
        Loaded in-memory ModelProto
    '''
    s = _load_bytes(f)
    model = load_model_from_string(s, format=format)

    if load_external_data:
        model_filepath = _get_file_path(f)
        if model_filepath:
            base_dir = os.path.dirname(model_filepath)
            load_external_data_for_model(model, base_dir)

    return model


def load_tensor(f: Union[IO[bytes], str], format: Optional[Any] = None) -> TensorProto:
    '''
    Loads a serialized TensorProto into memory

    Arguments:
        f: can be a file-like object (has "read" function) or a string containing a file name
        format: for future use

    Returns:
        Loaded in-memory TensorProto
    '''
    s = _load_bytes(f)
    return load_tensor_from_string(s, format=format)


def load_model_from_string(s: bytes, format: Optional[Any] = None) -> ModelProto:
    '''
    Loads a binary string (bytes) that contains serialized ModelProto

    Arguments:
        s: a string, which contains serialized ModelProto
        format: for future use

    Returns:
        Loaded in-memory ModelProto
    '''
    return _deserialize(s, ModelProto())


def load_tensor_from_string(s: bytes, format: Optional[Any] = None) -> TensorProto:
    '''
    Loads a binary string (bytes) that contains serialized TensorProto

    Arguments:
        s: a string, which contains serialized TensorProto
        format: for future use

    Returns:
        Loaded in-memory TensorProto
    '''
    return _deserialize(s, TensorProto())


def save_model(proto: Union[ModelProto, bytes], f: Union[IO[bytes], str], format: Optional[Any] = None, save_as_external_data: bool = False, all_tensors_to_one_file: bool = True, location: Optional[str] = None, size_threshold: int = 1024, convert_attribute: bool = False) -> None:
    '''
    Saves the ModelProto to the specified path and optionally, serialize tensors with raw data as external data before saving.

    Arguments:
        proto: should be a in-memory ModelProto
        f: can be a file-like object (has "write" function) or a string containing a file name format for future use
        all_tensors_to_one_file: If true, save all tensors to one external file specified by location.
            If false, save each tensor to a file named with the tensor name.
        location: specify the external file that all tensors to save to.
            If not specified, will use the model name.
        size_threshold: Threshold for size of data. Only when tensor's data is >= the size_threshold it will be converted
            to external data. To convert every tensor with raw data to external data set size_threshold=0.
        convert_attribute: If true, convert all tensors to external data
            If false, convert only non-attribute tensors to external data
    '''
    if isinstance(proto, bytes):
        proto = _deserialize(proto, ModelProto())

    if save_as_external_data:
        convert_model_to_external_data(proto, all_tensors_to_one_file, location, size_threshold, convert_attribute)

    model_filepath = _get_file_path(f)
    if model_filepath:
        basepath = os.path.dirname(model_filepath)
        proto = write_external_data_tensors(proto, basepath)

    s = _serialize(proto)
    _save_bytes(s, f)


def save_tensor(proto: TensorProto, f: Union[IO[bytes], str]) -> None:
    '''
    Saves the TensorProto to the specified path.

    Arguments:
        proto: should be a in-memory TensorProto
        f: can be a file-like object (has "write" function) or a string containing a file name
        format: for future use
    '''
    s = _serialize(proto)
    _save_bytes(s, f)


# For backward compatibility
load = load_model
load_from_string = load_model_from_string
save = save_model
