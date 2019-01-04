from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
from itertools import chain

from typing import Iterable, Text

from .onnx_pb import TensorProto, ModelProto


class ExternalDataInfo(object):

    def __init__(self, tensor):  # type: (TensorProto) -> None
        self.location = ''
        self.offset = None
        self.length = None
        self.checksum = None
        self.basepath = ''

        for entry in tensor.external_data:
            setattr(self, entry.key, entry.value)

        if self.offset:
            self.offset = int(self.offset)

        if self.length:
            self.length = int(self.length)


def load_external_data(tensor):  # type: (TensorProto) -> bytes
    """
    Load data from an external file based on a tensor's `external_data` field.

    @params
    tensor: a TensorProto object.

    @return
    raw_data: bytes string containing data in raw_data format
    """
    info = ExternalDataInfo(tensor)
    file_location = _sanitize_path(info.location)
    external_data_file_path = os.path.join(info.basepath, file_location)

    with open(external_data_file_path, 'rb') as data_file:

        if info.offset:
            data_file.seek(info.offset)

        if info.length:
            raw_data = data_file.read(info.length)
        else:
            raw_data = data_file.read()

    return raw_data


def save_external_data(tensor, base_path):  # type: (TensorProto, Text) -> None
    """
    Write tensor data to an external file according to information in the `external_data` field.

    @params
    tensor: Tensor object to be serialized
    base_path: System path of a folder where tensor data is to be stored
    """
    info = ExternalDataInfo(tensor)
    external_data_file_path = os.path.join(base_path, info.location)

    # Retrieve the tensor's data from raw_data or load external file
    if tensor.HasField("raw_data"):
        raw_data = tensor.raw_data
    else:
        raw_data = load_external_data(tensor)

    # Create file if it doesn't exist
    if not os.path.isfile(external_data_file_path):
        open(external_data_file_path, 'ab').close()

    # Open file for reading and writing at random locations ('r+b')
    with open(external_data_file_path, 'r+b') as data_file:
        if info.offset is not None:

            # Pad file to required offset if needed
            data_file.seek(0, 2)
            file_size = data_file.tell()
            if info.offset > file_size:
                data_file.write(b"\0" * (info.offset - file_size))

            data_file.seek(info.offset)
        data_file.write(raw_data)


def _get_all_tensors(onnx_model_proto):  # type: (ModelProto) -> Iterable[TensorProto]
    """Scan an ONNX model for all tensors and return as an iterator."""
    return chain(_get_initializer_tensors(onnx_model_proto),
                 _get_attribute_tensors(onnx_model_proto))


def _get_initializer_tensors(onnx_model_proto):  # type: (ModelProto) -> Iterable[TensorProto]
    """Create an iterator of initializer tensors from ONNX model."""
    for initializer in onnx_model_proto.graph.initializer:
        yield initializer


def _get_attribute_tensors(onnx_model_proto):  # type: (ModelProto) -> Iterable[TensorProto]
    """Create an iterator of tensors from node attributes of an ONNX model."""
    for node in onnx_model_proto.graph.node:
        for attribute in node.attribute:
            if attribute.HasField("t"):
                yield attribute.t
            for tensor in attribute.tensors:
                yield tensor


def _sanitize_path(path):  # type: (Text) -> Text
    """Remove path components which would allow traversing up a directory tree from a base path.

    Note: This method is currently very basic and should be expanded.
    """
    return path.lstrip('/.')


def uses_external_data(tensor):  # type: (TensorProto) -> bool
    """Return true if the tensor stores data in an external location."""
    return tensor.HasField("data_location") and tensor.data_location == TensorProto.EXTERNAL


def remove_external_data_field(tensor, field_key):  # type: (TensorProto, Text) -> None
    """
    Remove a field from a Tensor's external_data key-value store.

    Modifies tensor object in place.

    @params
    tensor: Tensor object from which value will be removed
    field_key: The key of the field to be removed
    """
    for (i, field) in enumerate(tensor.external_data):
        if field.key == field_key:
            del tensor.external_data[i]


def add_basepath_to_external_data_tensors(model, filepath):  # type: (ModelProto, Text) -> ModelProto
    """
    Add basepath value to the external_data field of all tensors in model.

    Base path information is useful for finding the external data files on disk.
    Modifies model object in place.

    @params
    model: Model object to modify.
    filepath: System path to the directory which should be treated as base path for external data.

    @return
    The modified model object.
    """
    for tensor in _get_all_tensors(model):
        if len(tensor.external_data):
            tensor.external_data.add()
            tensor.external_data[-1].key = 'basepath'
            tensor.external_data[-1].value = filepath
    return model


def write_external_data_tensors(model, filepath):  # type: (ModelProto, Text) -> ModelProto
    """
    Write external data of all tensors to files on disk.

    Note: This function also strips basepath information from all tensors' external_data fields.

    @params
    model: Model object which is the source of tensors to serialize.
    filepath: System path to the directory which should be treated as base path for external data.

    @return
    The modified model object.
    """
    for tensor in _get_all_tensors(model):
        if uses_external_data(tensor):
            save_external_data(tensor, filepath)
            remove_external_data_field(tensor, 'basepath')

    return model
