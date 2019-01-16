from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import uuid
import os
from itertools import chain
from typing import Iterable, Text, Optional
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


def load_external_data_for_tensor(tensor, base_dir):  # type: (TensorProto, Text) -> None
    """
    Load data from an external file for tensor.

    @params
    tensor: a TensorProto object.
    base_dir: directory that contains the external data.
    """
    if tensor.HasField("raw_data"):  # already loaded
        return
    info = ExternalDataInfo(tensor)
    file_location = _sanitize_path(info.location)
    external_data_file_path = os.path.join(base_dir, file_location)

    with open(external_data_file_path, 'rb') as data_file:

        if info.offset:
            data_file.seek(info.offset)

        if info.length:
            tensor.raw_data = data_file.read(info.length)
        else:
            tensor.raw_data = data_file.read()


def load_external_data_for_model(model, base_dir):  # type: (ModelProto, Text) -> None
    """
    Loads external tensors into model

    @params
    model: ModelProto to load external data to
    base_dir: directory that contains external data
    """
    for tensor in _get_all_tensors(model):
        if uses_external_data(tensor):
            load_external_data_for_tensor(tensor, base_dir)


def set_external_data(tensor,  # type: TensorProto
                      location,  # type: Text
                      offset=None,  # type: Optional[int]
                      length=None,  # type: Optional[int]
                      checksum=None,  # type: Optional[Text]
                      basepath=None  # type: Optional[Text]
                      ):  # type: (...) -> None
    del tensor.external_data[:]
    tensor.data_location = TensorProto.EXTERNAL
    for (k, v) in {
        'location': location,
        'offset': int(offset) if offset is not None else None,
        'length': int(length) if length is not None else None,
        'checksum': checksum,
        'basepath': basepath
    }.items():
        if v is not None:
            entry = tensor.external_data.add()
            entry.key = k
            entry.value = str(v)


def convert_model_to_external_data(model, all_tensors_to_one_file=True, location=None):
    # type: (ModelProto, bool, Optional[Text]) -> None
    """
    call to set all tensors as external data. save_model saves all the tensors data as external data after calling this function.
    @params
    model: ModelProto to be converted.
    all_tensors_to_one_file: If true, save all tensors to one external file specified by location.
                             If false, save each tensor to a file named with the tensor name.
    location: specify the external file that all tensors to save to.
              If not specified, will use the model name.
    """
    if all_tensors_to_one_file:
        file_name = Text(uuid.uuid1())
        if location:
            file_name = location
        for tensor in _get_all_tensors(model):
            set_external_data(tensor, file_name)
    else:
        for tensor in _get_all_tensors(model):
            set_external_data(tensor, tensor.name)


def convert_model_from_external_data(model):  # type: (ModelProto) -> None
    """
    call to set all tensors data as embedded data. save_model saves all the tensors data as embedded data after calling this function.
    @params
    model: ModelProto to be converted.
    """
    for tensor in _get_all_tensors(model):
        if uses_external_data(tensor):
            if not tensor.HasField("raw_data"):
                raise ValueError("raw_data field doesn't exist.")
            del tensor.external_data[:]
            tensor.data_location = TensorProto.DEFAULT


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
    if not tensor.HasField("raw_data"):
        raise ValueError("raw_data field doesn't exist.")

    # Create file if it doesn't exist
    if not os.path.isfile(external_data_file_path):
        open(external_data_file_path, 'ab').close()

    # Open file for reading and writing at random locations ('r+b')
    with open(external_data_file_path, 'r+b') as data_file:
        data_file.seek(0, 2)
        if info.offset is not None:
            # Pad file to required offset if needed
            file_size = data_file.tell()
            if info.offset > file_size:
                data_file.write(b"\0" * (info.offset - file_size))

            data_file.seek(info.offset)
        offset = data_file.tell()
        data_file.write(tensor.raw_data)
        set_external_data(tensor, info.location, offset, data_file.tell() - offset)


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
            tensor.ClearField(str('raw_data'))

    return model
