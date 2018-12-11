from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
from itertools import chain


class ExternalDataInfo(object):

    def __init__(self, tensor):
        self.type = 'file'
        self.location = ''
        self.basepath = ''
        self.offset = None
        self.length = None
        self.checksum = None

        for entry in tensor.external_data:
            setattr(self, entry.key, entry.value)

        if self.offset:
            self.offset = int(self.offset)

        if self.length:
            self.length = int(self.length)


def load_external_data(tensor):
    """Load data from an external file based on the `external_data` field.
    Inputs:
        tensor: a TensorProto object.
    Returns:
        raw_data: bytes string containing data in raw_data format
    """
    info = ExternalDataInfo(tensor)
    external_data_file_path = os.path.join(info.basepath, info.location)

    with open(external_data_file_path, 'rb') as data_file:

        if info.offset:
            data_file.seek(info.offset)

        if info.length:
            raw_data = data_file.read(info.length)
        else:
            raw_data = data_file.read()

    return raw_data


def save_external_data(tensor, new_basepath):
    info = ExternalDataInfo(tensor)
    external_data_file_path = os.path.join(new_basepath, info.location)
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


def _get_all_tensors(onnx_model_proto):
    """Scan an ONNX model for all tensors and return as an iterator."""
    return chain(_get_initializer_tensors(onnx_model_proto),
                 _get_attribute_tensors(onnx_model_proto))


def _get_initializer_tensors(onnx_model_proto):
    """Create an iterator of initializer tensors from ONNX model."""
    for initializer in onnx_model_proto.graph.initializer:
        yield initializer


def _get_attribute_tensors(onnx_model_proto):
    """Create an iterator of tensors from node attributes of an ONNX model."""
    for node in onnx_model_proto.graph.node:
        for attribute in node.attribute:
            if attribute.HasField("t"):
                yield attribute.t
            for tensor in attribute.tensors:
                yield tensor


def remove_external_data_field(tensor, field_key):
    for (i, field) in enumerate(tensor.external_data):
        if field.key == field_key:
            del tensor.external_data[i]


def annotate_external_data_tensors(model, filepath):
    for tensor in _get_all_tensors(model):
        tensor.external_data.add()
        tensor.external_data[-1].key = 'basepath'
        tensor.external_data[-1].value = filepath
    return model


def write_external_data_tensors(model, filepath):
    for tensor in _get_all_tensors(model):
        if tensor.HasField("data_location") and tensor.data_location == "external":
            save_external_data(tensor, filepath)
            remove_external_data_field(tensor, 'basepath')

    return model
