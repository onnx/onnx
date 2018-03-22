from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

from onnx import numpy_helper
from onnx import external_data_helper

from .onnx_pb import *  # noqa
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
    if hasattr(obj, 'read') and callable(obj.read):
        s = obj.read()
    else:
        with open(obj, 'rb') as f:
            s = f.read()
    return load_from_string(s)


def load_from_string(s):
    '''
    Loads a binary string that stores onnx model

    @params
    Takes a string object containing protobuf
    @return ONNX ModelProto object
    '''
    model = ModelProto()
    decoded = model.ParseFromString(s)
    # in python implementation ParseFromString returns None
    if decoded is not None and decoded != len(s):
        raise google.protobuf.message.DecodeError(
            "Protobuf decoding consumed too few bytes: {} out of {}".format(
                decoded, len(s)))
    return model


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
