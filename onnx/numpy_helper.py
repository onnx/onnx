from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys

import numpy as np  # type: ignore
from onnx import TensorProto, MapProto, SequenceProto, TypeProto
from onnx import mapping, helper
from six import text_type, binary_type
from typing import Sequence, Any, Optional, Text, List, Dict


def combine_pairs_to_complex(fa):  # type: (Sequence[int]) -> Sequence[np.complex64]
    return [complex(fa[i * 2], fa[i * 2 + 1]) for i in range(len(fa) // 2)]


def to_array(tensor):  # type: (TensorProto) -> np.ndarray[Any]
    """Converts a tensor def object to a numpy array.

    Inputs:
        tensor: a TensorProto object.
    Returns:
        arr: the converted array.
    """
    if tensor.HasField("segment"):
        raise ValueError(
            "Currently not supporting loading segments.")
    if tensor.data_type == TensorProto.UNDEFINED:
        raise TypeError("The element type in the input tensor is not defined.")

    tensor_dtype = tensor.data_type
    np_dtype = mapping.TENSOR_TYPE_TO_NP_TYPE[tensor_dtype]
    storage_type = mapping.TENSOR_TYPE_TO_STORAGE_TENSOR_TYPE[tensor_dtype]
    storage_np_dtype = mapping.TENSOR_TYPE_TO_NP_TYPE[storage_type]
    storage_field = mapping.STORAGE_TENSOR_TYPE_TO_FIELD[storage_type]
    dims = tensor.dims

    if tensor.data_type == TensorProto.STRING:
        utf8_strings = getattr(tensor, storage_field)
        ss = list(s.decode('utf-8') for s in utf8_strings)
        return np.asarray(ss).astype(np_dtype).reshape(dims)

    if tensor.HasField("raw_data"):
        # Raw_bytes support: using frombuffer.
        if sys.byteorder == 'big':
            # Convert endian from little to big
            convert_endian(tensor)
        return np.frombuffer(
            tensor.raw_data,
            dtype=np_dtype).reshape(dims)
    else:
        data = getattr(tensor, storage_field),  # type: Sequence[np.complex64]
        if (tensor_dtype == TensorProto.COMPLEX64
                or tensor_dtype == TensorProto.COMPLEX128):
            data = combine_pairs_to_complex(data)
        return (
            np.asarray(
                data,
                dtype=storage_np_dtype)
            .astype(np_dtype)
            .reshape(dims)
        )


def from_array(arr, name=None):  # type: (np.ndarray[Any], Optional[Text]) -> TensorProto
    """Converts a numpy array to a tensor def.

    Inputs:
        arr: a numpy array.
        name: (optional) the name of the tensor.
    Returns:
        tensor_def: the converted tensor def.
    """
    tensor = TensorProto()
    tensor.dims.extend(arr.shape)
    if name:
        tensor.name = name

    if arr.dtype == np.object:
        # Special care for strings.
        tensor.data_type = mapping.NP_TYPE_TO_TENSOR_TYPE[arr.dtype]
        # TODO: Introduce full string support.
        # We flatten the array in case there are 2-D arrays are specified
        # We throw the error below if we have a 3-D array or some kind of other
        # object. If you want more complex shapes then follow the below instructions.
        # Unlike other types where the shape is automatically inferred from
        # nested arrays of values, the only reliable way now to feed strings
        # is to put them into a flat array then specify type astype(np.object)
        # (otherwise all strings may have different types depending on their length)
        # and then specify shape .reshape([x, y, z])
        flat_array = arr.flatten()
        for e in flat_array:
            if isinstance(e, text_type):
                tensor.string_data.append(e.encode('utf-8'))
            elif isinstance(e, np.ndarray):
                for s in e:
                    if isinstance(s, text_type):
                        tensor.string_data.append(s.encode('utf-8'))
            else:
                raise NotImplementedError(
                    "Unrecognized object in the object array, expect a string, or array of bytes: ", str(type(e)))
        return tensor

    # For numerical types, directly use numpy raw bytes.
    try:
        dtype = mapping.NP_TYPE_TO_TENSOR_TYPE[arr.dtype]
    except KeyError:
        raise RuntimeError(
            "Numpy data type not understood yet: {}".format(str(arr.dtype)))
    tensor.data_type = dtype
    tensor.raw_data = arr.tobytes()  # note: tobytes() is only after 1.9.
    if sys.byteorder == 'big':
        # Convert endian from big to little
        convert_endian(tensor)

    return tensor


def to_list(sequence):  # type: (SequenceProto) -> List[Any]
    """Converts a sequence def to a Python list.

    Inputs:
        sequence: a SequenceProto object.
    Returns:
        lst: the converted list.
    """
    lst = []
    elem_type = sequence.elem_type
    value_field = mapping.STORAGE_ELEMENT_TYPE_TO_FIELD[elem_type]
    values = getattr(sequence, value_field)
    for value in values:
        if elem_type == Sequence.TENSOR or elem_type == Sequence.SPARSE_TENSOR:
            lst.append(to_array(value))
        elif elem_type == Sequence.SEQUENCE:
            lst.append(to_list(value))
        elif elem_type == Sequence.MAP:
            lst.append(to_dict(value))
        else:
            raise TypeError("The element type in the input sequence is not supported.")
    return lst


def from_list(lst, name=None):  # type: (List[Any], Optional[Text]) -> SequenceProto
    """Converts a list into a sequence def.

    Inputs:
        lst: a Python list
        name: (optional) the name of the sequence.
    Returns:
        sequence: the converted sequence def.
    """
    sequence = SequenceProto()
    if name:
        sequence.name = name
    elem_type = type(lst[0])
    if not all(isinstance(elem, elem_type) for elem in lst):
        raise TypeError("The element type in the input list is not the same "
                        "for all elements and therefore is not supported as a sequence.")
    if isinstance(elem_type, np.ndarray):
        for tensor in lst:
            sequence.tensor_values.append(helper.make_tensor(from_array(tensor)))
    elif isinstance(elem_type, list):
        for sequence in lst:
            sequence.sequence_values.append(helper.make_sequence(from_list(elem)))
    elif isinstance(elem_type, dict):
        for map in lst:
            sequence.map_values.append(helper.make_map(from_dict(elem)))
    else:
        raise TypeError("The element type in the input list is not a list, "
                        "dictionary, or np.ndarray and is not supported.")
    return sequence


def to_dict(map):  # type: (MapProto) -> np.ndarray[Any]
    """Converts a map def to a Python dictionary.

    Inputs:
        map: a MapProto object.
    Returns:
        dict: the converted dictionary.
    """
    keys = to_list(map.keys)
    values = to_list(map.values)
    if len(keys) != len(values):
        raise IndexError("Length of keys and values for MapProto (map name: ",
                        map.name(),
                        ") are not the same.")
    dictionary = dict(zip(keys, values))
    return dictionary


def from_dict(dict, name=None):  # type: (Dict[Any, Any], Optional[Text]) -> MapProto
    """Converts a Python dictionary into a map def.

    Inputs:
        dict: Python dictionary
        name: (optional) the name of the map.
    Returns:
        map: the converted map def.
    """
    map = MapProto()
    if name:
        map.name = name
    keys = list(dict.keys())
    raw_key_type = type(keys[0])
    key_type = mapping.NP_TYPE_TO_TENSOR_TYPE[np.array(keys)[0].dtype]

    if not all(isinstance(key, raw_key_type) for key in keys):
        raise TypeError("The key type in the input dictionary is not the same "
                        "for all keys and therefore is not valid as a map.")
    key_seq = from_list(keys)

    values = list(dict.values())
    raw_value_type = type(values[0])
    if not all(isinstance(val, raw_value_type) for val in values):
        raise TypeError("The value type in the input dictionary is not the same "
                        "for all values and therefore is not valid as a map.")

    value_seq = from_list(values)
    if isinstance(values[0], np.ndarray):
        value_type = Sequence.TENSOR
    elif isinstance(values[0], list):
        value_type = Sequence.SEQUENCE
    elif isinstance(values[0], dict):
        value_type = Sequence.MAP
    else:
        raise TypeError("The value type in the input dictionary is not a list, "
                        "dictionary, or np.ndarray and is not supported.")

    map.key_type = key_type
    map.keys = key_seq
    map.value_type = value_type
    map.values = value_seq
    return map


def convert_endian(tensor):  # type: (TensorProto) -> None
    """
    call to convert endianess of raw data in tensor.
    @params
    TensorProto: TensorProto to be converted.
    """
    tensor_dtype = tensor.data_type
    np_dtype = mapping.TENSOR_TYPE_TO_NP_TYPE[tensor_dtype]
    tensor.raw_data = np.frombuffer(tensor.raw_data, dtype=np_dtype).byteswap().tobytes()
