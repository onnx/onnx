from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import platform

import numpy as np  # type: ignore
from collections import abc
from onnx import TensorProto, MapProto, SequenceProto, TypeProto, SequenceMapElement
from onnx import mapping, helper
from six import text_type, binary_type
from typing import Sequence, Any, Optional, Text, List, Dict

if platform.system() != 'AIX' and sys.byteorder != 'little':
    raise RuntimeError(
        'Numpy helper for tensor/ndarray is not available on big endian '
        'systems yet.')


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

    return tensor


def to_list_from_sequence(sequence):  # type: (SequenceProto) -> List
    """Converts a sequence def to a Python list.

    Inputs:
        sequence: a SequenceProto object.
    Returns:
        lst: the converted list.
    """
    lst = []
    for elem in sequence.values:
        elem_type = elem.elem_type
        if elem_type == SequenceMapElement.TENSOR or elem_type == SequenceMapElement.SPARSE_TENSOR:
            lst.append(to_array(elem))
        elif elem_type == SequenceMapElement.SEQUENCE:
            lst.append(to_list_from_sequence(elem))
        elif elem_type == SequenceMapElement.MAP:
            lst.append(to_dict_from_map(elem))
        else:
            raise TypeError("The element type in the input sequence is not defined.")
    return lst


def from_list_to_sequence(lst, name=None):  # type: (List, Optional[Text]) -> SequenceProto
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
    for elem in lst:
        # If elem is a tensor
        if isinstance(elem, np.ndarray):
            sequence.values.append(helper.make_sequence_map_element(from_array(elem)))
        elif isinstance(elem, abc.Sequence):
            sequence.values.append(from_list_to_sequence(elem))
        elif isinstance(elem, abc.Mapping):
            sequence.values.append(from_dict_to_map(elem))
        else:
            raise TypeError("The element type in the input sequence is not a list,"
                            "dictionary, or np.ndarray and is not supported.")
    return sequence


def to_dict_from_map(map):  # type: (MapProto) -> np.ndarray[Any]
    """Converts a map def to a Python dictionary.

    Inputs:
        map: a MapProto object.
    Returns:
        dict: the converted dictionary.
    """
    dict = {}
    for kv_pair in map.pairs:
        key_type = kv_pair.key_type
        value_type = kv_pair.value_type
        key_field = mapping.STORAGE_MAP_KEY_TYPE_TO_FIELD[key_type]
        key = getattr(kv_pair, key_field)
        value = kv_pair.value.value
        if value_type == TypeProto.Tensor or value_type == TypeProto.SparseTensor:
            dict[key] = to_array(value)
        elif value_type == TypeProto.Map:
            dict[key] = to_dict_from_map(value)
        elif value_type == TypeProto.Sequence:
            dict[key] = to_list_from_sequence(value)
        else:
            raise TypeError("The value type in the Map is not defined.")
    return dict


def from_dict_to_map(d, name=None):  # type: (Dict, Optional[Text]) -> MapProto
    """Converts a Python dictionary into a map def.

    Inputs:
        d: Python dictionary
        name: (optional) the name of the map.
    Returns:
        map: the converted map def.
    """
    map = MapProto()
    if name:
        map.name = name
    for key, val in d.items():
        key_type = mapping.NP_TYPE_TO_MAP_KEY_TYPE(key.dtype)
        if isinstance(val, abc.Mapping):
            val_type = TypeProto.Map  # type: ignore
        elif isinstance(val, abc.Sequence):
            val_type = TypeProto.Sequence  # type: ignore
        elif isinstance(val, np.ndarray):
            val_type = TypeProto.Tensor  # type: ignore
        kv_pair = helper.make_key_value_pair(key, key_type, val, val_type)
        map.pairs.append(kv_pair)
    return map
