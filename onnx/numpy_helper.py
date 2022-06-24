# SPDX-License-Identifier: Apache-2.0

import sys
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np  # type: ignore

from onnx import MapProto, OptionalProto, SequenceProto, TensorProto, mapping
from onnx.external_data_helper import load_external_data_for_tensor, uses_external_data


def combine_pairs_to_complex(fa: Sequence[int]) -> List[complex]:
    return [complex(fa[i * 2], fa[i * 2 + 1]) for i in range(len(fa) // 2)]


def bfloat16_to_float32(data: np.ndarray, dims: Union[int, Sequence[int]]) -> np.ndarray:
    """Converts ndarray of bf16 (as uint32) to f32 (as uint32)."""
    shift = lambda x: x << 16  # noqa: E731
    return shift(data.astype(np.int32)).reshape(dims).view(np.float32)


def to_array(tensor: TensorProto, base_dir: str = "") -> np.ndarray:
    """Converts a tensor def object to a numpy array.

    Inputs:
        tensor: a TensorProto object.
        base_dir: if external tensor exists, base_dir can help to find the path to it
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

    # Load raw data from external tensor if it exists
    if uses_external_data(tensor):
        load_external_data_for_tensor(tensor, base_dir)

    if tensor.HasField("raw_data"):
        # Raw_bytes support: using frombuffer.
        if sys.byteorder == 'big':
            # Convert endian from little to big
            convert_endian(tensor)

        # manually convert bf16 since there's no numpy support
        if tensor_dtype == TensorProto.BFLOAT16:
            data = np.frombuffer(tensor.raw_data, dtype=np.int16)
            return bfloat16_to_float32(data, dims)

        return np.frombuffer(
            tensor.raw_data,
            dtype=np_dtype).reshape(dims)
    else:
        # float16 is stored as int32 (uint16 type); Need view to get the original value
        if tensor_dtype == TensorProto.FLOAT16:
            return (
                np.asarray(
                    tensor.int32_data,
                    dtype=np.uint16)
                .reshape(dims)
                .view(np.float16))

        # bfloat16 is stored as int32 (uint16 type); no numpy support for bf16
        if tensor_dtype == TensorProto.BFLOAT16:
            data = np.asarray(tensor.int32_data, dtype=np.int32)
            return bfloat16_to_float32(data, dims)

        data = getattr(tensor, storage_field)
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


def from_array(arr: np.ndarray, name: Optional[str] = None) -> TensorProto:
    """Converts a numpy array to a tensor def.

    Inputs:
        arr: a numpy array.
        name: (optional) the name of the tensor.
    Returns:
        TensorProto: the converted tensor def.
    """
    tensor = TensorProto()
    tensor.dims.extend(arr.shape)
    if name:
        tensor.name = name

    if arr.dtype == object:
        # Special care for strings.
        tensor.data_type = mapping.NP_TYPE_TO_TENSOR_TYPE[arr.dtype]
        # TODO: Introduce full string support.
        # We flatten the array in case there are 2-D arrays are specified
        # We throw the error below if we have a 3-D array or some kind of other
        # object. If you want more complex shapes then follow the below instructions.
        # Unlike other types where the shape is automatically inferred from
        # nested arrays of values, the only reliable way now to feed strings
        # is to put them into a flat array then specify type astype(object)
        # (otherwise all strings may have different types depending on their length)
        # and then specify shape .reshape([x, y, z])
        flat_array = arr.flatten()
        for e in flat_array:
            if isinstance(e, str):
                tensor.string_data.append(e.encode('utf-8'))
            elif isinstance(e, np.ndarray):
                for s in e:
                    if isinstance(s, str):
                        tensor.string_data.append(s.encode('utf-8'))
                    elif isinstance(s, bytes):
                        tensor.string_data.append(s)
            elif isinstance(e, bytes):
                tensor.string_data.append(e)
            else:
                raise NotImplementedError(
                    "Unrecognized object in the object array, expect a string, or array of bytes: ", str(type(e)))
        return tensor

    # For numerical types, directly use numpy raw bytes.
    try:
        dtype = mapping.NP_TYPE_TO_TENSOR_TYPE[arr.dtype]
    except KeyError:
        raise RuntimeError(
            f"Numpy data type not understood yet: {str(arr.dtype)}")
    tensor.data_type = dtype
    tensor.raw_data = arr.tobytes()  # note: tobytes() is only after 1.9.
    if sys.byteorder == 'big':
        # Convert endian from big to little
        convert_endian(tensor)

    return tensor


def to_list(sequence: SequenceProto) -> List[Any]:
    """Converts a sequence def to a Python list.

    Inputs:
        sequence: a SequenceProto object.
    Returns:
        list: the converted list.
    """
    lst: List[Any] = []
    elem_type = sequence.elem_type
    value_field = mapping.STORAGE_ELEMENT_TYPE_TO_FIELD[elem_type]
    values = getattr(sequence, value_field)
    for value in values:
        if elem_type == SequenceProto.TENSOR or elem_type == SequenceProto.SPARSE_TENSOR:
            lst.append(to_array(value))
        elif elem_type == SequenceProto.SEQUENCE:
            lst.append(to_list(value))
        elif elem_type == SequenceProto.MAP:
            lst.append(to_dict(value))
        else:
            raise TypeError("The element type in the input sequence is not supported.")
    return lst


def from_list(lst: List[Any], name: Optional[str] = None, dtype: Optional[int] = None) -> SequenceProto:
    """Converts a list into a sequence def.

    Inputs:
        lst: a Python list
        name: (optional) the name of the sequence.
        dtype: (optional) type of element in the input list, used for specifying
                          sequence values when converting an empty list.
    Returns:
        SequenceProto: the converted sequence def.
    """
    sequence = SequenceProto()
    if name:
        sequence.name = name

    if dtype:
        elem_type = dtype
    elif len(lst) > 0:
        first_elem = lst[0]
        if isinstance(first_elem, dict):
            elem_type = SequenceProto.MAP
        elif isinstance(first_elem, list):
            elem_type = SequenceProto.SEQUENCE
        else:
            elem_type = SequenceProto.TENSOR
    else:
        # if empty input list and no dtype specified
        # choose sequence of tensors on default
        elem_type = SequenceProto.TENSOR
    sequence.elem_type = elem_type

    if (len(lst) > 0) and not all(isinstance(elem, type(lst[0])) for elem in lst):
        raise TypeError("The element type in the input list is not the same "
                        "for all elements and therefore is not supported as a sequence.")

    if elem_type == SequenceProto.TENSOR:
        for tensor in lst:
            sequence.tensor_values.extend([from_array(tensor)])
    elif elem_type == SequenceProto.SEQUENCE:
        for seq in lst:
            sequence.sequence_values.extend([from_list(seq)])
    elif elem_type == SequenceProto.MAP:
        for map in lst:
            sequence.map_values.extend([from_dict(map)])
    else:
        raise TypeError("The element type in the input list is not a tensor, "
                        "sequence, or map and is not supported.")
    return sequence


def to_dict(map: MapProto) -> Dict[Any, Any]:
    """Converts a map def to a Python dictionary.

    Inputs:
        map: a MapProto object.
    Returns:
        dict: the converted dictionary.
    """
    key_list: List[Any] = []
    if map.key_type == TensorProto.STRING:
        key_list = list(map.string_keys)
    else:
        key_list = list(map.keys)

    value_list = to_list(map.values)
    if len(key_list) != len(value_list):
        raise IndexError("Length of keys and values for MapProto (map name: ",
                        map.name,
                        ") are not the same.")
    dictionary = dict(zip(key_list, value_list))
    return dictionary


def from_dict(dict: Dict[Any, Any], name: Optional[str] = None) -> MapProto:
    """Converts a Python dictionary into a map def.

    Inputs:
        dict: Python dictionary
        name: (optional) the name of the map.
    Returns:
        MapProto: the converted map def.
    """
    map = MapProto()
    if name:
        map.name = name
    keys = list(dict.keys())
    raw_key_type = np.array(keys[0]).dtype
    key_type = mapping.NP_TYPE_TO_TENSOR_TYPE[raw_key_type]

    valid_key_int_types = [TensorProto.INT8, TensorProto.INT16, TensorProto.INT32,
                           TensorProto.INT64, TensorProto.UINT8, TensorProto.UINT16,
                           TensorProto.UINT32, TensorProto.UINT64]

    if not all(isinstance(key, raw_key_type) for key in keys):
        raise TypeError("The key type in the input dictionary is not the same "
                        "for all keys and therefore is not valid as a map.")

    values = list(dict.values())
    raw_value_type = type(values[0])
    if not all(isinstance(val, raw_value_type) for val in values):
        raise TypeError("The value type in the input dictionary is not the same "
                        "for all values and therefore is not valid as a map.")

    value_seq = from_list(values)

    map.key_type = key_type
    if key_type == TensorProto.STRING:
        map.string_keys.extend(keys)
    elif key_type in valid_key_int_types:
        map.keys.extend(keys)
    map.values.CopyFrom(value_seq)
    return map


def to_optional(optional: OptionalProto) -> Optional[Any]:
    """Converts an optional def to a Python optional.

    Inputs:
        optional: an OptionalProto object.
    Returns:
        opt: the converted optional.
    """
    opt: Optional[Any] = None
    elem_type = optional.elem_type
    if elem_type == OptionalProto.UNDEFINED:
        return opt
    value_field = mapping.OPTIONAL_ELEMENT_TYPE_TO_FIELD[elem_type]
    value = getattr(optional, value_field)
    # TODO: create a map and replace conditional branches
    if elem_type == OptionalProto.TENSOR or elem_type == OptionalProto.SPARSE_TENSOR:
        opt = to_array(value)
    elif elem_type == OptionalProto.SEQUENCE:
        opt = to_list(value)
    elif elem_type == OptionalProto.MAP:
        opt = to_dict(value)
    elif elem_type == OptionalProto.OPTIONAL:
        return to_optional(value)
    else:
        raise TypeError("The element type in the input optional is not supported.")
    return opt


def from_optional(
        opt: Optional[Any],
        name: Optional[str] = None,
        dtype: Optional[int] = None
) -> OptionalProto:
    """Converts an optional value into a Optional def.

    Inputs:
        opt: a Python optional
        name: (optional) the name of the optional.
        dtype: (optional) type of element in the input, used for specifying
                          optional values when converting empty none. dtype must
                          be a valid OptionalProto.DataType value
    Returns:
        optional: the converted optional def.
    """
    # TODO: create a map and replace conditional branches
    optional = OptionalProto()
    if name:
        optional.name = name

    if dtype:
        # dtype must be a valid OptionalProto.DataType
        valid_dtypes = [v for v in OptionalProto.DataType.values()]
        assert dtype in valid_dtypes
        elem_type = dtype
    elif isinstance(opt, dict):
        elem_type = OptionalProto.MAP
    elif isinstance(opt, list):
        elem_type = OptionalProto.SEQUENCE
    elif opt is None:
        elem_type = OptionalProto.UNDEFINED
    else:
        elem_type = OptionalProto.TENSOR

    optional.elem_type = elem_type

    if opt is not None:
        if elem_type == OptionalProto.TENSOR:
            optional.tensor_value.CopyFrom(from_array(opt))
        elif elem_type == OptionalProto.SEQUENCE:
            optional.sequence_value.CopyFrom(from_list(opt))
        elif elem_type == OptionalProto.MAP:
            optional.map_value.CopyFrom(from_dict(opt))
        else:
            raise TypeError("The element type in the input is not a tensor, "
                            "sequence, or map and is not supported.")
    return optional


def convert_endian(tensor: TensorProto) -> None:
    """
    Call to convert endianess of raw data in tensor.

    Arguments:
        tensor (TensorProto): TensorProto to be converted.
    """
    tensor_dtype = tensor.data_type
    np_dtype = mapping.TENSOR_TYPE_TO_NP_TYPE[tensor_dtype]
    tensor.raw_data = np.frombuffer(tensor.raw_data, dtype=np_dtype).byteswap().tobytes()
