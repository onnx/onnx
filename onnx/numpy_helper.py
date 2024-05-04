# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys
import typing
from typing import Any, Sequence

import numpy as np

import onnx
import onnx.external_data_helper
from onnx import helper, subbyte

if typing.TYPE_CHECKING:
    import numpy.typing as npt


def combine_pairs_to_complex(fa: Sequence[int]) -> list[complex]:
    """Converts alternating [real, imaginary, ...] numbers to complex numbers."""
    return [complex(fa[i * 2], fa[i * 2 + 1]) for i in range(len(fa) // 2)]


def _left_shift_16_bits(
    data: npt.NDArray[np.uint16 | np.int32],
) -> npt.NDArray[np.uint32]:
    # The left shifted result is always int64, so we need to convert it back to uint32
    return (data << 16).astype(np.uint32)


def bfloat16_to_float32(
    data: npt.NDArray[np.uint16 | np.int32],
) -> npt.NDArray[np.float32]:
    """Converts ndarray of bf16 (as uint16 / uint32) to f32.

    Args:
        data: A numpy array, empty dimensions are allowed if dims is
            None.

    Returns:
        A numpy array of float32 with the same dimension.
    """
    return _left_shift_16_bits(data).view(np.float32)


def float8e4m3_to_float32(
    data: npt.NDArray[np.integer] | int, fn: bool = True, uz: bool = False
) -> npt.NDArray[np.float32]:
    """Converts ndarray of float8e4m3 (as uint) to float32.

    See :ref:`onnx-detail-float8` for technical details.

    Args:
        data: A numpy array, empty dimensions are allowed if dims is None.
        fn: Finite. No infinite values.
        uz: Unique zero. No negative zero or negative inf.

    Returns:
        A numpy array of converted float32.
    """
    if not fn:
        raise NotImplementedError(
            "float8e4m3_to_float32 not implemented with fn=False."
        )
    if not isinstance(data, np.ndarray):
        array = np.array(data, dtype=np.uint32)
    else:
        array = data.astype(np.uint32)
    is_scalar = array.ndim == 0
    if is_scalar:
        array = np.reshape(array, (1,))

    range_min = 0b0_0000_000  # 0x00, 0
    range_max = 0b1_1111_111  # 0xFF, 255
    fn_uz_nan = 0b1_0000_000  # 0x80, 128
    fn_nan = 0b0_1111_111  # 0x7F, 127
    if np.any(array < range_min) or np.any(array > range_max):
        raise ValueError(
            f"{array} is not a float8 value because its binary representation is out of range [0, 255]."
        )
    result = np.zeros_like(array, dtype=np.uint32)
    if uz:
        exponent_bias = 8
        # Only positive NaN is defined
        result[array == fn_uz_nan] = np.float32(np.nan).view(np.uint32)
        # Locations of the finite values
        finite_mask = array != fn_uz_nan
    else:
        exponent_bias = 7
        # Both positive and negative NaN are defined
        result[array == fn_nan] = np.float32(np.nan).view(np.uint32)
        result[array == range_max] = np.float32(-np.nan).view(np.uint32)
        # Locations of the finite values
        finite_mask = (array != fn_nan) & (array != range_max)

    # Mask out the sign, exponent and mantissa parts
    sign_mask = 0b1_0000_000  # First bit is the sign bit
    signs = array & sign_mask
    exponent_mask = 0b0_1111_000  # The next 4 bits are the exponent

    exponents = (array & exponent_mask) >> 3
    mantissa_mask = 0b0_0000_111  # The last 3 bits are the mantissa
    mantissas = array & mantissa_mask

    # Construct the float32 value
    # First move the sign bit to the correct position
    result[finite_mask] = signs << 24
    # Subnormal number
    # if mantissa > 0:
    #     exponent = 127 - exponent_bias
    #     if mantissa & 0b100 == 0:
    #         mantissa &= 0b011
    #         mantissa <<= 1
    #         exponent -= 1
    #     if mantissa & 0b100 == 0:
    #         mantissa &= 0b011
    #         mantissa <<= 1
    #         exponent -= 1
    #     result |= (mantissa & 0b011) << 21
    #     result |= exponent << 23
    subnormal_mask = finite_mask & (exponents == 0) & (mantissas > 0)
    subnormal_exponents = np.full_like(result, 127 - exponent_bias, dtype=np.uint32)
    subnormal_mantissas = mantissas * subnormal_mask

    subnormal_mantissa_selector = (subnormal_mantissas & 0b100) == 0
    subnormal_mantissas[subnormal_mantissa_selector] = (
        subnormal_mantissas[subnormal_mantissa_selector] & 0b011
    ) << 1
    subnormal_exponents[subnormal_mantissa_selector] -= 1

    subnormal_mantissa_selector = (subnormal_mantissas & 0b100) == 0
    subnormal_mantissas[subnormal_mantissa_selector] = (
        subnormal_mantissas & 0b011
    ) << 1
    subnormal_exponents[subnormal_mantissa_selector] -= 1

    result[subnormal_mask] |= (subnormal_mantissas & 0b011)[subnormal_mask] << 21
    result[subnormal_mask] |= subnormal_exponents[subnormal_mask] << 23

    # Normal number
    # float32: e8m23
    # []_[][][][][][][][]_[][][][][][][][][][][][][][][][][][][][][][][]
    # 31   29  27  25  23 22  20  18  16  14  12  10 9 8 7 6 5 4 3 2 1 0
    # S   0 0 0 0 E E E E  M M M 0 ....................................0
    #
    # result |= mantissa << 20
    # exponent += 127 - exponent_bias
    # result |= exponent << 23
    normal_mask = finite_mask & (exponents > 0)
    result[normal_mask] |= mantissas[normal_mask] << 20
    exponents[normal_mask] += 127 - exponent_bias
    result[normal_mask] |= exponents[normal_mask] << 23
    result = result.view(np.float32)
    if is_scalar:
        return result[0]
    return result


def float8e5m2_to_float32(
    data: npt.NDArray[np.integer] | int, fn: bool = False, uz: bool = False
) -> npt.NDArray[np.float32]:
    """Converts ndarray of float8, e5m2 (as uint32) to f32 (as uint32).

    See :ref:`onnx-detail-float8` for technical details.

    Args:
        data: A numpy array, empty dimensions are allowed if dims is None.
        fn: Finite. No infinite values.
        uz: Unique zero. No negative zero or negative inf.

    Returns:
        A numpy array of converted float32.
    """
    if not isinstance(data, np.ndarray):
        array = np.array(data, dtype=np.uint32)
    else:
        array = data.astype(np.uint32)
    is_scalar = array.ndim == 0
    if is_scalar:
        array = np.reshape(array, (1,))

    range_min = 0b0_00000_00  # 0x00, 0
    range_max = 0b1_11111_11  # 0xFF, 255
    if np.any(array < range_min) or np.any(array > range_max):
        raise ValueError(
            f"{array} is not a float8 value because its binary representation is out of range [0, 255]."
        )
    result = np.zeros_like(array, dtype=np.uint32)
    if fn and uz:
        exponent_bias = 16
        fn_uz_nan = 0b1_00000_00  # 0x80, 128
        result[array == fn_uz_nan] = np.float32(np.nan).view(np.uint32)
        finite_mask = array != fn_uz_nan
    elif not fn and not uz:
        exponent_bias = 15
        negative_nan = 0b1_11111_01
        result[array >= negative_nan] = np.float32(-np.nan).view(np.uint32)
        positive_nan_min = 0b0_11111_01
        positive_nan_max = 0b0_11111_11
        result[(array >= positive_nan_min) & (array <= positive_nan_max)] = np.float32(
            np.nan
        ).view(np.uint32)
        negative_inf = 0b1_11111_00
        result[array == negative_inf] = np.float32(-np.inf).view(np.uint32)
        positive_inf = 0b0_11111_00
        result[array == positive_inf] = np.float32(np.inf).view(np.uint32)
        finite_mask = ~(
            (array >= negative_nan)
            | ((array >= positive_nan_min) & (array <= positive_nan_max))
            | (array == negative_inf)
            | (array == positive_inf)
        )
    else:
        raise NotImplementedError("fn and uz must be both False or True.")

    # Mask out the sign, exponent and mantissa parts
    sign_mask = 0b1_0000_000  # First bit is the sign bit
    signs = array & sign_mask
    exponent_mask = 0b0_1111_000  # The next 4 bits are the exponent

    exponents = (array & exponent_mask) >> 3
    mantissa_mask = 0b0_0000_111  # The last 3 bits are the mantissa
    mantissas = array & mantissa_mask

    # Construct the float32 value
    # First move the sign bit to the correct position
    result[finite_mask] = signs << 24
    # if exponent == 0:
    #     # Subnormal number
    #     if mantissa > 0:
    #         exponent = 127 - exponent_bias
    #         if mantissa & 0b10 == 0:
    #             mantissa &= 0b01
    #             mantissa <<= 1
    #             exponent -= 1
    #         result |= (mantissa & 0b01) << 22
    #         result |= exponent << 23
    subnormal_mask = finite_mask & (exponents == 0) & (mantissas > 0)
    subnormal_exponents = np.full_like(result, 127 - exponent_bias, dtype=np.uint32)
    subnormal_mantissas = mantissas * subnormal_mask

    subnormal_mantissa_selector = (subnormal_mantissas & 0b10) == 0
    subnormal_mantissas[subnormal_mantissa_selector] = (
        subnormal_mantissas[subnormal_mantissa_selector] & 0b01
    ) << 1
    subnormal_exponents[subnormal_mantissa_selector] -= 1

    result[subnormal_mask] |= (subnormal_mantissas & 0b011)[subnormal_mask] << 22
    result[subnormal_mask] |= subnormal_exponents[subnormal_mask] << 23

    # Normal number
    # float32: e8m23
    # []_[][][][][][][][]_[][][][][][][][][][][][][][][][][][][][][][][]
    # 31   29  27  25  23 22  20  18  16  14  12  10 9 8 7 6 5 4 3 2 1 0
    # S   0 0 0 E E E E E  M M 0 ......................................0
    #
    # result |= mantissa << 21
    # exponent += 127 - exponent_bias
    # result |= exponent << 23
    normal_mask = finite_mask & (exponents > 0)
    result[normal_mask] |= mantissas[normal_mask] << 21
    exponents[normal_mask] += 127 - exponent_bias
    result[normal_mask] |= exponents[normal_mask] << 23
    result = result.view(np.float32)
    if is_scalar:
        return result[0]
    return result


def _small_endian_dtype(dtype) -> np.dtype:
    """Create a small endian dtype on all platforms.

    This is useful because ONNX always stores raw_data in small endian. On big
    endian platforms, we still need to interpret the raw_data in small endian.
    """
    return np.dtype(dtype).newbyteorder("<")


def to_array(  # noqa: PLR0911
    tensor: onnx.TensorProto, base_dir: str = ""
) -> np.ndarray:
    """Converts a tensor def object to a numpy array.

    Args:
        tensor: a TensorProto object.
        base_dir: if external tensor exists, base_dir can help to find the path to it

    Returns:
        arr: the converted array.
    """
    if tensor.HasField("segment"):
        raise ValueError("Currently not supporting loading segments.")
    if tensor.data_type == onnx.TensorProto.UNDEFINED:
        raise TypeError("The element type in the input tensor is not defined.")

    tensor_dtype = tensor.data_type
    np_dtype = helper.tensor_dtype_to_np_dtype(tensor_dtype)
    storage_np_dtype = helper.tensor_dtype_to_np_dtype(
        helper.tensor_dtype_to_storage_tensor_dtype(tensor_dtype)
    )
    storage_field = helper.tensor_dtype_to_field(tensor_dtype)
    dims = tensor.dims

    if tensor.data_type == onnx.TensorProto.STRING:
        utf8_strings = getattr(tensor, storage_field)
        return (
            np.asarray([s.decode("utf-8") for s in utf8_strings])
            .astype(np_dtype)
            .reshape(dims)
        )

    # Load raw data from external tensor if it exists
    if onnx.external_data_helper.uses_external_data(tensor):
        onnx.external_data_helper.load_external_data_for_tensor(tensor, base_dir)

    if tensor.HasField("raw_data"):
        # Raw_bytes support: using frombuffer.
        raw_data = tensor.raw_data

        # manually convert bf16 since there's no numpy support
        if tensor_dtype == onnx.TensorProto.BFLOAT16:
            data = np.frombuffer(raw_data, dtype=_small_endian_dtype(np.uint16))
            return bfloat16_to_float32(data).reshape(dims)

        if tensor_dtype == onnx.TensorProto.FLOAT8E4M3FN:
            data = np.frombuffer(raw_data, dtype=_small_endian_dtype(np.uint8))
            return float8e4m3_to_float32(data).reshape(dims)

        if tensor_dtype == onnx.TensorProto.FLOAT8E4M3FNUZ:
            data = np.frombuffer(raw_data, dtype=_small_endian_dtype(np.uint8))
            return float8e4m3_to_float32(data, uz=True).reshape(dims)

        if tensor_dtype == onnx.TensorProto.FLOAT8E5M2:
            data = np.frombuffer(raw_data, dtype=_small_endian_dtype(np.uint8))
            return float8e5m2_to_float32(data).reshape(dims)

        if tensor_dtype == onnx.TensorProto.FLOAT8E5M2FNUZ:
            data = np.frombuffer(raw_data, dtype=_small_endian_dtype(np.uint8))
            return float8e5m2_to_float32(data, fn=True, uz=True).reshape(dims)

        if tensor_dtype == onnx.TensorProto.UINT4:
            data = np.frombuffer(raw_data, dtype=_small_endian_dtype(np.uint8))
            return subbyte.unpack_int4(data, dims, signed=False)

        if tensor_dtype == onnx.TensorProto.INT4:
            data = np.frombuffer(raw_data, dtype=_small_endian_dtype(np.uint8))
            return subbyte.unpack_int4(data, dims, signed=True)

        return np.frombuffer(raw_data, dtype=_small_endian_dtype(np_dtype)).reshape(dims)  # type: ignore[no-any-return]

    # Load data from the sequential data fields
    # float16 is stored as int32 (uint16 type); Need view to get the original value
    if tensor_dtype == onnx.TensorProto.FLOAT16:
        return (
            np.asarray(tensor.int32_data, dtype=np.uint16)
            .reshape(dims)
            .view(np.float16)
        )

    # bfloat16 is stored as int32 (uint16 type); no numpy support for bf16
    if tensor_dtype == onnx.TensorProto.BFLOAT16:
        data = np.asarray(tensor.int32_data, dtype=np.int32)
        return bfloat16_to_float32(data).reshape(dims)

    if tensor_dtype == onnx.TensorProto.FLOAT8E4M3FN:
        data = np.asarray(tensor.int32_data, dtype=np.int32)
        return float8e4m3_to_float32(data).reshape(dims)

    if tensor_dtype == onnx.TensorProto.FLOAT8E4M3FNUZ:
        data = np.asarray(tensor.int32_data, dtype=np.int32)
        return float8e4m3_to_float32(data, uz=True).reshape(dims)

    if tensor_dtype == onnx.TensorProto.FLOAT8E5M2:
        data = np.asarray(tensor.int32_data, dtype=np.int32)
        return float8e5m2_to_float32(data).reshape(dims)

    if tensor_dtype == onnx.TensorProto.FLOAT8E5M2FNUZ:
        data = np.asarray(tensor.int32_data, dtype=np.int32)
        return float8e5m2_to_float32(data, fn=True, uz=True).reshape(dims)

    if tensor_dtype == onnx.TensorProto.UINT4:
        data = np.asarray(tensor.int32_data, dtype=np.int32).astype(np.uint8)
        return subbyte.unpack_int4(data, dims, signed=False)

    if tensor_dtype == onnx.TensorProto.INT4:
        data = np.asarray(tensor.int32_data, dtype=np.int32).astype(np.uint8)
        return subbyte.unpack_int4(data, dims, signed=True)

    data = getattr(tensor, storage_field)
    if tensor_dtype in (onnx.TensorProto.COMPLEX64, onnx.TensorProto.COMPLEX128):
        return np.asarray(combine_pairs_to_complex(data)).astype(np_dtype).reshape(dims)

    return np.asarray(data, dtype=storage_np_dtype).astype(np_dtype).reshape(dims)


def from_array(
    arr: np.ndarray | np.generic, name: str | None = None
) -> onnx.TensorProto:
    """Converts a numpy array to a tensor def.

    Args:
        arr: a numpy array.
        name: (optional) the name of the tensor.

    Returns:
        TensorProto: the converted tensor def.

    Raises:
        TypeError: if the input is not a numpy array or np.generic.
    """
    if not isinstance(arr, (np.ndarray, np.generic)):
        raise TypeError(
            f"arr must be of type np.generic or np.ndarray, got {type(arr)}"
        )

    tensor = onnx.TensorProto()
    tensor.dims.extend(arr.shape)
    if name:
        tensor.name = name

    if arr.dtype == object:
        # Special care for strings.
        tensor.data_type = helper.np_dtype_to_tensor_dtype(arr.dtype)
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
                tensor.string_data.append(e.encode("utf-8"))
            elif isinstance(e, np.ndarray):
                for s in e:
                    if isinstance(s, str):
                        tensor.string_data.append(s.encode("utf-8"))
                    elif isinstance(s, bytes):
                        tensor.string_data.append(s)
            elif isinstance(e, bytes):
                tensor.string_data.append(e)
            else:
                raise NotImplementedError(
                    "Unrecognized object in the object array, expect a string, or array of bytes: ",
                    str(type(e)),
                )
        return tensor

    # For numerical types, directly use numpy raw bytes.
    try:
        dtype = helper.np_dtype_to_tensor_dtype(arr.dtype)
    except KeyError as e:
        raise RuntimeError(f"Numpy data type not understood yet: {arr.dtype!r}") from e
    tensor.data_type = dtype

    if sys.byteorder == "big":
        # Convert endian from big to little
        arr = arr.astype(arr.dtype.newbyteorder("<"))

    tensor.raw_data = arr.tobytes()
    return tensor


def to_list(sequence: onnx.SequenceProto) -> list[Any]:
    """Converts a sequence def to a Python list.

    Args:
        sequence: a SequenceProto object.

    Returns:
        list: the converted list.
    """
    elem_type = sequence.elem_type
    if elem_type == onnx.SequenceProto.TENSOR:
        return [to_array(v) for v in sequence.tensor_values]  # type: ignore[arg-type]
    if elem_type == onnx.SequenceProto.SPARSE_TENSOR:
        return [to_array(v) for v in sequence.sparse_tensor_values]  # type: ignore[arg-type]
    if elem_type == onnx.SequenceProto.SEQUENCE:
        return [to_list(v) for v in sequence.sequence_values]
    if elem_type == onnx.SequenceProto.MAP:
        return [to_dict(v) for v in sequence.map_values]
    raise TypeError("The element type in the input sequence is not supported.")


def from_list(
    lst: list[Any], name: str | None = None, dtype: int | None = None
) -> onnx.SequenceProto:
    """Converts a list into a sequence def.

    Args:
        lst: a Python list
        name: The name of the sequence.
        dtype: Type of element in the input list, used for specifying
            sequence values when converting an empty list.

    Returns:
        SequenceProto: the converted sequence def.
    """
    sequence = onnx.SequenceProto()
    if name:
        sequence.name = name

    if dtype:
        elem_type = dtype
    elif len(lst) > 0:
        first_elem = lst[0]
        if isinstance(first_elem, dict):
            elem_type = onnx.SequenceProto.MAP
        elif isinstance(first_elem, list):
            elem_type = onnx.SequenceProto.SEQUENCE
        else:
            elem_type = onnx.SequenceProto.TENSOR
    else:
        # if empty input list and no dtype specified
        # choose sequence of tensors on default
        elem_type = onnx.SequenceProto.TENSOR
    sequence.elem_type = elem_type

    if (len(lst) > 0) and not all(isinstance(elem, type(lst[0])) for elem in lst):
        raise TypeError(
            "The element type in the input list is not the same "
            "for all elements and therefore is not supported as a sequence."
        )

    if elem_type == onnx.SequenceProto.TENSOR:
        for tensor in lst:
            sequence.tensor_values.extend([from_array(tensor)])
    elif elem_type == onnx.SequenceProto.SEQUENCE:
        for seq in lst:
            sequence.sequence_values.extend([from_list(seq)])
    elif elem_type == onnx.SequenceProto.MAP:
        for mapping in lst:
            sequence.map_values.extend([from_dict(mapping)])
    else:
        raise TypeError(
            "The element type in the input list is not a tensor, "
            "sequence, or map and is not supported."
        )
    return sequence


def to_dict(map_proto: onnx.MapProto) -> dict[Any, Any]:
    """Converts a map def to a Python dictionary.

    Args:
        map_proto: a MapProto object.

    Returns:
        The converted dictionary.
    """
    key_list: list[Any] = []
    if map_proto.key_type == onnx.TensorProto.STRING:
        key_list = list(map_proto.string_keys)
    else:
        key_list = list(map_proto.keys)

    value_list = to_list(map_proto.values)
    if len(key_list) != len(value_list):
        raise IndexError(
            "Length of keys and values for MapProto (map name: ",
            map_proto.name,
            ") are not the same.",
        )
    dictionary = dict(zip(key_list, value_list))
    return dictionary


def from_dict(dict_: dict[Any, Any], name: str | None = None) -> onnx.MapProto:
    """Converts a Python dictionary into a map def.

    Args:
        dict_: Python dictionary
        name: (optional) the name of the map.

    Returns:
        MapProto: the converted map def.
    """
    map_proto = onnx.MapProto()
    if name:
        map_proto.name = name
    keys = list(dict_)
    raw_key_type = np.result_type(keys[0])
    key_type = helper.np_dtype_to_tensor_dtype(raw_key_type)

    valid_key_int_types = [
        onnx.TensorProto.INT8,
        onnx.TensorProto.INT16,
        onnx.TensorProto.INT32,
        onnx.TensorProto.INT64,
        onnx.TensorProto.UINT8,
        onnx.TensorProto.UINT16,
        onnx.TensorProto.UINT32,
        onnx.TensorProto.UINT64,
    ]

    if not (
        all(
            np.result_type(key) == raw_key_type  # type: ignore[arg-type]
            for key in keys
        )
    ):
        raise TypeError(
            "The key type in the input dictionary is not the same "
            "for all keys and therefore is not valid as a map."
        )

    values = list(dict_.values())
    raw_value_type = np.result_type(values[0])
    if not all(np.result_type(val) == raw_value_type for val in values):
        raise TypeError(
            "The value type in the input dictionary is not the same "
            "for all values and therefore is not valid as a map."
        )

    value_seq = from_list(values)

    map_proto.key_type = key_type
    if key_type == onnx.TensorProto.STRING:
        map_proto.string_keys.extend(keys)
    elif key_type in valid_key_int_types:
        map_proto.keys.extend(keys)
    map_proto.values.CopyFrom(value_seq)
    return map_proto


def to_optional(optional: onnx.OptionalProto) -> Any | None:
    """Converts an optional def to a Python optional.

    Args:
        optional: an OptionalProto object.

    Returns:
        opt: the converted optional.
    """
    elem_type = optional.elem_type
    if elem_type == onnx.OptionalProto.UNDEFINED:
        return None
    if elem_type == onnx.OptionalProto.TENSOR:
        return to_array(optional.tensor_value)
    if elem_type == onnx.OptionalProto.SPARSE_TENSOR:
        return to_array(optional.sparse_tensor_value)  # type: ignore[arg-type]
    if elem_type == onnx.OptionalProto.SEQUENCE:
        return to_list(optional.sequence_value)
    if elem_type == onnx.OptionalProto.MAP:
        return to_dict(optional.map_value)
    if elem_type == onnx.OptionalProto.OPTIONAL:
        return to_optional(optional.optional_value)
    raise TypeError("The element type in the input optional is not supported.")


def from_optional(
    opt: Any | None, name: str | None = None, dtype: int | None = None
) -> onnx.OptionalProto:
    """Converts an optional value into a Optional def.

    Args:
        opt: a Python optional
        name: The name of the optional.
        dtype: Type of element in the input, used for specifying
            optional values when converting empty none. dtype must
            be a valid OptionalProto.DataType value

    Returns:
        optional: the converted optional def.
    """
    # TODO: create a map and replace conditional branches
    optional = onnx.OptionalProto()
    if name:
        optional.name = name

    if dtype:
        # dtype must be a valid onnx.OptionalProto.DataType
        valid_dtypes = list(onnx.OptionalProto.DataType.values())
        if dtype not in valid_dtypes:
            raise TypeError(f"{dtype} must be a valid onnx.OptionalProto.DataType.")
        elem_type = dtype
    elif isinstance(opt, dict):
        elem_type = onnx.OptionalProto.MAP
    elif isinstance(opt, list):
        elem_type = onnx.OptionalProto.SEQUENCE
    elif opt is None:
        elem_type = onnx.OptionalProto.UNDEFINED
    else:
        elem_type = onnx.OptionalProto.TENSOR

    optional.elem_type = elem_type

    if opt is not None:
        if elem_type == onnx.OptionalProto.TENSOR:
            optional.tensor_value.CopyFrom(from_array(opt))
        elif elem_type == onnx.OptionalProto.SEQUENCE:
            optional.sequence_value.CopyFrom(from_list(opt))
        elif elem_type == onnx.OptionalProto.MAP:
            optional.map_value.CopyFrom(from_dict(opt))
        else:
            raise TypeError(
                "The element type in the input is not a tensor, "
                "sequence, or map and is not supported."
            )
    return optional


def create_random_int(
    input_shape: tuple[int], dtype: np.dtype, seed: int = 1
) -> np.ndarray:
    """Create random integer array for backend/test/case/node.

    Args:
        input_shape: The shape for the returned integer array.
        dtype: The NumPy data type for the returned integer array.
        seed: The seed for np.random.

    Returns:
        Random integer array.
    """
    np.random.seed(seed)
    if dtype in (
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
    ):
        # the range of np.random.randint is int32; set a fixed boundary if overflow
        end = min(np.iinfo(dtype).max, np.iinfo(np.int32).max)
        start = max(np.iinfo(dtype).min, np.iinfo(np.int32).min)
        return np.random.randint(start, end, size=input_shape).astype(dtype)
    else:
        raise TypeError(f"{dtype} is not supported by create_random_int.")
