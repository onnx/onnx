# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import ml_dtypes
import numpy as np
import numpy.typing as npt
import typing_extensions

import onnx.external_data_helper
from onnx import helper, subbyte

if TYPE_CHECKING:
    from collections.abc import Sequence

# System is little endian
_IS_LITTLE_ENDIAN = sys.byteorder == "little"


@typing_extensions.deprecated(
    "Deprecated since 1.18. Scheduled to remove in 1.20. Consider using libraries like ml_dtypes for dtype conversion",
    category=DeprecationWarning,
)
def bfloat16_to_float32(
    data: np.int16 | np.int32 | np.ndarray,
    dims: int | Sequence[int] | None = None,
) -> np.ndarray:
    """Converts ndarray of bf16 (as uint32) to f32 (as uint32).

    Args:
        data: A numpy array, empty dimensions are allowed if dims is
            None.
        dims: If specified, the function reshapes the results.

    Returns:
        A numpy array of float32 with the same dimension if dims is
        None, or reshaped to dims if specified
    """
    shift = lambda x: x << 16  # noqa: E731
    if dims is None:
        if len(data.shape) == 0:
            return shift(np.array([data]).astype(np.int32)).view(np.float32)[0]  # type: ignore[no-any-return]
        return shift(data.astype(np.int32)).view(np.float32)  # type: ignore[no-any-return]
    return shift(data.astype(np.int32)).reshape(dims).view(np.float32)  # type: ignore[no-any-return]


def _float8e4m3_to_float32_scalar(ival: int, fn: bool, uz: bool) -> np.float32:
    if not fn:
        raise NotImplementedError("fn=False is not implemented.")
    if ival < 0 or ival > 255:  # noqa: PLR2004
        raise ValueError(f"{ival} is not a float8.")
    if uz:
        exponent_bias = 8
        if ival == 0x80:  # noqa: PLR2004
            return np.nan  # type: ignore[return-value]
    else:
        exponent_bias = 7
        if ival == 255:  # noqa: PLR2004
            return np.float32(-np.nan)
        if ival == 127:  # noqa: PLR2004
            return np.float32(np.nan)

    ival = np.uint32(ival)  # type: ignore[assignment]
    expo = (ival & 0x78) >> 3
    mant = ival & 0x07
    sign = ival & 0x80
    res = sign << 24
    if expo == 0:
        if mant > 0:
            expo = 0x7F - exponent_bias
            if mant & 0x4 == 0:
                mant &= 0x3
                mant <<= 1
                expo -= 1
            if mant & 0x4 == 0:
                mant &= 0x3
                mant <<= 1
                expo -= 1
            res |= (mant & 0x3) << 21
            res |= expo << 23
    else:
        res |= mant << 20
        expo += 0x7F - exponent_bias
        res |= expo << 23
    f = np.uint32(res).view(np.float32)
    return f


_float8e4m3_to_float32 = np.vectorize(
    _float8e4m3_to_float32_scalar, excluded=["fn", "uz"]
)


@typing_extensions.deprecated(
    "Deprecated since 1.18. Scheduled to remove in 1.20. Consider using libraries like ml_dtypes for dtype conversion",
    category=DeprecationWarning,
)
def float8e4m3_to_float32(
    data: np.int16 | np.int32 | np.ndarray,
    dims: int | Sequence[int] | None = None,
    fn: bool = True,
    uz: bool = False,
) -> np.ndarray:
    """Converts ndarray of float8, e4m3 (as uint32) to f32 (as uint32).

    See :ref:`onnx-detail-float8` for technical details.

    Args:
        data: A numpy array, empty dimensions are allowed if dims is None.
        dims: If specified, the function reshapes the results.
        fn: No infinite values.
        uz: No negative zero.

    Returns:
        A numpy array of float32 with the same dimension if dims is None,
        or reshaped to dims if specified.
    """
    if not fn:
        raise NotImplementedError(
            "float32_to_float8e4m3 not implemented with fn=False."
        )
    res = _float8e4m3_to_float32(data, fn=fn, uz=uz)
    if dims is None:
        return res  # type: ignore[no-any-return]
    return res.reshape(dims)  # type: ignore[no-any-return]


def _float8e5m2_to_float32_scalar(ival: int, fn: bool, uz: bool) -> np.float32:
    if fn and uz:
        if ival == 0x80:  # noqa: PLR2004
            return np.float32(np.nan)
        exponent_bias = 16
    elif not fn and not uz:
        if ival in {253, 254, 255}:
            return np.float32(-np.nan)
        if ival in {125, 126, 127}:
            return np.float32(np.nan)
        if ival == 252:  # noqa: PLR2004
            return np.float32(-np.inf)
        if ival == 124:  # noqa: PLR2004
            return np.float32(np.inf)
        exponent_bias = 15
    else:
        raise NotImplementedError("fn and uz must be both False or True.")

    ival = np.uint32(ival)  # type: ignore[assignment]
    expo = (ival & 0x7C) >> 2
    mant = ival & 0x03
    sign = ival & 0x80
    res = sign << 24
    if expo == 0:
        if mant > 0:
            expo = 0x7F - exponent_bias
            if mant & 0x2 == 0:
                mant &= 0x1
                mant <<= 1
                expo -= 1
            res |= (mant & 0x1) << 22
            res |= expo << 23
    else:
        res |= mant << 21
        expo += 0x7F - exponent_bias
        res |= expo << 23
    f = np.uint32(res).view(np.float32)
    return f


_float8e5m2_to_float32 = np.vectorize(
    _float8e5m2_to_float32_scalar, excluded=["fn", "uz"]
)


@typing_extensions.deprecated(
    "Deprecated since 1.18. Scheduled to remove in 1.20. Consider using libraries like ml_dtypes for dtype conversion",
    category=DeprecationWarning,
)
def float8e5m2_to_float32(
    data: np.int16 | np.int32 | np.ndarray,
    dims: int | Sequence[int] | None = None,
    fn: bool = False,
    uz: bool = False,
) -> np.ndarray:
    """Converts ndarray of float8, e5m2 (as uint32) to f32 (as uint32).

    See :ref:`onnx-detail-float8` for technical details.

    Args:
        data: A numpy array, empty dimensions are allowed if dims is None.
        dims: If specified, the function reshapes the results.
        fn: No infinite values.
        uz: No negative zero.

    Returns:
        A numpy array of float32 with the same dimension if dims is None,
        or reshaped to dims if specified
    """
    res = _float8e5m2_to_float32(data, fn=fn, uz=uz)
    if dims is None:
        return res  # type: ignore[no-any-return]
    return res.reshape(dims)  # type: ignore[no-any-return]


def to_float8e8m0(
    x: np.ndarray,
    saturate: bool = True,
    round_mode: str = "up",
) -> np.ndarray:
    """Convert float32 NumPy array to float8e8m0 representation. If the input
    is not a float32 array, it will be cast to one first.

    Args:
        x: Input array to convert.
        saturate: Whether to saturate at max/min float8e8m0 value.
        round_mode: "nearest", "up", or "down".

    Returns:
        np.ndarray: Array of ml_dtypes.float8_e8m0fnu values.
    """
    x_f32 = np.asarray(x, dtype=np.float32)
    f_bits = x_f32.view(np.uint32)

    # Extract exponent bits
    exponent = (f_bits >> 23) & 0xFF
    exponent = exponent.astype(
        np.uint16
    )  # use uint16 to prevent overflow during computation

    # Identify NaN or Inf
    special_mask = exponent == 0xFF  # noqa: PLR2004
    output = np.zeros_like(exponent, dtype=np.uint8)
    output[special_mask] = 0xFF  # Preserve NaN/Inf as max exponent

    # Process normal numbers
    normal_mask = ~special_mask

    if round_mode == "nearest":
        # Get guard, round, sticky, and least significant bits
        g = ((f_bits & 0x400000) > 0).astype(np.uint8)
        r = ((f_bits & 0x200000) > 0).astype(np.uint8)
        s = ((f_bits & 0x1FFFFF) > 0).astype(np.uint8)
        lsb = (exponent > 0).astype(np.uint8)

        round_up = (g == 1) & ((r == 1) | (s == 1) | (lsb == 1))

        increment = np.zeros_like(exponent)
        increment[round_up & normal_mask] = 1

        if saturate:
            max_mask = (exponent == 0xFE) & round_up & normal_mask  # noqa: PLR2004
            increment[max_mask] = 0  # Don't overflow past max value

        exponent += increment

    elif round_mode == "up":
        has_fraction = (f_bits & 0x4FFFFF) > 0
        round_up = has_fraction & normal_mask

        if saturate:
            max_mask = (exponent == 0xFE) & round_up  # noqa: PLR2004
            round_up[max_mask] = False

        exponent += round_up.astype(np.uint16)

    elif round_mode == "down":
        pass  # No rounding needed

    else:
        raise ValueError(f"Unsupported rounding mode: {round_mode}")

    # Clip exponent to uint8 range
    exponent = exponent.astype(np.uint8)

    output[normal_mask] = exponent[normal_mask]

    return output.view(ml_dtypes.float8_e8m0fnu)


@typing_extensions.deprecated(
    "Deprecated since 1.18. Scheduled to remove in 1.20. Consider implementing your own unpack logic",
    category=DeprecationWarning,
)
def unpack_int4(
    data: np.int32 | np.ndarray,
    dims: int | Sequence[int],
    signed: bool,
) -> np.ndarray:
    """Converts ndarray of int4 (as packed uint8) to f32
    See :ref:`onnx-detail-int4` for technical details.

    Args:
        data: A numpy array, empty dimensions are allowed if dims is
            None.
        dims: The dimensions are used to reshape the unpacked buffer
        signed: Whether the 4 bit integer is signed or unsigned

    Returns:
        A numpy array of float32 reshaped to dims.
    """
    single_func = lambda x: subbyte.unpack_single_4bitx2(x, signed)  # noqa: E731
    func = np.frompyfunc(single_func, 1, 2)

    res_high, res_low = func(data.ravel())
    res = np.empty((res_high.size + res_low.size,), dtype=np.float32)
    res[0::2] = res_high
    res[1::2] = res_low

    if (
        res.size == np.prod(dims) + 1
    ):  # handle single-element padding due to odd number of elements
        res = res.ravel()[:-1]
    res = res.reshape(dims)
    return res


def _unpacked_float4e2m1_to_float32(
    x: npt.NDArray[np.uint8],
) -> npt.NDArray[np.float32]:
    """Evaluate the numerical value of an array of unpacked float4e2m1 values (as uint8)
    See :ref:`onnx-detail-int4` for technical details.

    Args:
        x: an array of uint8 elements representing a float4e2m1 (using the 4 LSB)

    Returns:
        An array of float32 elements representing the values of the float4e2m1 input.
    """
    # x is stored in 4 LSB of int
    sign = np.where(np.bitwise_and(x, 0x08), -1, 1)
    mantissa = (x & 0x01).astype(np.float32)
    exponent = ((x & 0x06) >> 1).astype(np.float32)

    val = np.where(
        exponent == 0,
        sign * (mantissa / 2.0),
        sign * (1.0 + mantissa / 2.0) * 2.0 ** (exponent - 1),
    )  # denormalized, normalized
    return val


def _unpack_4bit(
    data: npt.NDArray[np.uint8], dims: Sequence[int]
) -> npt.NDArray[np.uint8]:
    """Convert a packed uint4 array to unpacked uint4 array represented as uint8.

    Args:
        data: A numpy array.
        dims: The dimensions are used to reshape the unpacked buffer.

    Returns:
        A numpy array of int8/uint8 reshaped to dims.
    """
    result = np.empty([data.size * 2], dtype=data.dtype)
    array_low = data & np.uint8(0x0F)
    array_high = data & np.uint8(0xF0)
    array_high >>= np.uint8(4)
    result[0::2] = array_low
    result[1::2] = array_high
    if result.size == np.prod(dims) + 1:
        # handle single-element padding due to odd number of elements
        result = result[:-1]
    result.resize(dims, refcheck=False)
    return result


def _pack_4bitx2(array: np.ndarray) -> npt.NDArray[np.uint8]:
    """Convert a numpy array to flatten, packed int4/uint4. Elements must be in the correct range."""
    # Create a 1D copy
    array_flat = array.ravel().view(np.uint8).copy()
    size = array.size
    odd_sized = size % 2 == 1
    if odd_sized:
        array_flat.resize([size + 1], refcheck=False)
    array_flat &= 0x0F
    array_flat[1::2] <<= 4
    return array_flat[0::2] | array_flat[1::2]  # type: ignore[return-type]


def to_array(tensor: onnx.TensorProto, base_dir: str = "") -> np.ndarray:  # noqa: PLR0911
    """Converts a tensor def object to a numpy array.

    This function uses ml_dtypes if the dtype is not a native numpy dtype.

    Args:
        tensor: a TensorProto object.
        base_dir: if external tensor exists, base_dir can help to find the path to it

    Returns:
        arr: the converted array.
    """
    if tensor.HasField("segment"):
        raise ValueError("Currently not supporting loading segments.")
    if tensor.data_type == onnx.TensorProto.UNDEFINED:
        raise TypeError("The element type in the input tensor is UNDEFINED.")

    tensor_dtype = tensor.data_type
    np_dtype = helper.tensor_dtype_to_np_dtype(tensor_dtype)
    storage_np_dtype = helper.tensor_dtype_to_np_dtype(
        helper.tensor_dtype_to_storage_tensor_dtype(tensor_dtype)
    )
    storage_field = helper.tensor_dtype_to_field(tensor_dtype)
    dims = tensor.dims

    if tensor.data_type == onnx.TensorProto.STRING:
        utf8_strings = getattr(tensor, storage_field)
        ss = [s.decode("utf-8") for s in utf8_strings]
        return np.asarray(ss).astype(np_dtype).reshape(dims)

    # Load raw data from external tensor if it exists
    if onnx.external_data_helper.uses_external_data(tensor):
        onnx.external_data_helper.load_external_data_for_tensor(tensor, base_dir)

    if tensor.HasField("raw_data"):
        # Raw_bytes support: using frombuffer.
        raw_data = tensor.raw_data
        if sys.byteorder == "big":
            # Convert endian from little to big
            raw_data = np.frombuffer(raw_data, dtype=np_dtype).byteswap().tobytes()

        if tensor_dtype in {
            onnx.TensorProto.INT4,
            onnx.TensorProto.UINT4,
            onnx.TensorProto.FLOAT4E2M1,
        }:
            data = np.frombuffer(raw_data, dtype=np.uint8)
            return _unpack_4bit(data, dims).view(np_dtype)

        return np.frombuffer(raw_data, dtype=np_dtype).reshape(dims)

    if tensor_dtype in {
        onnx.TensorProto.BFLOAT16,
        onnx.TensorProto.FLOAT16,
        onnx.TensorProto.INT16,
        onnx.TensorProto.UINT16,
    }:
        return (
            np.array(tensor.int32_data, dtype=np.int32)
            .view(np.uint32)
            .astype(np.uint16)
            .reshape(dims)
            .view(np_dtype)
        )

    if tensor_dtype in {
        onnx.TensorProto.FLOAT8E4M3FN,
        onnx.TensorProto.FLOAT8E4M3FNUZ,
        onnx.TensorProto.FLOAT8E5M2,
        onnx.TensorProto.FLOAT8E5M2FNUZ,
        onnx.TensorProto.FLOAT8E8M0,
        onnx.TensorProto.BOOL,
    }:
        return (
            np.array(tensor.int32_data, dtype=np.int32)
            .view(np.uint32)
            .astype(np.uint8)
            .view(np_dtype)
            .reshape(dims)
        )

    if tensor_dtype in {
        onnx.TensorProto.UINT4,
        onnx.TensorProto.INT4,
        onnx.TensorProto.FLOAT4E2M1,
    }:
        data = (
            np.array(tensor.int32_data, dtype=np.int32).view(np.uint32).astype(np.uint8)
        )
        return _unpack_4bit(data, dims).view(np_dtype)

    data = getattr(tensor, storage_field)
    if tensor_dtype in (onnx.TensorProto.COMPLEX64, onnx.TensorProto.COMPLEX128):
        return np.array(data, dtype=storage_np_dtype).view(dtype=np_dtype).reshape(dims)

    return np.asarray(data, dtype=storage_np_dtype).astype(np_dtype).reshape(dims)


def from_array(array: np.ndarray, /, name: str | None = None) -> onnx.TensorProto:
    """Converts an array into a TensorProto including

    Args:
        array: a numpy array.
        name: (optional) the name of the tensor.

    Returns:
        TensorProto: the converted tensor def.
    """
    tensor = onnx.TensorProto()
    tensor.dims.extend(array.shape)
    if name:
        tensor.name = name
    if array.dtype == object or np.issubdtype(array.dtype, np.str_):
        # Special care for strings.
        tensor.data_type = onnx.TensorProto.STRING
        # TODO: Introduce full string support.
        # We flatten the array in case there are n-D arrays are specified
        # If you want more complex shapes then follow the below instructions.
        # Unlike other types where the shape is automatically inferred from
        # nested arrays of values, the only reliable way now to feed strings
        # is to put them into a flat array then specify type astype(object)
        # (otherwise all strings may have different types depending on their length)
        # and then specify shape .reshape([x, y, z])
        flat_array = array.flatten()
        for e in flat_array:
            if isinstance(e, str):
                tensor.string_data.append(e.encode("utf-8"))
            elif isinstance(e, bytes):
                tensor.string_data.append(e)
            else:
                raise NotImplementedError(
                    "Unrecognized object in the object array, expect a string, or array of bytes: ",
                    str(type(e)),
                )
        return tensor

    dtype = helper.np_dtype_to_tensor_dtype(array.dtype)
    if dtype in {
        onnx.TensorProto.INT4,
        onnx.TensorProto.UINT4,
        onnx.TensorProto.FLOAT4E2M1,
    }:
        # Pack the array into int4
        array = _pack_4bitx2(array)
    if not _IS_LITTLE_ENDIAN:
        array = array.view(array.dtype.newbyteorder("<"))

    tensor.raw_data = array.tobytes()
    tensor.data_type = dtype
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
        return [to_array(v) for v in sequence.tensor_values]
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
        name: (optional) the name of the sequence.
        dtype: (optional) type of element in the input list, used for specifying
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
            sequence.tensor_values.extend([from_array(np.asarray(tensor))])
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

    valid_key_int_types = {
        onnx.TensorProto.INT8,
        onnx.TensorProto.INT16,
        onnx.TensorProto.INT32,
        onnx.TensorProto.INT64,
        onnx.TensorProto.UINT8,
        onnx.TensorProto.UINT16,
        onnx.TensorProto.UINT32,
        onnx.TensorProto.UINT64,
    }

    if not (all(np.result_type(key) == raw_key_type for key in keys)):
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
        name: (optional) the name of the optional.
        dtype: (optional) type of element in the input, used for specifying
                          optional values when converting empty none. dtype must
                          be a valid OptionalProto.DataType value

    Returns:
        optional: the converted optional def.
    """
    # TODO: create a map and replace conditional branches
    optional = onnx.OptionalProto()
    if name:
        optional.name = name

    if dtype is not None:
        # dtype must be a valid onnx.OptionalProto.DataType
        if dtype not in onnx.OptionalProto.DataType.values():
            raise TypeError(f"{dtype} must be a valid OptionalProto.DataType.")
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
        np.ndarray: Random integer array.
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


def saturate_cast(x: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Saturate cast for numeric types.

    This function ensures that values outside the representable range
    of the target dtype are clamped to the maximum or minimum representable
    value of that dtype.
    """
    if np.issubdtype(dtype, np.integer) or dtype in (ml_dtypes.int4, ml_dtypes.uint4):
        info = ml_dtypes.iinfo(dtype)
        x = np.round(x)
    else:
        info = ml_dtypes.finfo(dtype)  # type: ignore[assignment]

    return np.clip(x, info.min, info.max).astype(dtype)
