# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import string

import numpy as np

import onnx
from onnx import TensorProto
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


def typeof(data_type: TensorProto.DataType) -> np.dtype:
    custom_type_map = {
        TensorProto.BFLOAT16: np.dtype("uint16"),
        TensorProto.FLOAT16: np.dtype("uint16"),
        TensorProto.FLOAT8E4M3FN: np.dtype("uint8"),
        TensorProto.FLOAT8E4M3FNUZ: np.dtype("uint8"),
        TensorProto.FLOAT8E5M2: np.dtype("uint8"),
        TensorProto.FLOAT8E5M2FNUZ: np.dtype("uint8"),
        TensorProto.FLOAT4E2M1: np.dtype("uint8"),
        TensorProto.INT4: np.dtype("uint8"),
        TensorProto.UINT4: np.dtype("uint8"),
    }

    return custom_type_map.get(
        data_type, onnx.helper.tensor_dtype_to_np_dtype(data_type)
    )


def expected_bitcast(input: TensorProto, to: TensorProto.DataType) -> TensorProto:
    # Extract raw tensor data
    data = np.array(
        getattr(input, onnx.helper.tensor_dtype_to_field(input.data_type)),
        dtype=typeof(input.data_type),
    )
    if input.data_type == TensorProto.STRING:
        data = data.astype("S")

    four_bit_types = [
        TensorProto.INT4,
        TensorProto.UINT4,
        TensorProto.FLOAT4E2M1,
    ]
    from_size = data.itemsize if input.data_type not in four_bit_types else 0.5

    to_data = np.frombuffer(
        data.tobytes(),
        dtype=(
            typeof(to)
            if to != TensorProto.STRING
            else "S" + str(int(input.dims[-1] * from_size))
        ),
    )

    to_size = to_data.itemsize if to not in four_bit_types else 0.5
    shape = tuple(input.dims)

    if from_size > to_size:
        shape = (*shape, int(from_size // to_size))
    elif from_size < to_size:
        shape = shape[:-1]

    output_tensor = onnx.TensorProto()
    output_tensor.name = "output"
    output_tensor.dims.extend(shape)
    output_tensor.data_type = to
    getattr(output_tensor, onnx.helper.tensor_dtype_to_field(to)).extend(
        to_data.flatten()
    )

    return output_tensor


class BitCast(Base):
    @staticmethod
    def export() -> None:
        # Test that bitcasting some sample data
        # from one data type to another gives
        # expected results.

        shape = np.array([32], dtype=np.int32)

        # Random combinations of bits for each size
        from_data_4 = np.random.randint(15, size=shape, dtype=np.uint8)
        from_data_8 = np.random.randint(255, size=shape, dtype=np.uint8)
        from_data_16 = np.random.randint(65535, size=shape, dtype=np.uint16)
        from_data_32 = np.random.randint(4294967295, size=shape, dtype=np.uint32)
        from_data_64 = np.random.randint(
            18446744073709551615, size=shape, dtype=np.uint64
        )

        # Random string of characters
        string_data = np.array(
            [
                "".join(np.random.choice(list(string.ascii_letters), size=(8)))
                for _ in range(32)
            ]
        )

        # Random true/false values
        bool_data = np.random.choice([True, False], size=shape)

        # Test data to use for each data type
        data_map = {
            TensorProto.FLOAT: from_data_32,
            TensorProto.UINT8: from_data_8,
            TensorProto.INT8: from_data_8,
            TensorProto.UINT16: from_data_16,
            TensorProto.INT16: from_data_16,
            TensorProto.INT32: from_data_32,
            TensorProto.INT64: from_data_64,
            TensorProto.FLOAT16: from_data_16,
            TensorProto.DOUBLE: from_data_64,
            TensorProto.UINT32: from_data_32,
            TensorProto.UINT64: from_data_64,
            TensorProto.BFLOAT16: from_data_16,
            TensorProto.FLOAT8E4M3FN: from_data_8,
            TensorProto.FLOAT8E4M3FNUZ: from_data_8,
            TensorProto.FLOAT8E5M2: from_data_8,
            TensorProto.FLOAT8E5M2FNUZ: from_data_8,
            TensorProto.UINT4: from_data_4,
            TensorProto.INT4: from_data_4,
            TensorProto.FLOAT4E2M1: from_data_4,
            TensorProto.STRING: string_data,
            TensorProto.BOOL: bool_data,
        }

        # Same as the Cast test cases
        test_cases = [
            (TensorProto.FLOAT, TensorProto.FLOAT16),
            (TensorProto.FLOAT, TensorProto.DOUBLE),
            (TensorProto.FLOAT16, TensorProto.FLOAT),
            (TensorProto.FLOAT16, TensorProto.DOUBLE),
            (TensorProto.DOUBLE, TensorProto.FLOAT),
            (TensorProto.DOUBLE, TensorProto.FLOAT16),
            (TensorProto.FLOAT, TensorProto.STRING),
            (TensorProto.STRING, TensorProto.FLOAT),
            (TensorProto.FLOAT, TensorProto.BFLOAT16),
            (TensorProto.BFLOAT16, TensorProto.FLOAT),
            (TensorProto.FLOAT, TensorProto.FLOAT8E4M3FN),
            (TensorProto.FLOAT16, TensorProto.FLOAT8E4M3FN),
            (TensorProto.FLOAT, TensorProto.FLOAT8E4M3FNUZ),
            (TensorProto.FLOAT16, TensorProto.FLOAT8E4M3FNUZ),
            (TensorProto.FLOAT8E4M3FN, TensorProto.FLOAT),
            (TensorProto.FLOAT8E4M3FN, TensorProto.FLOAT16),
            (TensorProto.FLOAT8E4M3FNUZ, TensorProto.FLOAT),
            (TensorProto.FLOAT8E4M3FNUZ, TensorProto.FLOAT16),
            (TensorProto.FLOAT, TensorProto.FLOAT8E5M2),
            (TensorProto.FLOAT16, TensorProto.FLOAT8E5M2),
            (TensorProto.FLOAT, TensorProto.FLOAT8E5M2FNUZ),
            (TensorProto.FLOAT16, TensorProto.FLOAT8E5M2FNUZ),
            (TensorProto.FLOAT8E5M2, TensorProto.FLOAT),
            (TensorProto.FLOAT8E5M2, TensorProto.FLOAT16),
            (TensorProto.FLOAT8E5M2FNUZ, TensorProto.FLOAT),
            (TensorProto.FLOAT8E5M2FNUZ, TensorProto.FLOAT16),
            (TensorProto.FLOAT, TensorProto.UINT4),
            (TensorProto.FLOAT16, TensorProto.UINT4),
            (TensorProto.FLOAT, TensorProto.INT4),
            (TensorProto.FLOAT16, TensorProto.INT4),
            (TensorProto.UINT4, TensorProto.FLOAT),
            (TensorProto.UINT4, TensorProto.FLOAT16),
            (TensorProto.UINT4, TensorProto.UINT8),
            (TensorProto.INT4, TensorProto.FLOAT),
            (TensorProto.INT4, TensorProto.FLOAT16),
            (TensorProto.INT4, TensorProto.INT8),
            (TensorProto.FLOAT4E2M1, TensorProto.FLOAT),
            (TensorProto.FLOAT4E2M1, TensorProto.FLOAT16),
            (TensorProto.FLOAT, TensorProto.FLOAT4E2M1),
            (TensorProto.FLOAT16, TensorProto.FLOAT4E2M1),
        ]

        for from_type, to_type in test_cases:
            from_data = data_map[from_type]

            # If we are converting to string then make sure
            # we are only using values in the ASCII character
            # range.
            if to_type == TensorProto.STRING:
                from_data = from_data.clip(33, 126)

            node = onnx.helper.make_node(
                "BitCast",
                inputs=["input"],
                outputs=["output"],
                to=to_type,
            )

            four_bit_types = [
                TensorProto.INT4,
                TensorProto.UINT4,
                TensorProto.FLOAT4E2M1,
            ]
            from_size = (
                typeof(from_type).itemsize if from_type not in four_bit_types else 0.5
            )
            to_size = typeof(to_type).itemsize if to_type not in four_bit_types else 0.5

            if to_size > from_size:
                from_data = from_data.reshape(-1, int(to_size // from_size))

            if to_type == TensorProto.STRING:
                from_data = from_data.reshape(-1, 4)

            if from_type not in [TensorProto.STRING, TensorProto.BOOL]:
                from_data = from_data.view(typeof(from_type))

            input_tensor = onnx.helper.make_tensor(
                "input", from_type, from_data.shape, from_data
            )
            output_tensor = expected_bitcast(input_tensor, to_type)

            from_str = onnx.helper.tensor_dtype_to_string(from_type).split(".")[-1]
            to_str = onnx.helper.tensor_dtype_to_string(to_type).split(".")[-1]
            expect(
                node,
                [input_tensor],
                [output_tensor],
                f"test_bitcast_{from_str}_to_{to_str}",
            )

    @staticmethod
    def export_bitcast_higher_dimensionality_to_larger_type() -> None:
        shape = (32, 32, 2)
        from_data = np.random.randint(255, size=shape, dtype=np.uint8)

        node = onnx.helper.make_node(
            "BitCast",
            inputs=["input"],
            outputs=["output"],
            to=TensorProto.UINT16,
        )

        input_tensor = onnx.helper.make_tensor(
            "input", TensorProto.UINT8, from_data.shape, from_data
        )
        output_tensor = expected_bitcast(input_tensor, TensorProto.UINT16)
        expect(
            node,
            [input_tensor],
            [output_tensor],
            "test_bitcast_higher_dimensionality_to_larger_type",
        )

    @staticmethod
    def export_bitcast_higher_dimensionality_to_smaller_type() -> None:
        shape = (32, 32, 2)
        from_data = np.random.randint(65535, size=shape, dtype=np.uint16)

        node = onnx.helper.make_node(
            "BitCast",
            inputs=["input"],
            outputs=["output"],
            to=TensorProto.UINT8,
        )

        input_tensor = onnx.helper.make_tensor(
            "input", TensorProto.UINT16, from_data.shape, from_data
        )
        output_tensor = expected_bitcast(input_tensor, TensorProto.UINT8)
        expect(
            node,
            [input_tensor],
            [output_tensor],
            "test_bitcast_higher_dimensionality_to_smaller_type",
        )

    @staticmethod
    def export_bitcast_higher_dimensionality_to_larger_type_string() -> None:
        from_data = np.array(
            [
                [
                    "".join(np.random.choice(list(string.ascii_letters), size=(8)))
                    for _ in range(32)
                ]
            ]
            * 32
        )

        node = onnx.helper.make_node(
            "BitCast",
            inputs=["input"],
            outputs=["output"],
            to=TensorProto.UINT16,
        )

        input_tensor = onnx.helper.make_tensor(
            "input", TensorProto.STRING, from_data.shape, from_data
        )
        output_tensor = expected_bitcast(input_tensor, TensorProto.UINT16)
        expect(
            node,
            [input_tensor],
            [output_tensor],
            "test_bitcast_higher_dimensionality_to_larger_type_string",
        )

    @staticmethod
    def export_bitcast_higher_dimensionality_to_smaller_type_string() -> None:
        from_data = np.array(
            [
                [
                    "".join(np.random.choice(list(string.ascii_letters), size=(8)))
                    for _ in range(32)
                ]
            ]
            * 32
        )

        node = onnx.helper.make_node(
            "BitCast",
            inputs=["input"],
            outputs=["output"],
            to=TensorProto.UINT8,
        )

        input_tensor = onnx.helper.make_tensor(
            "input", TensorProto.STRING, from_data.shape, from_data
        )
        output_tensor = expected_bitcast(input_tensor, TensorProto.UINT8)
        expect(
            node,
            [input_tensor],
            [output_tensor],
            "test_bitcast_higher_dimensionality_to_smaller_type_string",
        )

    @staticmethod
    def export_bitcast_from_scalar() -> None:
        shape = ()
        from_data = np.random.randint(65535, size=shape, dtype=np.uint16)

        node = onnx.helper.make_node(
            "BitCast",
            inputs=["input"],
            outputs=["output"],
            to=TensorProto.UINT8,
        )

        input_tensor = onnx.helper.make_tensor(
            "input", TensorProto.UINT16, from_data.shape, from_data.reshape(1)
        )
        output_tensor = expected_bitcast(input_tensor, TensorProto.UINT8)
        expect(
            node,
            [input_tensor],
            [output_tensor],
            "test_bitcast_from_scalar",
        )

    @staticmethod
    def export_bitcast_to_scalar() -> None:
        shape = 2
        from_data = np.random.randint(255, size=shape, dtype=np.uint8)

        node = onnx.helper.make_node(
            "BitCast",
            inputs=["input"],
            outputs=["output"],
            to=TensorProto.UINT16,
        )

        input_tensor = onnx.helper.make_tensor(
            "input", TensorProto.UINT8, from_data.shape, from_data
        )
        output_tensor = expected_bitcast(input_tensor, TensorProto.UINT16)
        expect(
            node,
            [input_tensor],
            [output_tensor],
            "test_bitcast_to_scalar",
        )

    @staticmethod
    def export_bitcast_from_scalar_string() -> None:
        from_data = np.array("AB")

        node = onnx.helper.make_node(
            "BitCast",
            inputs=["input"],
            outputs=["output"],
            to=TensorProto.UINT8,
        )

        input_tensor = onnx.helper.make_tensor(
            "input", TensorProto.STRING, from_data.shape, from_data.reshape(1)
        )
        output_tensor = expected_bitcast(input_tensor, TensorProto.UINT8)
        expect(
            node,
            [input_tensor],
            [output_tensor],
            "test_bitcast_from_scalar_string",
        )

    @staticmethod
    def export_bitcast_to_scalar_string() -> None:
        from_data = np.array([65, 66])

        node = onnx.helper.make_node(
            "BitCast",
            inputs=["input"],
            outputs=["output"],
            to=TensorProto.STRING,
        )

        input_tensor = onnx.helper.make_tensor(
            "input", TensorProto.UINT8, from_data.shape, from_data
        )
        output_tensor = expected_bitcast(input_tensor, TensorProto.STRING)
        expect(
            node,
            [input_tensor],
            [output_tensor],
            "test_bitcast_to_scalar_string",
        )
