# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

import numpy as np
import parameterized

import onnx
import onnx.reference
from onnx import helper, numpy_helper


class TestNumpyHelper(unittest.TestCase):
    def _test_numpy_helper_float_type(self, dtype: np.number) -> None:
        a = np.random.rand(13, 37).astype(dtype)
        tensor_def = numpy_helper.from_array(a, "test")
        self.assertEqual(tensor_def.name, "test")
        a_recover = numpy_helper.to_array(tensor_def)
        np.testing.assert_equal(a, a_recover)

    def _test_numpy_helper_int_type(self, dtype: np.number) -> None:
        a = np.random.randint(
            np.iinfo(dtype).min, np.iinfo(dtype).max, dtype=dtype, size=(13, 37)
        )
        tensor_def = numpy_helper.from_array(a, "test")
        self.assertEqual(tensor_def.name, "test")
        a_recover = numpy_helper.to_array(tensor_def)
        np.testing.assert_equal(a, a_recover)

    def test_float(self) -> None:
        self._test_numpy_helper_float_type(np.float32)

    def test_uint8(self) -> None:
        self._test_numpy_helper_int_type(np.uint8)

    def test_int8(self) -> None:
        self._test_numpy_helper_int_type(np.int8)

    def test_uint16(self) -> None:
        self._test_numpy_helper_int_type(np.uint16)

    def test_int16(self) -> None:
        self._test_numpy_helper_int_type(np.int16)

    def test_int32(self) -> None:
        self._test_numpy_helper_int_type(np.int32)

    def test_int64(self) -> None:
        self._test_numpy_helper_int_type(np.int64)

    def test_string(self) -> None:
        a = np.array(["Amy", "Billy", "Cindy", "David"]).astype(object)
        tensor_def = numpy_helper.from_array(a, "test")
        self.assertEqual(tensor_def.name, "test")
        a_recover = numpy_helper.to_array(tensor_def)
        np.testing.assert_equal(a, a_recover)

    def test_bool(self) -> None:
        a = np.random.randint(2, size=(13, 37)).astype(bool)
        tensor_def = numpy_helper.from_array(a, "test")
        self.assertEqual(tensor_def.name, "test")
        a_recover = numpy_helper.to_array(tensor_def)
        np.testing.assert_equal(a, a_recover)

    def test_float16(self) -> None:
        self._test_numpy_helper_float_type(np.float32)

    def test_complex64(self) -> None:
        self._test_numpy_helper_float_type(np.complex64)

    def test_complex128(self) -> None:
        self._test_numpy_helper_float_type(np.complex128)

    def test_from_dict_values_are_np_arrays_of_float(self):
        map_proto = numpy_helper.from_dict({0: np.array(0.1), 1: np.array(0.9)})
        self.assertIsInstance(map_proto, onnx.MapProto)
        self.assertEqual(
            numpy_helper.to_array(map_proto.values.tensor_values[0]), np.array(0.1)
        )
        self.assertEqual(
            numpy_helper.to_array(map_proto.values.tensor_values[1]), np.array(0.9)
        )

    def test_from_dict_values_are_np_arrays_of_int(self):
        map_proto = numpy_helper.from_dict({0: np.array(1), 1: np.array(9)})
        self.assertIsInstance(map_proto, onnx.MapProto)
        self.assertEqual(
            numpy_helper.to_array(map_proto.values.tensor_values[0]), np.array(1)
        )
        self.assertEqual(
            numpy_helper.to_array(map_proto.values.tensor_values[1]), np.array(9)
        )

    def test_from_dict_values_are_np_arrays_of_ints(self):
        zero_array = np.array([1, 2])
        one_array = np.array([9, 10])
        map_proto = numpy_helper.from_dict({0: zero_array, 1: one_array})
        self.assertIsInstance(map_proto, onnx.MapProto)

        out_tensor = numpy_helper.to_array(map_proto.values.tensor_values[0])
        self.assertEqual(out_tensor[0], zero_array[0])
        self.assertEqual(out_tensor[1], zero_array[1])

        out_tensor = numpy_helper.to_array(map_proto.values.tensor_values[1])
        self.assertEqual(out_tensor[0], one_array[0])
        self.assertEqual(out_tensor[1], one_array[1])

    def test_from_dict_differing_key_types(self):
        with self.assertRaises(TypeError):
            # Differing key types should raise a TypeError
            numpy_helper.from_dict({0: np.array(0.1), 1.1: np.array(0.9)})

    def test_from_dict_differing_value_types(self):
        with self.assertRaises(TypeError):
            # Differing value types should raise a TypeError
            numpy_helper.from_dict({0: np.array(1), 1: np.array(0.9)})

    def _to_array_from_array(self, value: int, check_dtype: bool = True):
        onnx_model = helper.make_model(
            helper.make_graph(
                [helper.make_node("Cast", ["X"], ["Y"], to=value)],
                "test",
                [helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [4])],
                [helper.make_tensor_value_info("Y", value, [4])],
            )
        )
        ref = onnx.reference.ReferenceEvaluator(onnx_model)
        start = ref.run(None, {"X": np.array([0, 1, -2, 3], dtype=np.float32)})
        tp = numpy_helper.from_array(start[0], name="check")
        self.assertEqual(tp.data_type, value)
        back = numpy_helper.to_array(tp)
        self.assertEqual(start[0].shape, back.shape)
        if check_dtype:
            self.assertEqual(start[0].dtype, back.dtype)
        again = numpy_helper.from_array(back, name="check")
        self.assertEqual(tp.data_type, again.data_type)
        self.assertEqual(tp.name, again.name)
        self.assertEqual(len(tp.raw_data), len(again.raw_data))
        self.assertEqual(list(tp.raw_data), list(again.raw_data))
        self.assertEqual(tp.raw_data, again.raw_data)
        self.assertEqual(tuple(tp.dims), tuple(again.dims))
        self.assertEqual(tp.SerializeToString(), again.SerializeToString())
        self.assertEqual(tp.data_type, helper.np_dtype_to_tensor_dtype(back.dtype))

    @parameterized.parameterized.expand(
        [
            ("FLOAT", onnx.TensorProto.FLOAT),
            ("UINT8", onnx.TensorProto.UINT8),
            ("INT8", onnx.TensorProto.INT8),
            ("UINT16", onnx.TensorProto.UINT16),
            ("INT16", onnx.TensorProto.INT16),
            ("INT32", onnx.TensorProto.INT32),
            ("INT64", onnx.TensorProto.INT64),
            ("BOOL", onnx.TensorProto.BOOL),
            ("FLOAT16", onnx.TensorProto.FLOAT16),
            ("DOUBLE", onnx.TensorProto.DOUBLE),
            ("UINT32", onnx.TensorProto.UINT32),
            ("UINT64", onnx.TensorProto.UINT64),
            ("COMPLEX64", onnx.TensorProto.COMPLEX64),
            ("COMPLEX128", onnx.TensorProto.COMPLEX128),
            ("BFLOAT16", onnx.TensorProto.BFLOAT16),
            ("FLOAT8E4M3FN", onnx.TensorProto.FLOAT8E4M3FN),
            ("FLOAT8E4M3FNUZ", onnx.TensorProto.FLOAT8E4M3FNUZ),
            ("FLOAT8E5M2", onnx.TensorProto.FLOAT8E5M2),
            ("FLOAT8E5M2FNUZ", onnx.TensorProto.FLOAT8E5M2FNUZ),
            ("FLOAT8E8M0", onnx.TensorProto.FLOAT8E8M0),
            ("UINT4", onnx.TensorProto.UINT4),
            ("INT4", onnx.TensorProto.INT4),
            ("FLOAT4E2M1", onnx.TensorProto.FLOAT4E2M1),
        ]
    )
    def test_to_array_from_array(self, _: str, data_type: onnx.TensorProto.DataType):
        self._to_array_from_array(data_type)

    def test_to_array_from_array_string(self):
        self._to_array_from_array(onnx.TensorProto.STRING, False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
