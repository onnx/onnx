# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0

import unittest
from typing import Any

import numpy as np
import parameterized

from onnx import helper, numpy_helper


def bfloat16_to_float32(ival: int) -> Any:
    if ival == 0x7FC0:
        return np.float32(np.nan)

    expo = ival >> 7
    prec = ival - (expo << 7)
    sign = expo & 256
    powe = expo & 255
    fval = float(prec * 2 ** (-7) + 1) * 2.0 ** (powe - 127)
    if sign:
        fval = -fval
    return np.float32(fval)


def float8e4m3_to_float32(ival: int) -> Any:
    if ival < 0 or ival > 255:
        raise ValueError(f"{ival} is not a float8.")
    if ival == 255:
        return np.float32(-np.nan)
    if ival == 127:
        return np.float32(np.nan)
    if (ival & 0x7F) == 0:
        return np.float32(0)

    sign = ival & 0x80
    ival &= 0x7F
    expo = ival >> 3
    mant = ival & 0x07
    powe = expo & 0x0F
    if expo == 0:
        powe -= 6
        fraction = 0
    else:
        powe -= 7
        fraction = 1
    fval = float(mant / 8 + fraction) * 2.0**powe
    if sign:
        fval = -fval
    return np.float32(fval)


def float8e5m2_to_float32(ival: int) -> Any:
    if ival < 0 or ival > 255:
        raise ValueError(f"{ival} is not a float8.")
    if ival in (255, 254, 253):
        return np.float32(-np.nan)
    if ival in (127, 126, 125):
        return np.float32(np.nan)
    if ival == 252:
        return -np.float32(np.inf)
    if ival == 124:
        return np.float32(np.inf)
    if (ival & 0x7F) == 0:
        return np.float32(0)

    sign = ival & 0x80
    ival &= 0x7F
    expo = ival >> 2
    mant = ival & 0x03
    powe = expo & 0x1F
    if expo == 0:
        powe -= 14
        fraction = 0
    else:
        powe -= 15
        fraction = 1
    fval = float(mant / 4 + fraction) * 2.0**powe
    if sign:
        fval = -fval
    return np.float32(fval)


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

    @parameterized.parameterized.expand(
        [
            (1,),
            (0.100097656,),
            (130048,),
            (1.2993813e-5,),
            (np.nan,),
            (np.inf,),
        ]
    )
    def test_bfloat16_to_float32(self, f):
        f32 = np.float32(f)
        bf16 = helper.float32_to_bfloat16(f32)
        assert isinstance(bf16, int)
        f32_1 = numpy_helper.bfloat16_to_float32(np.array([bf16]))[0]
        f32_2 = bfloat16_to_float32(bf16)
        if np.isnan(f32):
            assert np.isnan(f32_1)
            assert np.isnan(f32_2)
        else:
            self.assertEqual(f32, f32_1)
            self.assertEqual(f32, f32_2)

    def test_float8e4m3_to_float32(self):
        self.assertEqual(numpy_helper.float8e4m3_to_float32(int("1111110", 2)), 448)
        self.assertEqual(numpy_helper.float8e4m3_to_float32(int("1000", 2)), 2 ** (-6))
        self.assertEqual(numpy_helper.float8e4m3_to_float32(int("1", 2)), 2 ** (-9))
        self.assertEqual(
            numpy_helper.float8e4m3_to_float32(int("111", 2)), 0.875 * 2 ** (-6)
        )
        for f in [
            0,
            1,
            -1,
            0.5,
            -0.5,
            0.1015625,
            -0.1015625,
            2,
            3,
            -2,
            -3,
            448,
            2 ** (-6),
            2 ** (-9),
            0.875 * 2 ** (-6),
            np.nan,
        ]:
            with self.subTest(f=f):
                f32 = np.float32(f)
                f8 = helper.float32_to_float8e4m3(f32)
                assert isinstance(f8, int)
                f32_1 = numpy_helper.float8e4m3_to_float32(np.array([f8]))[0]
                f32_2 = float8e4m3_to_float32(f8)
                if np.isnan(f32):
                    assert np.isnan(f32_1)
                    assert np.isnan(f32_2)
                else:
                    self.assertEqual(f32, f32_1)
                    self.assertEqual(f32, f32_2)

    def test_float8e5m2_to_float32(self):
        self.assertEqual(numpy_helper.float8e5m2_to_float32(int("1111011", 2)), 57344)
        self.assertEqual(numpy_helper.float8e5m2_to_float32(int("100", 2)), 2 ** (-14))
        self.assertEqual(
            numpy_helper.float8e5m2_to_float32(int("11", 2)), 0.75 * 2 ** (-14)
        )
        self.assertEqual(numpy_helper.float8e5m2_to_float32(int("1", 2)), 2 ** (-16))
        self.assertTrue(np.isnan(numpy_helper.float8e5m2_to_float32(int("1111101", 2))))
        self.assertTrue(np.isnan(numpy_helper.float8e5m2_to_float32(int("1111110", 2))))
        self.assertTrue(np.isnan(numpy_helper.float8e5m2_to_float32(int("1111111", 2))))
        self.assertTrue(
            np.isnan(numpy_helper.float8e5m2_to_float32(int("11111101", 2)))
        )
        self.assertTrue(
            np.isnan(numpy_helper.float8e5m2_to_float32(int("11111110", 2)))
        )
        self.assertTrue(
            np.isnan(numpy_helper.float8e5m2_to_float32(int("11111111", 2)))
        )
        self.assertEqual(numpy_helper.float8e5m2_to_float32(int("1111100", 2)), np.inf)
        self.assertEqual(
            numpy_helper.float8e5m2_to_float32(int("11111100", 2)), -np.inf
        )
        for f in [
            0,
            0.0017089844,
            20480,
            14,
            -3584,
            np.nan,
        ]:
            with self.subTest(f=f):
                f32 = np.float32(f)
                f8 = helper.float32_to_float8e5m2(f32)
                assert isinstance(f8, int)
                f32_1 = numpy_helper.float8e5m2_to_float32(np.array([f8]))[0]
                f32_2 = float8e5m2_to_float32(f8)
                if np.isnan(f32):
                    assert np.isnan(f32_1)
                    assert np.isnan(f32_2)
                else:
                    self.assertEqual(f32, f32_1)
                    self.assertEqual(f32, f32_2)

    def test_float8_e4m3fn_inf(self):
        x = np.float32(np.inf)
        to = helper.float32_to_float8e4m3(x)
        back = numpy_helper.float8e4m3_to_float32(to)
        self.assertEqual(back, 448)

        x = np.float32(np.inf)
        to = helper.float32_to_float8e4m3(x, saturate=False)
        back = numpy_helper.float8e4m3_to_float32(to)
        self.assertTrue(np.isnan(back))

        x = np.float32(-np.inf)
        to = helper.float32_to_float8e4m3(x)
        self.assertEqual(to & 0x80, 0x80)
        back = numpy_helper.float8e4m3_to_float32(to)
        self.assertEqual(back, -448)

        x = np.float32(-np.inf)
        to = helper.float32_to_float8e4m3(x, saturate=False)
        self.assertEqual(to & 0x80, 0x80)
        back = numpy_helper.float8e4m3_to_float32(to)
        self.assertTrue(np.isnan(back))

    def test_float8_e4m3fnuz_inf(self):
        x = np.float32(np.inf)
        to = helper.float32_to_float8e4m3(x, uz=True)
        back = numpy_helper.float8e4m3_to_float32(to, uz=True)
        self.assertEqual(back, 240)

        x = np.float32(np.inf)
        to = helper.float32_to_float8e4m3(x, uz=True, saturate=False)
        back = numpy_helper.float8e4m3_to_float32(to, uz=True)
        self.assertTrue(np.isnan(back))

        x = np.float32(-np.inf)
        to = helper.float32_to_float8e4m3(x, uz=True)
        back = numpy_helper.float8e4m3_to_float32(to, uz=True)
        self.assertEqual(back, -240)

        x = np.float32(-np.inf)
        to = helper.float32_to_float8e4m3(x, uz=True, saturate=False)
        back = numpy_helper.float8e4m3_to_float32(to, uz=True)
        self.assertTrue(np.isnan(back))

    def test_float8_e5m2_inf(self):
        x = np.float32(np.inf)
        to = helper.float32_to_float8e5m2(x)
        back = numpy_helper.float8e5m2_to_float32(to)
        self.assertEqual(back, 57344)

        x = np.float32(np.inf)
        to = helper.float32_to_float8e5m2(x, saturate=False)
        back = numpy_helper.float8e5m2_to_float32(to)
        self.assertTrue(np.isinf(back))

        x = np.float32(-np.inf)
        to = helper.float32_to_float8e5m2(x)
        self.assertEqual(to & 0x80, 0x80)
        back = numpy_helper.float8e5m2_to_float32(to)
        self.assertEqual(back, -57344)

        x = np.float32(-np.inf)
        to = helper.float32_to_float8e5m2(x, saturate=False)
        self.assertEqual(to & 0x80, 0x80)
        back = numpy_helper.float8e5m2_to_float32(to)
        self.assertTrue(np.isinf(back))
        self.assertLess(back, 0)

    def test_float8_e5m2fnuz_inf(self):
        x = np.float32(np.inf)
        to = helper.float32_to_float8e5m2(x, fn=True, uz=True)
        back = numpy_helper.float8e5m2_to_float32(to, fn=True, uz=True)
        self.assertEqual(back, 57344)

        x = np.float32(np.inf)
        to = helper.float32_to_float8e5m2(x, fn=True, uz=True, saturate=False)
        back = numpy_helper.float8e5m2_to_float32(to, fn=True, uz=True)
        self.assertTrue(np.isnan(back))

        x = np.float32(-np.inf)
        to = helper.float32_to_float8e5m2(x, fn=True, uz=True)
        back = numpy_helper.float8e5m2_to_float32(to, fn=True, uz=True)
        self.assertEqual(back, -57344)

        x = np.float32(-np.inf)
        to = helper.float32_to_float8e5m2(x, fn=True, uz=True, saturate=False)
        back = numpy_helper.float8e5m2_to_float32(to, fn=True, uz=True)
        self.assertTrue(np.isnan(back))

    def test_float8_e4m3fn_out_of_range(self):
        x = np.float32(1000000)
        to = helper.float32_to_float8e4m3(x)
        back = numpy_helper.float8e4m3_to_float32(to)
        self.assertEqual(back, 448)

        x = np.float32(1000000)
        to = helper.float32_to_float8e4m3(x, saturate=False)
        back = numpy_helper.float8e4m3_to_float32(to)
        self.assertTrue(np.isnan(back))

        x = np.float32(-1000000)
        to = helper.float32_to_float8e4m3(x)
        back = numpy_helper.float8e4m3_to_float32(to)
        self.assertEqual(back, -448)

        x = np.float32(-1000000)
        to = helper.float32_to_float8e4m3(x, saturate=False)
        back = numpy_helper.float8e4m3_to_float32(to)
        self.assertTrue(np.isnan(back))

    def test_float8_e4m3fnuz_out_of_range(self):
        x = np.float32(1000000)
        to = helper.float32_to_float8e4m3(x, uz=True)
        back = numpy_helper.float8e4m3_to_float32(to, uz=True)
        self.assertEqual(back, 240)

        x = np.float32(1000000)
        to = helper.float32_to_float8e4m3(x, uz=True, saturate=False)
        back = numpy_helper.float8e4m3_to_float32(to, uz=True)
        self.assertTrue(np.isnan(back))

        x = np.float32(-1000000)
        to = helper.float32_to_float8e4m3(x, uz=True)
        back = numpy_helper.float8e4m3_to_float32(to, uz=True)
        self.assertEqual(back, -240)

        x = np.float32(-1000000)
        to = helper.float32_to_float8e4m3(x, uz=True, saturate=False)
        back = numpy_helper.float8e4m3_to_float32(to, uz=True)
        self.assertTrue(np.isnan(back))

    def test_float8_e5m2_out_of_range(self):
        x = np.float32(1000000)
        to = helper.float32_to_float8e5m2(x)
        back = numpy_helper.float8e5m2_to_float32(to)
        self.assertEqual(back, 57344)

        x = np.float32(1000000)
        to = helper.float32_to_float8e5m2(x, saturate=False)
        back = numpy_helper.float8e5m2_to_float32(to)
        self.assertTrue(np.isinf(back))

        x = np.float32(-1000000)
        to = helper.float32_to_float8e5m2(x)
        back = numpy_helper.float8e5m2_to_float32(to)
        self.assertEqual(back, -57344)

        x = np.float32(-1000000)
        to = helper.float32_to_float8e5m2(x, saturate=False)
        back = numpy_helper.float8e5m2_to_float32(to)
        self.assertTrue(np.isinf(back))

    def test_float8_e5m2fnuz_out_of_range(self):
        x = np.float32(1000000)
        to = helper.float32_to_float8e5m2(x, fn=True, uz=True)
        back = numpy_helper.float8e5m2_to_float32(to, fn=True, uz=True)
        self.assertEqual(back, 57344)

        x = np.float32(1000000)
        to = helper.float32_to_float8e5m2(x, fn=True, uz=True, saturate=False)
        back = numpy_helper.float8e5m2_to_float32(to, fn=True, uz=True)
        self.assertTrue(np.isnan(back))

        x = np.float32(-1000000)
        to = helper.float32_to_float8e5m2(x, fn=True, uz=True)
        back = numpy_helper.float8e5m2_to_float32(to, fn=True, uz=True)
        self.assertEqual(back, -57344)

        x = np.float32(-1000000)
        to = helper.float32_to_float8e5m2(x, fn=True, uz=True, saturate=False)
        back = numpy_helper.float8e5m2_to_float32(to, fn=True, uz=True)
        self.assertTrue(np.isnan(back))

    def test_float8_e4m3fn_negative_zero(self):
        x = numpy_helper.float8e5m2_to_float32(0x80)  # -0
        to = helper.float32_to_float8e4m3(x)
        self.assertEqual(to, 0x80)
        back = numpy_helper.float8e4m3_to_float32(to)
        self.assertEqual(back, 0)

        x = numpy_helper.float8e5m2_to_float32(0x80)  # -0
        to = helper.float32_to_float8e4m3(x, saturate=False)
        self.assertEqual(to, 0x80)
        back = numpy_helper.float8e4m3_to_float32(to)
        self.assertEqual(back, 0)

    def test_float8_e4m3fnuz_negative_zero(self):
        x = numpy_helper.float8e5m2_to_float32(0x80)  # -0
        to = helper.float32_to_float8e4m3(x, uz=True)
        self.assertEqual(to, 0)
        back = numpy_helper.float8e4m3_to_float32(to, uz=True)
        self.assertEqual(back, 0)

        x = numpy_helper.float8e5m2_to_float32(0x80)  # -0
        to = helper.float32_to_float8e4m3(x, uz=True, saturate=False)
        back = numpy_helper.float8e4m3_to_float32(to, uz=True)
        self.assertEqual(back, 0)
        self.assertEqual(to, 0)

    def test_float8_e5m2_negative_zero(self):
        x = numpy_helper.float8e5m2_to_float32(0x80)  # -0
        to = helper.float32_to_float8e5m2(x)
        self.assertEqual(to, 0x80)
        back = numpy_helper.float8e4m3_to_float32(to)
        self.assertEqual(back, 0)

        x = numpy_helper.float8e5m2_to_float32(0x80)  # -0
        to = helper.float32_to_float8e5m2(x, saturate=False)
        self.assertEqual(to, 0x80)
        back = numpy_helper.float8e4m3_to_float32(to)
        self.assertEqual(back, 0)

    def test_float8_e5m2fnuz_negative_zero(self):
        x = numpy_helper.float8e5m2_to_float32(0x80)  # -0
        to = helper.float32_to_float8e5m2(x, fn=True, uz=True)
        self.assertEqual(to, 0)
        back = numpy_helper.float8e4m3_to_float32(to, fn=True, uz=True)
        self.assertEqual(back, 0)

        x = numpy_helper.float8e5m2_to_float32(0x80)  # -0
        to = helper.float32_to_float8e5m2(x, fn=True, uz=True, saturate=False)
        self.assertEqual(to, 0)
        back = numpy_helper.float8e4m3_to_float32(to, fn=True, uz=True)
        self.assertEqual(back, 0)

    def test_float8_e4m3fn_negative_nan(self):
        x = numpy_helper.float8e5m2_to_float32(255)  # -nan
        to = helper.float32_to_float8e4m3(x)
        self.assertEqual(to, 255)
        back = numpy_helper.float8e4m3_to_float32(to)
        self.assertTrue(np.isnan(back))

        x = numpy_helper.float8e5m2_to_float32(255)  # -nan
        to = helper.float32_to_float8e4m3(x, saturate=False)
        self.assertEqual(to, 255)
        back = numpy_helper.float8e4m3_to_float32(to)
        self.assertTrue(np.isnan(back))

    def test_float8_e4m3fnuz_negative_nan(self):
        x = numpy_helper.float8e5m2_to_float32(255)  # -nan
        to = helper.float32_to_float8e4m3(x, uz=True)
        self.assertEqual(to, 0x80)
        back = numpy_helper.float8e4m3_to_float32(to, uz=True)
        self.assertTrue(np.isnan(back))

        x = numpy_helper.float8e5m2_to_float32(255)  # -nan
        to = helper.float32_to_float8e4m3(x, uz=True, saturate=False)
        self.assertEqual(to, 0x80)
        back = numpy_helper.float8e4m3_to_float32(to, uz=True)
        self.assertTrue(np.isnan(back))

    def test_float8_e5m2_negative_nan(self):
        x = numpy_helper.float8e5m2_to_float32(255)  # -nan
        to = helper.float32_to_float8e5m2(x)
        self.assertEqual(to, 255)
        back = numpy_helper.float8e4m3_to_float32(to)
        self.assertTrue(np.isnan(back))

        x = numpy_helper.float8e5m2_to_float32(255)  # -nan
        to = helper.float32_to_float8e5m2(x, saturate=False)
        self.assertEqual(to, 255)
        back = numpy_helper.float8e4m3_to_float32(to)
        self.assertTrue(np.isnan(back))

    def test_float8_e5m2fnuz_negative_nan(self):
        x = numpy_helper.float8e5m2_to_float32(255)  # -nan
        to = helper.float32_to_float8e5m2(x, fn=True, uz=True)
        self.assertEqual(to, 0x80)
        back = numpy_helper.float8e4m3_to_float32(to, fn=True, uz=True)
        self.assertTrue(np.isnan(back))

        x = numpy_helper.float8e5m2_to_float32(255)  # -nan
        to = helper.float32_to_float8e5m2(x, fn=True, uz=True, saturate=False)
        self.assertEqual(to, 0x80)
        back = numpy_helper.float8e4m3_to_float32(to, fn=True, uz=True)
        self.assertTrue(np.isnan(back))


if __name__ == "__main__":
    unittest.main()
