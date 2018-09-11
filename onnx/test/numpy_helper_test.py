from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

from onnx import numpy_helper

import unittest


class TestNumpyHelper(unittest.TestCase):

    def _test_numpy_helper_float_type(self, dtype):  # type: (np.number) -> None
        a = np.random.rand(13, 37).astype(dtype)
        tensor_def = numpy_helper.from_array(a, "test")
        self.assertEqual(tensor_def.name, "test")
        a_recover = numpy_helper.to_array(tensor_def)
        np.testing.assert_equal(a, a_recover)

    def _test_numpy_helper_int_type(self, dtype):  # type: (np.number) -> None
        a = np.random.randint(
            np.iinfo(dtype).min,
            np.iinfo(dtype).max,
            dtype=dtype,
            size=(13, 37))
        tensor_def = numpy_helper.from_array(a, "test")
        self.assertEqual(tensor_def.name, "test")
        a_recover = numpy_helper.to_array(tensor_def)
        np.testing.assert_equal(a, a_recover)

    def test_float(self):  # type: () -> None
        self._test_numpy_helper_float_type(np.float32)

    def test_uint8(self):  # type: () -> None
        self._test_numpy_helper_int_type(np.uint8)

    def test_int8(self):  # type: () -> None
        self._test_numpy_helper_int_type(np.int8)

    def test_uint16(self):  # type: () -> None
        self._test_numpy_helper_int_type(np.uint16)

    def test_int16(self):  # type: () -> None
        self._test_numpy_helper_int_type(np.int16)

    def test_int32(self):  # type: () -> None
        self._test_numpy_helper_int_type(np.int32)

    def test_int64(self):  # type: () -> None
        self._test_numpy_helper_int_type(np.int64)

    def test_string(self):  # type: () -> None
        pass

    def test_bool(self):  # type: () -> None
        a = np.random.randint(2, size=(13, 37)).astype(np.bool)
        tensor_def = numpy_helper.from_array(a, "test")
        self.assertEqual(tensor_def.name, "test")
        a_recover = numpy_helper.to_array(tensor_def)
        np.testing.assert_equal(a, a_recover)

    def test_float16(self):  # type: () -> None
        self._test_numpy_helper_float_type(np.float32)

    def test_complex64(self):  # type: () -> None
        self._test_numpy_helper_float_type(np.complex64)

    def test_complex128(self):  # type: () -> None
        self._test_numpy_helper_float_type(np.complex128)


if __name__ == '__main__':
    unittest.main()
