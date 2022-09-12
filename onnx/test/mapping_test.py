import unittest

import numpy as np  # type: ignore
from numpy.testing import assert_almost_equal  # type: ignore
from paddle_bfloat import bfloat16  # type: ignore

from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
from onnx.numpy_helper import from_array, to_array


class TestBasicFunctions(unittest.TestCase):
    def test_numeric_types(self):  # type: ignore
        dtypes = [
            np.float16,
            np.float32,
            np.float64,
            bfloat16,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
            np.complex64,
            np.complex128,
        ]
        for dt in dtypes:
            with self.subTest(dtype=dt):
                t = np.array([0, 1, 2], dtype=dt)
                ot = from_array(t)
                u = to_array(ot)
                self.assertEqual(t.dtype, u.dtype)
                if t.dtype == bfloat16:
                    # assert_almost_equal does not work is this case
                    assert_almost_equal(t.astype(np.float32), u.astype(np.float32))
                else:
                    assert_almost_equal(t, u)


if __name__ == "__main__":
    unittest.main()
