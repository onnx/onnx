import unittest

import numpy as np  # type: ignore
from numpy.testing import assert_almost_equal  # type: ignore

from onnx import TensorProto
from onnx.helper import deprecated_make_tensor, make_tensor
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
from onnx.numpy_helper import from_array, to_array


class TestBasicFunctions(unittest.TestCase):
    def test_numeric_types(self):  # type: ignore
        dtypes = [
            np.float16,
            np.float32,
            np.float64,
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
                assert_almost_equal(t, u)

    def test_deprecated_make_tensor(self):  # type: ignore
        for pt, dt in TENSOR_TYPE_TO_NP_TYPE.items():
            if pt == TensorProto.BFLOAT16:
                continue
            with self.subTest(dt=dt, pt=pt, raw=False):
                if pt == TensorProto.STRING:
                    t = np.array(
                        [[b"i0", b"i1", b"i2"], [b"i6", b"i7", b"i8"]], dtype=dt
                    )
                else:
                    t = np.array([[0, 1, 2], [6, 7, 8]], dtype=dt)
                ot = deprecated_make_tensor("test", pt, t.shape, t, raw=False)
                self.assertFalse(ot is None)
                if pt != TensorProto.STRING:
                    u = to_array(ot)
                    self.assertEqual(t.dtype, u.dtype)
                    self.assertEqual(t.tolist(), u.tolist())
            with self.subTest(dt=dt, pt=pt, raw=True):
                if pt != TensorProto.STRING:
                    t = np.array([[0, 1, 2], [6, 7, 8]], dtype=dt)
                    ot = deprecated_make_tensor(
                        "test", pt, t.shape, t.tobytes(), raw=True
                    )
                    self.assertFalse(ot is None)
                    u = to_array(ot)
                    self.assertEqual(t.dtype, u.dtype)
                    assert_almost_equal(t, u)

    def test_make_tensor(self):  # type: ignore
        for pt, dt in TENSOR_TYPE_TO_NP_TYPE.items():
            if pt == TensorProto.BFLOAT16:
                continue
            with self.subTest(dt=dt, pt=pt, raw=False):
                if pt == TensorProto.STRING:
                    t = np.array([["i0", "i1", "i2"], ["i6", "i7", "i8"]], dtype=dt)
                else:
                    t = np.array([[0, 1, 2], [6, 7, 8]], dtype=dt)
                ot = make_tensor("test", pt, t.shape, t, raw=False)
                self.assertFalse(ot is None)
                u = to_array(ot)
                self.assertEqual(t.dtype, u.dtype)
                self.assertEqual(t.tolist(), u.tolist())
            with self.subTest(dt=dt, pt=pt, raw=True):
                t = np.array([[0, 1, 2], [6, 7, 8]], dtype=dt)
                if pt == TensorProto.STRING:
                    with self.assertRaises(TypeError):
                        make_tensor("test", pt, t.shape, t.tobytes(), raw=True)
                else:
                    ot = make_tensor("test", pt, t.shape, t.tobytes(), raw=True)
                    self.assertFalse(ot is None)
                    u = to_array(ot)
                    self.assertEqual(t.dtype, u.dtype)
                    assert_almost_equal(t, u)


if __name__ == "__main__":
    unittest.main()
