from __future__ import annotations

import pytest

import env  # noqa: F401
from pybind11_tests import numpy_array as m

np = pytest.importorskip("numpy")


def test_dtypes():
    # See issue #1328.
    # - Platform-dependent sizes.
    for size_check in m.get_platform_dtype_size_checks():
        print(size_check)
        assert size_check.size_cpp == size_check.size_numpy, size_check
    # - Concrete sizes.
    for check in m.get_concrete_dtype_checks():
        print(check)
        assert check.numpy == check.pybind11, check
        if check.numpy.num != check.pybind11.num:
            print(
                f"NOTE: typenum mismatch for {check}: {check.numpy.num} != {check.pybind11.num}"
            )


@pytest.fixture
def arr():
    return np.array([[1, 2, 3], [4, 5, 6]], "=u2")


def test_array_attributes():
    a = np.array(0, "f8")
    assert m.ndim(a) == 0
    assert all(m.shape(a) == [])
    assert all(m.strides(a) == [])
    with pytest.raises(IndexError) as excinfo:
        m.shape(a, 0)
    assert str(excinfo.value) == "invalid axis: 0 (ndim = 0)"
    with pytest.raises(IndexError) as excinfo:
        m.strides(a, 0)
    assert str(excinfo.value) == "invalid axis: 0 (ndim = 0)"
    assert m.writeable(a)
    assert m.size(a) == 1
    assert m.itemsize(a) == 8
    assert m.nbytes(a) == 8
    assert m.owndata(a)

    a = np.array([[1, 2, 3], [4, 5, 6]], "u2").view()
    a.flags.writeable = False
    assert m.ndim(a) == 2
    assert all(m.shape(a) == [2, 3])
    assert m.shape(a, 0) == 2
    assert m.shape(a, 1) == 3
    assert all(m.strides(a) == [6, 2])
    assert m.strides(a, 0) == 6
    assert m.strides(a, 1) == 2
    with pytest.raises(IndexError) as excinfo:
        m.shape(a, 2)
    assert str(excinfo.value) == "invalid axis: 2 (ndim = 2)"
    with pytest.raises(IndexError) as excinfo:
        m.strides(a, 2)
    assert str(excinfo.value) == "invalid axis: 2 (ndim = 2)"
    assert not m.writeable(a)
    assert m.size(a) == 6
    assert m.itemsize(a) == 2
    assert m.nbytes(a) == 12
    assert not m.owndata(a)


@pytest.mark.parametrize(
    ("args", "ret"), [([], 0), ([0], 0), ([1], 3), ([0, 1], 1), ([1, 2], 5)]
)
def test_index_offset(arr, args, ret):
    assert m.index_at(arr, *args) == ret
    assert m.index_at_t(arr, *args) == ret
    assert m.offset_at(arr, *args) == ret * arr.dtype.itemsize
    assert m.offset_at_t(arr, *args) == ret * arr.dtype.itemsize


def test_dim_check_fail(arr):
    for func in (
        m.index_at,
        m.index_at_t,
        m.offset_at,
        m.offset_at_t,
        m.data,
        m.data_t,
        m.mutate_data,
        m.mutate_data_t,
    ):
        with pytest.raises(IndexError) as excinfo:
            func(arr, 1, 2, 3)
        assert str(excinfo.value) == "too many indices for an array: 3 (ndim = 2)"


@pytest.mark.parametrize(
    ("args", "ret"),
    [
        ([], [1, 2, 3, 4, 5, 6]),
        ([1], [4, 5, 6]),
        ([0, 1], [2, 3, 4, 5, 6]),
        ([1, 2], [6]),
    ],
)
def test_data(arr, args, ret):
    from sys import byteorder

    assert all(m.data_t(arr, *args) == ret)
    assert all(m.data(arr, *args)[(0 if byteorder == "little" else 1) :: 2] == ret)
    assert all(m.data(arr, *args)[(1 if byteorder == "little" else 0) :: 2] == 0)


@pytest.mark.parametrize("dim", [0, 1, 3])
def test_at_fail(arr, dim):
    for func in m.at_t, m.mutate_at_t:
        with pytest.raises(IndexError) as excinfo:
            func(arr, *([0] * dim))
        assert str(excinfo.value) == f"index dimension mismatch: {dim} (ndim = 2)"


def test_at(arr):
    assert m.at_t(arr, 0, 2) == 3
    assert m.at_t(arr, 1, 0) == 4

    assert all(m.mutate_at_t(arr, 0, 2).ravel() == [1, 2, 4, 4, 5, 6])
    assert all(m.mutate_at_t(arr, 1, 0).ravel() == [1, 2, 4, 5, 5, 6])


def test_mutate_readonly(arr):
    arr.flags.writeable = False
    for func, args in (
        (m.mutate_data, ()),
        (m.mutate_data_t, ()),
        (m.mutate_at_t, (0, 0)),
    ):
        with pytest.raises(ValueError) as excinfo:
            func(arr, *args)
        assert str(excinfo.value) == "array is not writeable"


def test_mutate_data(arr):
    assert all(m.mutate_data(arr).ravel() == [2, 4, 6, 8, 10, 12])
    assert all(m.mutate_data(arr).ravel() == [4, 8, 12, 16, 20, 24])
    assert all(m.mutate_data(arr, 1).ravel() == [4, 8, 12, 32, 40, 48])
    assert all(m.mutate_data(arr, 0, 1).ravel() == [4, 16, 24, 64, 80, 96])
    assert all(m.mutate_data(arr, 1, 2).ravel() == [4, 16, 24, 64, 80, 192])

    assert all(m.mutate_data_t(arr).ravel() == [5, 17, 25, 65, 81, 193])
    assert all(m.mutate_data_t(arr).ravel() == [6, 18, 26, 66, 82, 194])
    assert all(m.mutate_data_t(arr, 1).ravel() == [6, 18, 26, 67, 83, 195])
    assert all(m.mutate_data_t(arr, 0, 1).ravel() == [6, 19, 27, 68, 84, 196])
    assert all(m.mutate_data_t(arr, 1, 2).ravel() == [6, 19, 27, 68, 84, 197])


def test_bounds_check(arr):
    for func in (
        m.index_at,
        m.index_at_t,
        m.data,
        m.data_t,
        m.mutate_data,
        m.mutate_data_t,
        m.at_t,
        m.mutate_at_t,
    ):
        with pytest.raises(IndexError) as excinfo:
            func(arr, 2, 0)
        assert str(excinfo.value) == "index 2 is out of bounds for axis 0 with size 2"
        with pytest.raises(IndexError) as excinfo:
            func(arr, 0, 4)
        assert str(excinfo.value) == "index 4 is out of bounds for axis 1 with size 3"


def test_make_c_f_array():
    assert m.make_c_array().flags.c_contiguous
    assert not m.make_c_array().flags.f_contiguous
    assert m.make_f_array().flags.f_contiguous
    assert not m.make_f_array().flags.c_contiguous


def test_make_empty_shaped_array():
    m.make_empty_shaped_array()

    # empty shape means numpy scalar, PEP 3118
    assert m.scalar_int().ndim == 0
    assert m.scalar_int().shape == ()
    assert m.scalar_int() == 42


def test_wrap():
    def assert_references(a, b, base=None):
        if base is None:
            base = a
        assert a is not b
        assert a.__array_interface__["data"][0] == b.__array_interface__["data"][0]
        assert a.shape == b.shape
        assert a.strides == b.strides
        assert a.flags.c_contiguous == b.flags.c_contiguous
        assert a.flags.f_contiguous == b.flags.f_contiguous
        assert a.flags.writeable == b.flags.writeable
        assert a.flags.aligned == b.flags.aligned
        assert a.flags.writebackifcopy == b.flags.writebackifcopy
        assert np.all(a == b)
        assert not b.flags.owndata
        assert b.base is base
        if a.flags.writeable and a.ndim == 2:
            a[0, 0] = 1234
            assert b[0, 0] == 1234

    a1 = np.array([1, 2], dtype=np.int16)
    assert a1.flags.owndata
    assert a1.base is None
    a2 = m.wrap(a1)
    assert_references(a1, a2)

    a1 = np.array([[1, 2], [3, 4]], dtype=np.float32, order="F")
    assert a1.flags.owndata
    assert a1.base is None
    a2 = m.wrap(a1)
    assert_references(a1, a2)

    a1 = np.array([[1, 2], [3, 4]], dtype=np.float32, order="C")
    a1.flags.writeable = False
    a2 = m.wrap(a1)
    assert_references(a1, a2)

    a1 = np.random.random((4, 4, 4))
    a2 = m.wrap(a1)
    assert_references(a1, a2)

    a1t = a1.transpose()
    a2 = m.wrap(a1t)
    assert_references(a1t, a2, a1)

    a1d = a1.diagonal()
    a2 = m.wrap(a1d)
    assert_references(a1d, a2, a1)

    a1m = a1[::-1, ::-1, ::-1]
    a2 = m.wrap(a1m)
    assert_references(a1m, a2, a1)


def test_numpy_view(capture):
    with capture:
        ac = m.ArrayClass()
        ac_view_1 = ac.numpy_view()
        ac_view_2 = ac.numpy_view()
        assert np.all(ac_view_1 == np.array([1, 2], dtype=np.int32))
        del ac
        pytest.gc_collect()
    assert (
        capture
        == """
        ArrayClass()
        ArrayClass::numpy_view()
        ArrayClass::numpy_view()
    """
    )
    ac_view_1[0] = 4
    ac_view_1[1] = 3
    assert ac_view_2[0] == 4
    assert ac_view_2[1] == 3
    with capture:
        del ac_view_1
        del ac_view_2
        pytest.gc_collect()
        pytest.gc_collect()
    assert (
        capture
        == """
        ~ArrayClass()
    """
    )


def test_cast_numpy_int64_to_uint64():
    m.function_taking_uint64(123)
    m.function_taking_uint64(np.uint64(123))


def test_isinstance():
    assert m.isinstance_untyped(np.array([1, 2, 3]), "not an array")
    assert m.isinstance_typed(np.array([1.0, 2.0, 3.0]))


def test_constructors():
    defaults = m.default_constructors()
    for a in defaults.values():
        assert a.size == 0
    assert defaults["array"].dtype == np.array([]).dtype
    assert defaults["array_t<int32>"].dtype == np.int32
    assert defaults["array_t<double>"].dtype == np.float64

    results = m.converting_constructors([1, 2, 3])
    for a in results.values():
        np.testing.assert_array_equal(a, [1, 2, 3])
    assert results["array"].dtype == np.dtype(int)
    assert results["array_t<int32>"].dtype == np.int32
    assert results["array_t<double>"].dtype == np.float64


def test_overload_resolution(msg):
    # Exact overload matches:
    assert m.overloaded(np.array([1], dtype="float64")) == "double"
    assert m.overloaded(np.array([1], dtype="float32")) == "float"
    assert m.overloaded(np.array([1], dtype="ushort")) == "unsigned short"
    assert m.overloaded(np.array([1], dtype="intc")) == "int"
    assert m.overloaded(np.array([1], dtype="longlong")) == "long long"
    assert m.overloaded(np.array([1], dtype="complex")) == "double complex"
    assert m.overloaded(np.array([1], dtype="csingle")) == "float complex"

    # No exact match, should call first convertible version:
    assert m.overloaded(np.array([1], dtype="uint8")) == "double"

    with pytest.raises(TypeError) as excinfo:
        m.overloaded("not an array")
    assert (
        msg(excinfo.value)
        == """
        overloaded(): incompatible function arguments. The following argument types are supported:
            1. (arg0: numpy.ndarray[numpy.float64]) -> str
            2. (arg0: numpy.ndarray[numpy.float32]) -> str
            3. (arg0: numpy.ndarray[numpy.int32]) -> str
            4. (arg0: numpy.ndarray[numpy.uint16]) -> str
            5. (arg0: numpy.ndarray[numpy.int64]) -> str
            6. (arg0: numpy.ndarray[numpy.complex128]) -> str
            7. (arg0: numpy.ndarray[numpy.complex64]) -> str

        Invoked with: 'not an array'
    """
    )

    assert m.overloaded2(np.array([1], dtype="float64")) == "double"
    assert m.overloaded2(np.array([1], dtype="float32")) == "float"
    assert m.overloaded2(np.array([1], dtype="complex64")) == "float complex"
    assert m.overloaded2(np.array([1], dtype="complex128")) == "double complex"
    assert m.overloaded2(np.array([1], dtype="float32")) == "float"

    assert m.overloaded3(np.array([1], dtype="float64")) == "double"
    assert m.overloaded3(np.array([1], dtype="intc")) == "int"
    expected_exc = """
        overloaded3(): incompatible function arguments. The following argument types are supported:
            1. (arg0: numpy.ndarray[numpy.int32]) -> str
            2. (arg0: numpy.ndarray[numpy.float64]) -> str

        Invoked with: """

    with pytest.raises(TypeError) as excinfo:
        m.overloaded3(np.array([1], dtype="uintc"))
    assert msg(excinfo.value) == expected_exc + repr(np.array([1], dtype="uint32"))
    with pytest.raises(TypeError) as excinfo:
        m.overloaded3(np.array([1], dtype="float32"))
    assert msg(excinfo.value) == expected_exc + repr(np.array([1.0], dtype="float32"))
    with pytest.raises(TypeError) as excinfo:
        m.overloaded3(np.array([1], dtype="complex"))
    assert msg(excinfo.value) == expected_exc + repr(np.array([1.0 + 0.0j]))

    # Exact matches:
    assert m.overloaded4(np.array([1], dtype="double")) == "double"
    assert m.overloaded4(np.array([1], dtype="longlong")) == "long long"
    # Non-exact matches requiring conversion.  Since float to integer isn't a
    # save conversion, it should go to the double overload, but short can go to
    # either (and so should end up on the first-registered, the long long).
    assert m.overloaded4(np.array([1], dtype="float32")) == "double"
    assert m.overloaded4(np.array([1], dtype="short")) == "long long"

    assert m.overloaded5(np.array([1], dtype="double")) == "double"
    assert m.overloaded5(np.array([1], dtype="uintc")) == "unsigned int"
    assert m.overloaded5(np.array([1], dtype="float32")) == "unsigned int"


def test_greedy_string_overload():
    """Tests fix for #685 - ndarray shouldn't go to std::string overload"""

    assert m.issue685("abc") == "string"
    assert m.issue685(np.array([97, 98, 99], dtype="b")) == "array"
    assert m.issue685(123) == "other"


def test_array_unchecked_fixed_dims(msg):
    z1 = np.array([[1, 2], [3, 4]], dtype="float64")
    m.proxy_add2(z1, 10)
    assert np.all(z1 == [[11, 12], [13, 14]])

    with pytest.raises(ValueError) as excinfo:
        m.proxy_add2(np.array([1.0, 2, 3]), 5.0)
    assert (
        msg(excinfo.value) == "array has incorrect number of dimensions: 1; expected 2"
    )

    expect_c = np.ndarray(shape=(3, 3, 3), buffer=np.array(range(3, 30)), dtype="int")
    assert np.all(m.proxy_init3(3.0) == expect_c)
    expect_f = np.transpose(expect_c)
    assert np.all(m.proxy_init3F(3.0) == expect_f)

    assert m.proxy_squared_L2_norm(np.array(range(6))) == 55
    assert m.proxy_squared_L2_norm(np.array(range(6), dtype="float64")) == 55

    assert m.proxy_auxiliaries2(z1) == [11, 11, True, 2, 8, 2, 2, 4, 32]
    assert m.proxy_auxiliaries2(z1) == m.array_auxiliaries2(z1)

    assert m.proxy_auxiliaries1_const_ref(z1[0, :])
    assert m.proxy_auxiliaries2_const_ref(z1)


def test_array_unchecked_dyn_dims():
    z1 = np.array([[1, 2], [3, 4]], dtype="float64")
    m.proxy_add2_dyn(z1, 10)
    assert np.all(z1 == [[11, 12], [13, 14]])

    expect_c = np.ndarray(shape=(3, 3, 3), buffer=np.array(range(3, 30)), dtype="int")
    assert np.all(m.proxy_init3_dyn(3.0) == expect_c)

    assert m.proxy_auxiliaries2_dyn(z1) == [11, 11, True, 2, 8, 2, 2, 4, 32]
    assert m.proxy_auxiliaries2_dyn(z1) == m.array_auxiliaries2(z1)


def test_array_failure():
    with pytest.raises(ValueError) as excinfo:
        m.array_fail_test()
    assert str(excinfo.value) == "cannot create a pybind11::array from a nullptr"

    with pytest.raises(ValueError) as excinfo:
        m.array_t_fail_test()
    assert str(excinfo.value) == "cannot create a pybind11::array_t from a nullptr"

    with pytest.raises(ValueError) as excinfo:
        m.array_fail_test_negative_size()
    assert str(excinfo.value) == "negative dimensions are not allowed"


def test_initializer_list():
    assert m.array_initializer_list1().shape == (1,)
    assert m.array_initializer_list2().shape == (1, 2)
    assert m.array_initializer_list3().shape == (1, 2, 3)
    assert m.array_initializer_list4().shape == (1, 2, 3, 4)


def test_array_resize():
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype="float64")
    m.array_reshape2(a)
    assert a.size == 9
    assert np.all(a == [[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # total size change should succced with refcheck off
    m.array_resize3(a, 4, False)
    assert a.size == 64
    # ... and fail with refcheck on
    try:
        m.array_resize3(a, 3, True)
    except ValueError as e:
        assert str(e).startswith("cannot resize an array")  # noqa: PT017
    # transposed array doesn't own data
    b = a.transpose()
    try:
        m.array_resize3(b, 3, False)
    except ValueError as e:
        assert str(e).startswith(  # noqa: PT017
            "cannot resize this array: it does not own its data"
        )
    # ... but reshape should be fine
    m.array_reshape2(b)
    assert b.shape == (8, 8)


@pytest.mark.xfail("env.PYPY")
def test_array_create_and_resize():
    a = m.create_and_resize(2)
    assert a.size == 4
    assert np.all(a == 42.0)


def test_array_view():
    a = np.ones(100 * 4).astype("uint8")
    a_float_view = m.array_view(a, "float32")
    assert a_float_view.shape == (100 * 1,)  # 1 / 4 bytes = 8 / 32

    a_int16_view = m.array_view(a, "int16")  # 1 / 2 bytes = 16 / 32
    assert a_int16_view.shape == (100 * 2,)


def test_array_view_invalid():
    a = np.ones(100 * 4).astype("uint8")
    with pytest.raises(TypeError):
        m.array_view(a, "deadly_dtype")


def test_reshape_initializer_list():
    a = np.arange(2 * 7 * 3) + 1
    x = m.reshape_initializer_list(a, 2, 7, 3)
    assert x.shape == (2, 7, 3)
    assert list(x[1][4]) == [34, 35, 36]
    with pytest.raises(ValueError) as excinfo:
        m.reshape_initializer_list(a, 1, 7, 3)
    assert str(excinfo.value) == "cannot reshape array of size 42 into shape (1,7,3)"


def test_reshape_tuple():
    a = np.arange(3 * 7 * 2) + 1
    x = m.reshape_tuple(a, (3, 7, 2))
    assert x.shape == (3, 7, 2)
    assert list(x[1][4]) == [23, 24]
    y = m.reshape_tuple(x, (x.size,))
    assert y.shape == (42,)
    with pytest.raises(ValueError) as excinfo:
        m.reshape_tuple(a, (3, 7, 1))
    assert str(excinfo.value) == "cannot reshape array of size 42 into shape (3,7,1)"
    with pytest.raises(ValueError) as excinfo:
        m.reshape_tuple(a, ())
    assert str(excinfo.value) == "cannot reshape array of size 42 into shape ()"


def test_index_using_ellipsis():
    a = m.index_using_ellipsis(np.zeros((5, 6, 7)))
    assert a.shape == (6,)


@pytest.mark.parametrize(
    "test_func",
    [
        m.test_fmt_desc_float,
        m.test_fmt_desc_double,
        m.test_fmt_desc_const_float,
        m.test_fmt_desc_const_double,
    ],
)
def test_format_descriptors_for_floating_point_types(test_func):
    assert "numpy.ndarray[numpy.float" in test_func.__doc__


@pytest.mark.parametrize("forcecast", [False, True])
@pytest.mark.parametrize("contiguity", [None, "C", "F"])
@pytest.mark.parametrize("noconvert", [False, True])
@pytest.mark.filterwarnings(
    "ignore:Casting complex values to real discards the imaginary part:"
    + (
        "numpy.exceptions.ComplexWarning"
        if hasattr(np, "exceptions")
        else "numpy.ComplexWarning"
    )
)
def test_argument_conversions(forcecast, contiguity, noconvert):
    function_name = "accept_double"
    if contiguity == "C":
        function_name += "_c_style"
    elif contiguity == "F":
        function_name += "_f_style"
    if forcecast:
        function_name += "_forcecast"
    if noconvert:
        function_name += "_noconvert"
    function = getattr(m, function_name)

    for dtype in [np.dtype("float32"), np.dtype("float64"), np.dtype("complex128")]:
        for order in ["C", "F"]:
            for shape in [(2, 2), (1, 3, 1, 1), (1, 1, 1), (0,)]:
                if not noconvert:
                    # If noconvert is not passed, only complex128 needs to be truncated and
                    # "cannot be safely obtained". So without `forcecast`, the argument shouldn't
                    # be accepted.
                    should_raise = dtype.name == "complex128" and not forcecast
                else:
                    # If noconvert is passed, only float64 and the matching order is accepted.
                    # If at most one dimension has a size greater than 1, the array is also
                    # trivially contiguous.
                    trivially_contiguous = sum(1 for d in shape if d > 1) <= 1
                    should_raise = dtype.name != "float64" or (
                        contiguity is not None
                        and contiguity != order
                        and not trivially_contiguous
                    )

                array = np.zeros(shape, dtype=dtype, order=order)
                if not should_raise:
                    function(array)
                else:
                    with pytest.raises(
                        TypeError, match="incompatible function arguments"
                    ):
                        function(array)


@pytest.mark.xfail("env.PYPY")
def test_dtype_refcount_leak():
    from sys import getrefcount

    # Was np.float_ but that alias for float64 was removed in NumPy 2.
    dtype = np.dtype(np.float64)
    a = np.array([1], dtype=dtype)
    before = getrefcount(dtype)
    m.ndim(a)
    after = getrefcount(dtype)
    assert after == before


def test_round_trip_float():
    arr = np.zeros((), np.float64)
    arr[()] = 37.2
    assert m.round_trip_float(arr) == 37.2


# HINT: An easy and robust way (although only manual unfortunately) to check for
#       ref-count leaks in the test_.*pyobject_ptr.* functions below is to
#           * temporarily insert `while True:` (one-by-one),
#           * run this test, and
#           * run the Linux `top` command in another shell to visually monitor
#             `RES` for a minute or two.
#       If there is a leak, it is usually evident in seconds because the `RES`
#       value increases without bounds. (Don't forget to Ctrl-C the test!)


# For use as a temporary user-defined object, to maximize sensitivity of the tests below:
#     * Ref-count leaks will be immediately evident.
#     * Sanitizers are much more likely to detect heap-use-after-free due to
#       other ref-count bugs.
class PyValueHolder:
    def __init__(self, value):
        self.value = value


def WrapWithPyValueHolder(*values):
    return [PyValueHolder(v) for v in values]


def UnwrapPyValueHolder(vhs):
    return [vh.value for vh in vhs]


def test_pass_array_pyobject_ptr_return_sum_str_values_ndarray():
    # Intentionally all temporaries, do not change.
    assert (
        m.pass_array_pyobject_ptr_return_sum_str_values(
            np.array(WrapWithPyValueHolder(-3, "four", 5.0), dtype=object)
        )
        == "-3four5.0"
    )


def test_pass_array_pyobject_ptr_return_sum_str_values_list():
    # Intentionally all temporaries, do not change.
    assert (
        m.pass_array_pyobject_ptr_return_sum_str_values(
            WrapWithPyValueHolder(2, "three", -4.0)
        )
        == "2three-4.0"
    )


def test_pass_array_pyobject_ptr_return_as_list():
    # Intentionally all temporaries, do not change.
    assert UnwrapPyValueHolder(
        m.pass_array_pyobject_ptr_return_as_list(
            np.array(WrapWithPyValueHolder(-1, "two", 3.0), dtype=object)
        )
    ) == [-1, "two", 3.0]


@pytest.mark.parametrize(
    ("return_array_pyobject_ptr", "unwrap"),
    [
        (m.return_array_pyobject_ptr_cpp_loop, list),
        (m.return_array_pyobject_ptr_from_list, UnwrapPyValueHolder),
    ],
)
def test_return_array_pyobject_ptr_cpp_loop(return_array_pyobject_ptr, unwrap):
    # Intentionally all temporaries, do not change.
    arr_from_list = return_array_pyobject_ptr(WrapWithPyValueHolder(6, "seven", -8.0))
    assert isinstance(arr_from_list, np.ndarray)
    assert arr_from_list.dtype == np.dtype("O")
    assert unwrap(arr_from_list) == [6, "seven", -8.0]
