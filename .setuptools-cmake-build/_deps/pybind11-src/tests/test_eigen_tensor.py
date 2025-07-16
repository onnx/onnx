from __future__ import annotations

import sys

import pytest

np = pytest.importorskip("numpy")
eigen_tensor = pytest.importorskip("pybind11_tests.eigen_tensor")
submodules = [eigen_tensor.c_style, eigen_tensor.f_style]
try:
    import eigen_tensor_avoid_stl_array as avoid

    submodules += [avoid.c_style, avoid.f_style]
except ImportError as e:
    # Ensure config, build, toolchain, etc. issues are not masked here:
    msg = (
        "import eigen_tensor_avoid_stl_array FAILED, while "
        "import pybind11_tests.eigen_tensor succeeded. "
        "Please ensure that "
        "test_eigen_tensor.cpp & "
        "eigen_tensor_avoid_stl_array.cpp "
        "are built together (or both are not built if Eigen is not available)."
    )
    raise RuntimeError(msg) from e

tensor_ref = np.empty((3, 5, 2), dtype=np.int64)

for i in range(tensor_ref.shape[0]):
    for j in range(tensor_ref.shape[1]):
        for k in range(tensor_ref.shape[2]):
            tensor_ref[i, j, k] = i * (5 * 2) + j * 2 + k

indices = (2, 3, 1)


@pytest.fixture(autouse=True)
def cleanup():
    for module in submodules:
        module.setup()

    yield

    for module in submodules:
        assert module.is_ok()


def test_import_avoid_stl_array():
    pytest.importorskip("eigen_tensor_avoid_stl_array")
    assert len(submodules) == 4


def assert_equal_tensor_ref(mat, writeable=True, modified=None):
    assert mat.flags.writeable == writeable

    copy = np.array(tensor_ref)
    if modified is not None:
        copy[indices] = modified

    np.testing.assert_array_equal(mat, copy)


@pytest.mark.parametrize("m", submodules)
@pytest.mark.parametrize("member_name", ["member", "member_view"])
def test_reference_internal(m, member_name):
    if not hasattr(sys, "getrefcount"):
        pytest.skip("No reference counting")
    foo = m.CustomExample()
    counts = sys.getrefcount(foo)
    mem = getattr(foo, member_name)
    assert_equal_tensor_ref(mem, writeable=False)
    new_counts = sys.getrefcount(foo)
    assert new_counts == counts + 1
    assert_equal_tensor_ref(mem, writeable=False)
    del mem
    assert sys.getrefcount(foo) == counts


assert_equal_funcs = [
    "copy_tensor",
    "copy_fixed_tensor",
    "copy_const_tensor",
    "move_tensor_copy",
    "move_fixed_tensor_copy",
    "take_tensor",
    "take_fixed_tensor",
    "reference_tensor",
    "reference_tensor_v2",
    "reference_fixed_tensor",
    "reference_view_of_tensor",
    "reference_view_of_tensor_v3",
    "reference_view_of_tensor_v5",
    "reference_view_of_fixed_tensor",
]

assert_equal_const_funcs = [
    "reference_view_of_tensor_v2",
    "reference_view_of_tensor_v4",
    "reference_view_of_tensor_v6",
    "reference_const_tensor",
    "reference_const_tensor_v2",
]


@pytest.mark.parametrize("m", submodules)
@pytest.mark.parametrize("func_name", assert_equal_funcs + assert_equal_const_funcs)
def test_convert_tensor_to_py(m, func_name):
    writeable = func_name in assert_equal_funcs
    assert_equal_tensor_ref(getattr(m, func_name)(), writeable=writeable)


@pytest.mark.parametrize("m", submodules)
def test_bad_cpp_to_python_casts(m):
    with pytest.raises(
        RuntimeError, match="Cannot use reference internal when there is no parent"
    ):
        m.reference_tensor_internal()

    with pytest.raises(RuntimeError, match="Cannot move from a constant reference"):
        m.move_const_tensor()

    with pytest.raises(
        RuntimeError, match="Cannot take ownership of a const reference"
    ):
        m.take_const_tensor()

    with pytest.raises(
        RuntimeError,
        match="Invalid return_value_policy for Eigen Map type, must be either reference or reference_internal",
    ):
        m.take_view_tensor()


@pytest.mark.parametrize("m", submodules)
def test_bad_python_to_cpp_casts(m):
    with pytest.raises(
        TypeError, match=r"^round_trip_tensor\(\): incompatible function arguments"
    ):
        m.round_trip_tensor(np.zeros((2, 3)))

    with pytest.raises(TypeError, match=r"^Cannot cast array data from dtype"):
        m.round_trip_tensor(np.zeros(dtype=np.str_, shape=(2, 3, 1)))

    with pytest.raises(
        TypeError,
        match=r"^round_trip_tensor_noconvert\(\): incompatible function arguments",
    ):
        m.round_trip_tensor_noconvert(tensor_ref)

    assert_equal_tensor_ref(
        m.round_trip_tensor_noconvert(tensor_ref.astype(np.float64))
    )

    bad_options = "C" if m.needed_options == "F" else "F"
    # Shape, dtype and the order need to be correct for a TensorMap cast
    with pytest.raises(
        TypeError, match=r"^round_trip_view_tensor\(\): incompatible function arguments"
    ):
        m.round_trip_view_tensor(
            np.zeros((3, 5, 2), dtype=np.float64, order=bad_options)
        )

    with pytest.raises(
        TypeError, match=r"^round_trip_view_tensor\(\): incompatible function arguments"
    ):
        m.round_trip_view_tensor(
            np.zeros((3, 5, 2), dtype=np.float32, order=m.needed_options)
        )

    with pytest.raises(
        TypeError, match=r"^round_trip_view_tensor\(\): incompatible function arguments"
    ):
        m.round_trip_view_tensor(
            np.zeros((3, 5), dtype=np.float64, order=m.needed_options)
        )

    temp = np.zeros((3, 5, 2), dtype=np.float64, order=m.needed_options)
    with pytest.raises(
        TypeError, match=r"^round_trip_view_tensor\(\): incompatible function arguments"
    ):
        m.round_trip_view_tensor(
            temp[:, ::-1, :],
        )

    temp = np.zeros((3, 5, 2), dtype=np.float64, order=m.needed_options)
    temp.setflags(write=False)
    with pytest.raises(
        TypeError, match=r"^round_trip_view_tensor\(\): incompatible function arguments"
    ):
        m.round_trip_view_tensor(temp)


@pytest.mark.parametrize("m", submodules)
def test_references_actually_refer(m):
    a = m.reference_tensor()
    temp = a[indices]
    a[indices] = 100
    assert_equal_tensor_ref(m.copy_const_tensor(), modified=100)
    a[indices] = temp
    assert_equal_tensor_ref(m.copy_const_tensor())

    a = m.reference_view_of_tensor()
    a[indices] = 100
    assert_equal_tensor_ref(m.copy_const_tensor(), modified=100)
    a[indices] = temp
    assert_equal_tensor_ref(m.copy_const_tensor())


@pytest.mark.parametrize("m", submodules)
def test_round_trip(m):
    assert_equal_tensor_ref(m.round_trip_tensor(tensor_ref))

    with pytest.raises(TypeError, match="^Cannot cast array data from"):
        assert_equal_tensor_ref(m.round_trip_tensor2(tensor_ref))

    assert_equal_tensor_ref(m.round_trip_tensor2(np.array(tensor_ref, dtype=np.int32)))
    assert_equal_tensor_ref(m.round_trip_fixed_tensor(tensor_ref))
    assert_equal_tensor_ref(m.round_trip_aligned_view_tensor(m.reference_tensor()))

    copy = np.array(tensor_ref, dtype=np.float64, order=m.needed_options)
    assert_equal_tensor_ref(m.round_trip_view_tensor(copy))
    assert_equal_tensor_ref(m.round_trip_view_tensor_ref(copy))
    assert_equal_tensor_ref(m.round_trip_view_tensor_ptr(copy))
    copy.setflags(write=False)
    assert_equal_tensor_ref(m.round_trip_const_view_tensor(copy))

    np.testing.assert_array_equal(
        tensor_ref[:, ::-1, :], m.round_trip_tensor(tensor_ref[:, ::-1, :])
    )

    assert m.round_trip_rank_0(np.float64(3.5)) == 3.5
    assert m.round_trip_rank_0(3.5) == 3.5

    with pytest.raises(
        TypeError,
        match=r"^round_trip_rank_0_noconvert\(\): incompatible function arguments",
    ):
        m.round_trip_rank_0_noconvert(np.float64(3.5))

    with pytest.raises(
        TypeError,
        match=r"^round_trip_rank_0_noconvert\(\): incompatible function arguments",
    ):
        m.round_trip_rank_0_noconvert(3.5)

    with pytest.raises(
        TypeError, match=r"^round_trip_rank_0_view\(\): incompatible function arguments"
    ):
        m.round_trip_rank_0_view(np.float64(3.5))

    with pytest.raises(
        TypeError, match=r"^round_trip_rank_0_view\(\): incompatible function arguments"
    ):
        m.round_trip_rank_0_view(3.5)


@pytest.mark.parametrize("m", submodules)
def test_round_trip_references_actually_refer(m):
    # Need to create a copy that matches the type on the C side
    copy = np.array(tensor_ref, dtype=np.float64, order=m.needed_options)
    a = m.round_trip_view_tensor(copy)
    temp = a[indices]
    a[indices] = 100
    assert_equal_tensor_ref(copy, modified=100)
    a[indices] = temp
    assert_equal_tensor_ref(copy)


@pytest.mark.parametrize("m", submodules)
def test_doc_string(m, doc):
    assert (
        doc(m.copy_tensor) == "copy_tensor() -> numpy.ndarray[numpy.float64[?, ?, ?]]"
    )
    assert (
        doc(m.copy_fixed_tensor)
        == "copy_fixed_tensor() -> numpy.ndarray[numpy.float64[3, 5, 2]]"
    )
    assert (
        doc(m.reference_const_tensor)
        == "reference_const_tensor() -> numpy.ndarray[numpy.float64[?, ?, ?]]"
    )

    order_flag = f"flags.{m.needed_options.lower()}_contiguous"
    assert doc(m.round_trip_view_tensor) == (
        f"round_trip_view_tensor(arg0: numpy.ndarray[numpy.float64[?, ?, ?], flags.writeable, {order_flag}])"
        f" -> numpy.ndarray[numpy.float64[?, ?, ?], flags.writeable, {order_flag}]"
    )
    assert doc(m.round_trip_const_view_tensor) == (
        f"round_trip_const_view_tensor(arg0: numpy.ndarray[numpy.float64[?, ?, ?], {order_flag}])"
        " -> numpy.ndarray[numpy.float64[?, ?, ?]]"
    )
