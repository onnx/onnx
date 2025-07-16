from __future__ import annotations

import re

import pytest

import env  # noqa: F401
from pybind11_tests import PYBIND11_NUMPY_1_ONLY
from pybind11_tests import numpy_dtypes as m

np = pytest.importorskip("numpy")


@pytest.fixture(scope="module")
def simple_dtype():
    ld = np.dtype("longdouble")
    return np.dtype(
        {
            "names": ["bool_", "uint_", "float_", "ldbl_"],
            "formats": ["?", "u4", "f4", f"f{ld.itemsize}"],
            "offsets": [0, 4, 8, (16 if ld.alignment > 4 else 12)],
        }
    )


@pytest.fixture(scope="module")
def packed_dtype():
    return np.dtype([("bool_", "?"), ("uint_", "u4"), ("float_", "f4"), ("ldbl_", "g")])


def dt_fmt():
    from sys import byteorder

    e = "<" if byteorder == "little" else ">"
    return (
        "{{'names':['bool_','uint_','float_','ldbl_'],"
        "'formats':['?','" + e + "u4','" + e + "f4','" + e + "f{}'],"
        "'offsets':[0,4,8,{}],'itemsize':{}}}"
    )


def simple_dtype_fmt():
    ld = np.dtype("longdouble")
    simple_ld_off = 12 + 4 * (ld.alignment > 4)
    return dt_fmt().format(ld.itemsize, simple_ld_off, simple_ld_off + ld.itemsize)


def packed_dtype_fmt():
    from sys import byteorder

    return "[('bool_','?'),('uint_','{e}u4'),('float_','{e}f4'),('ldbl_','{e}f{}')]".format(
        np.dtype("longdouble").itemsize, e="<" if byteorder == "little" else ">"
    )


def partial_ld_offset():
    return (
        12
        + 4 * (np.dtype("uint64").alignment > 4)
        + 8
        + 8 * (np.dtype("longdouble").alignment > 8)
    )


def partial_dtype_fmt():
    ld = np.dtype("longdouble")
    partial_ld_off = partial_ld_offset()
    partial_size = partial_ld_off + ld.itemsize
    partial_end_padding = partial_size % np.dtype("uint64").alignment
    return dt_fmt().format(
        ld.itemsize, partial_ld_off, partial_size + partial_end_padding
    )


def partial_nested_fmt():
    ld = np.dtype("longdouble")
    partial_nested_off = 8 + 8 * (ld.alignment > 8)
    partial_ld_off = partial_ld_offset()
    partial_size = partial_ld_off + ld.itemsize
    partial_end_padding = partial_size % np.dtype("uint64").alignment
    partial_nested_size = partial_nested_off * 2 + partial_size + partial_end_padding
    return f"{{'names':['a'],'formats':[{partial_dtype_fmt()}],'offsets':[{partial_nested_off}],'itemsize':{partial_nested_size}}}"


def assert_equal(actual, expected_data, expected_dtype):
    np.testing.assert_equal(actual, np.array(expected_data, dtype=expected_dtype))


def test_format_descriptors():
    with pytest.raises(RuntimeError) as excinfo:
        m.get_format_unbound()
    assert re.match(
        "^NumPy type info missing for .*UnboundStruct.*$", str(excinfo.value)
    )

    ld = np.dtype("longdouble")
    ldbl_fmt = ("4x" if ld.alignment > 4 else "") + ld.char
    ss_fmt = "^T{?:bool_:3xI:uint_:f:float_:" + ldbl_fmt + ":ldbl_:}"
    dbl = np.dtype("double")
    end_padding = ld.itemsize % np.dtype("uint64").alignment
    partial_fmt = (
        "^T{?:bool_:3xI:uint_:f:float_:"
        + str(4 * (dbl.alignment > 4) + dbl.itemsize + 8 * (ld.alignment > 8))
        + "xg:ldbl_:"
        + (str(end_padding) + "x}" if end_padding > 0 else "}")
    )
    nested_extra = str(max(8, ld.alignment))
    assert m.print_format_descriptors() == [
        ss_fmt,
        "^T{?:bool_:I:uint_:f:float_:g:ldbl_:}",
        "^T{" + ss_fmt + ":a:^T{?:bool_:I:uint_:f:float_:g:ldbl_:}:b:}",
        partial_fmt,
        "^T{" + nested_extra + "x" + partial_fmt + ":a:" + nested_extra + "x}",
        "^T{3s:a:3s:b:}",
        "^T{(3)4s:a:(2)i:b:(3)B:c:1x(4, 2)f:d:}",
        "^T{q:e1:B:e2:}",
        "^T{Zf:cflt:Zd:cdbl:}",
    ]


def test_dtype(simple_dtype):
    from sys import byteorder

    e = "<" if byteorder == "little" else ">"

    assert [x.replace(" ", "") for x in m.print_dtypes()] == [
        simple_dtype_fmt(),
        packed_dtype_fmt(),
        f"[('a',{simple_dtype_fmt()}),('b',{packed_dtype_fmt()})]",
        partial_dtype_fmt(),
        partial_nested_fmt(),
        "[('a','S3'),('b','S3')]",
        (
            "{'names':['a','b','c','d'],"
            f"'formats':[('S4',(3,)),('{e}i4',(2,)),('u1',(3,)),('{e}f4',(4,2))],"
            "'offsets':[0,12,20,24],'itemsize':56}"
        ),
        "[('e1','" + e + "i8'),('e2','u1')]",
        "[('x','i1'),('y','" + e + "u8')]",
        "[('cflt','" + e + "c8'),('cdbl','" + e + "c16')]",
    ]

    d1 = np.dtype(
        {
            "names": ["a", "b"],
            "formats": ["int32", "float64"],
            "offsets": [1, 10],
            "itemsize": 20,
        }
    )
    d2 = np.dtype([("a", "i4"), ("b", "f4")])
    assert m.test_dtype_ctors() == [
        np.dtype("int32"),
        np.dtype("float64"),
        np.dtype("bool"),
        d1,
        d1,
        np.dtype("uint32"),
        d2,
        np.dtype("d"),
    ]

    assert m.test_dtype_methods() == [
        np.dtype("int32"),
        simple_dtype,
        False,
        True,
        np.dtype("int32").itemsize,
        simple_dtype.itemsize,
    ]

    assert m.trailing_padding_dtype() == m.buffer_to_dtype(
        np.zeros(1, m.trailing_padding_dtype())
    )

    expected_chars = list("bhilqBHILQefdgFDG?MmO")
    # Note that int_ and uint size and mapping is NumPy version dependent:
    expected_chars += [np.dtype(_).char for _ in ("int_", "uint", "intp", "uintp")]
    assert m.test_dtype_kind() == list("iiiiiuuuuuffffcccbMmOiuiu")
    assert m.test_dtype_char_() == list(expected_chars)
    assert m.test_dtype_num() == [np.dtype(ch).num for ch in expected_chars]
    assert m.test_dtype_byteorder() == [np.dtype(ch).byteorder for ch in expected_chars]
    assert m.test_dtype_alignment() == [np.dtype(ch).alignment for ch in expected_chars]
    if not PYBIND11_NUMPY_1_ONLY:
        assert m.test_dtype_flags() == [np.dtype(ch).flags for ch in expected_chars]
    else:
        assert m.test_dtype_flags() == [
            chr(np.dtype(ch).flags) for ch in expected_chars
        ]


def test_recarray(simple_dtype, packed_dtype):
    elements = [(False, 0, 0.0, -0.0), (True, 1, 1.5, -2.5), (False, 2, 3.0, -5.0)]

    for func, dtype in [
        (m.create_rec_simple, simple_dtype),
        (m.create_rec_packed, packed_dtype),
    ]:
        arr = func(0)
        assert arr.dtype == dtype
        assert_equal(arr, [], simple_dtype)
        assert_equal(arr, [], packed_dtype)

        arr = func(3)
        assert arr.dtype == dtype
        assert_equal(arr, elements, simple_dtype)
        assert_equal(arr, elements, packed_dtype)

        # Show what recarray's look like in NumPy.
        assert type(arr[0]) == np.void
        assert type(arr[0].item()) == tuple

        if dtype == simple_dtype:
            assert m.print_rec_simple(arr) == [
                "s:0,0,0,-0",
                "s:1,1,1.5,-2.5",
                "s:0,2,3,-5",
            ]
        else:
            assert m.print_rec_packed(arr) == [
                "p:0,0,0,-0",
                "p:1,1,1.5,-2.5",
                "p:0,2,3,-5",
            ]

    nested_dtype = np.dtype([("a", simple_dtype), ("b", packed_dtype)])

    arr = m.create_rec_nested(0)
    assert arr.dtype == nested_dtype
    assert_equal(arr, [], nested_dtype)

    arr = m.create_rec_nested(3)
    assert arr.dtype == nested_dtype
    assert_equal(
        arr,
        [
            ((False, 0, 0.0, -0.0), (True, 1, 1.5, -2.5)),
            ((True, 1, 1.5, -2.5), (False, 2, 3.0, -5.0)),
            ((False, 2, 3.0, -5.0), (True, 3, 4.5, -7.5)),
        ],
        nested_dtype,
    )
    assert m.print_rec_nested(arr) == [
        "n:a=s:0,0,0,-0;b=p:1,1,1.5,-2.5",
        "n:a=s:1,1,1.5,-2.5;b=p:0,2,3,-5",
        "n:a=s:0,2,3,-5;b=p:1,3,4.5,-7.5",
    ]

    arr = m.create_rec_partial(3)
    assert str(arr.dtype).replace(" ", "") == partial_dtype_fmt()
    partial_dtype = arr.dtype
    assert "" not in arr.dtype.fields
    assert partial_dtype.itemsize > simple_dtype.itemsize
    assert_equal(arr, elements, simple_dtype)
    assert_equal(arr, elements, packed_dtype)

    arr = m.create_rec_partial_nested(3)
    assert str(arr.dtype).replace(" ", "") == partial_nested_fmt()
    assert "" not in arr.dtype.fields
    assert "" not in arr.dtype.fields["a"][0].fields
    assert arr.dtype.itemsize > partial_dtype.itemsize
    np.testing.assert_equal(arr["a"], m.create_rec_partial(3))


def test_array_constructors():
    data = np.arange(1, 7, dtype="int32")
    for i in range(8):
        np.testing.assert_array_equal(m.test_array_ctors(10 + i), data.reshape((3, 2)))
        np.testing.assert_array_equal(m.test_array_ctors(20 + i), data.reshape((3, 2)))
    for i in range(5):
        np.testing.assert_array_equal(m.test_array_ctors(30 + i), data)
        np.testing.assert_array_equal(m.test_array_ctors(40 + i), data)


def test_string_array():
    arr = m.create_string_array(True)
    assert str(arr.dtype) == "[('a', 'S3'), ('b', 'S3')]"
    assert m.print_string_array(arr) == [
        "a='',b=''",
        "a='a',b='a'",
        "a='ab',b='ab'",
        "a='abc',b='abc'",
    ]
    dtype = arr.dtype
    assert arr["a"].tolist() == [b"", b"a", b"ab", b"abc"]
    assert arr["b"].tolist() == [b"", b"a", b"ab", b"abc"]
    arr = m.create_string_array(False)
    assert dtype == arr.dtype


def test_array_array():
    from sys import byteorder

    e = "<" if byteorder == "little" else ">"

    arr = m.create_array_array(3)
    assert str(arr.dtype).replace(" ", "") == (
        "{'names':['a','b','c','d'],"
        f"'formats':[('S4',(3,)),('{e}i4',(2,)),('u1',(3,)),('{e}f4',(4,2))],"
        "'offsets':[0,12,20,24],'itemsize':56}"
    )
    assert m.print_array_array(arr) == [
        "a={{A,B,C,D},{K,L,M,N},{U,V,W,X}},b={0,1},"
        "c={0,1,2},d={{0,1},{10,11},{20,21},{30,31}}",
        "a={{W,X,Y,Z},{G,H,I,J},{Q,R,S,T}},b={1000,1001},"
        "c={10,11,12},d={{100,101},{110,111},{120,121},{130,131}}",
        "a={{S,T,U,V},{C,D,E,F},{M,N,O,P}},b={2000,2001},"
        "c={20,21,22},d={{200,201},{210,211},{220,221},{230,231}}",
    ]
    assert arr["a"].tolist() == [
        [b"ABCD", b"KLMN", b"UVWX"],
        [b"WXYZ", b"GHIJ", b"QRST"],
        [b"STUV", b"CDEF", b"MNOP"],
    ]
    assert arr["b"].tolist() == [[0, 1], [1000, 1001], [2000, 2001]]
    assert m.create_array_array(0).dtype == arr.dtype


def test_enum_array():
    from sys import byteorder

    e = "<" if byteorder == "little" else ">"

    arr = m.create_enum_array(3)
    dtype = arr.dtype
    assert dtype == np.dtype([("e1", e + "i8"), ("e2", "u1")])
    assert m.print_enum_array(arr) == ["e1=A,e2=X", "e1=B,e2=Y", "e1=A,e2=X"]
    assert arr["e1"].tolist() == [-1, 1, -1]
    assert arr["e2"].tolist() == [1, 2, 1]
    assert m.create_enum_array(0).dtype == dtype


def test_complex_array():
    from sys import byteorder

    e = "<" if byteorder == "little" else ">"

    arr = m.create_complex_array(3)
    dtype = arr.dtype
    assert dtype == np.dtype([("cflt", e + "c8"), ("cdbl", e + "c16")])
    assert m.print_complex_array(arr) == [
        "c:(0,0.25),(0.5,0.75)",
        "c:(1,1.25),(1.5,1.75)",
        "c:(2,2.25),(2.5,2.75)",
    ]
    assert arr["cflt"].tolist() == [0.0 + 0.25j, 1.0 + 1.25j, 2.0 + 2.25j]
    assert arr["cdbl"].tolist() == [0.5 + 0.75j, 1.5 + 1.75j, 2.5 + 2.75j]
    assert m.create_complex_array(0).dtype == dtype


def test_signature(doc):
    assert (
        doc(m.create_rec_nested)
        == "create_rec_nested(arg0: int) -> numpy.ndarray[NestedStruct]"
    )


def test_scalar_conversion():
    n = 3
    arrays = [
        m.create_rec_simple(n),
        m.create_rec_packed(n),
        m.create_rec_nested(n),
        m.create_enum_array(n),
    ]
    funcs = [m.f_simple, m.f_packed, m.f_nested]

    for i, func in enumerate(funcs):
        for j, arr in enumerate(arrays):
            if i == j and i < 2:
                assert [func(arr[k]) for k in range(n)] == [k * 10 for k in range(n)]
            else:
                with pytest.raises(TypeError) as excinfo:
                    func(arr[0])
                assert "incompatible function arguments" in str(excinfo.value)


def test_vectorize():
    n = 3
    array = m.create_rec_simple(n)
    values = m.f_simple_vectorized(array)
    np.testing.assert_array_equal(values, [0, 10, 20])
    array_2 = m.f_simple_pass_thru_vectorized(array)
    np.testing.assert_array_equal(array, array_2)


def test_cls_and_dtype_conversion(simple_dtype):
    s = m.SimpleStruct()
    assert s.astuple() == (False, 0, 0.0, 0.0)
    assert m.SimpleStruct.fromtuple(s.astuple()).astuple() == s.astuple()

    s.uint_ = 2
    assert m.f_simple(s) == 20

    # Try as recarray of shape==(1,).
    s_recarray = np.array([(False, 2, 0.0, 0.0)], dtype=simple_dtype)
    # Show that this will work for vectorized case.
    np.testing.assert_array_equal(m.f_simple_vectorized(s_recarray), [20])

    # Show as a scalar that inherits from np.generic.
    s_scalar = s_recarray[0]
    assert isinstance(s_scalar, np.void)
    assert m.f_simple(s_scalar) == 20

    # Show that an *array* scalar (np.ndarray.shape == ()) does not convert.
    # More specifically, conversion to SimpleStruct is not implicit.
    s_recarray_scalar = s_recarray.reshape(())
    assert isinstance(s_recarray_scalar, np.ndarray)
    assert s_recarray_scalar.dtype == simple_dtype
    with pytest.raises(TypeError) as excinfo:
        m.f_simple(s_recarray_scalar)
    assert "incompatible function arguments" in str(excinfo.value)
    # Explicitly convert to m.SimpleStruct.
    assert m.f_simple(m.SimpleStruct.fromtuple(s_recarray_scalar.item())) == 20

    # Show that an array of dtype=object does *not* convert.
    s_array_object = np.array([s])
    assert s_array_object.dtype == object
    with pytest.raises(TypeError) as excinfo:
        m.f_simple_vectorized(s_array_object)
    assert "incompatible function arguments" in str(excinfo.value)
    # Explicitly convert to `np.array(..., dtype=simple_dtype)`
    s_array = np.array([s.astuple()], dtype=simple_dtype)
    np.testing.assert_array_equal(m.f_simple_vectorized(s_array), [20])


def test_register_dtype():
    with pytest.raises(RuntimeError) as excinfo:
        m.register_dtype()
    assert "dtype is already registered" in str(excinfo.value)


@pytest.mark.xfail("env.PYPY")
def test_str_leak():
    from sys import getrefcount

    fmt = "f4"
    pytest.gc_collect()
    start = getrefcount(fmt)
    d = m.dtype_wrapper(fmt)
    assert d is np.dtype("f4")
    del d
    pytest.gc_collect()
    assert getrefcount(fmt) == start


def test_compare_buffer_info():
    assert all(m.compare_buffer_info())
