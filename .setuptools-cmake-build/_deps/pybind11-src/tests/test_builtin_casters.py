from __future__ import annotations

import sys

import pytest

import env
from pybind11_tests import IncType, UserType
from pybind11_tests import builtin_casters as m


def test_simple_string():
    assert m.string_roundtrip("const char *") == "const char *"


def test_unicode_conversion():
    """Tests unicode conversion and error reporting."""
    assert m.good_utf8_string() == "Say utf8‽ 🎂 𝐀"
    assert m.good_utf16_string() == "b‽🎂𝐀z"
    assert m.good_utf32_string() == "a𝐀🎂‽z"
    assert m.good_wchar_string() == "a⸘𝐀z"
    if hasattr(m, "has_u8string"):
        assert m.good_utf8_u8string() == "Say utf8‽ 🎂 𝐀"

    with pytest.raises(UnicodeDecodeError):
        m.bad_utf8_string()

    with pytest.raises(UnicodeDecodeError):
        m.bad_utf16_string()

    # These are provided only if they actually fail (they don't when 32-bit)
    if hasattr(m, "bad_utf32_string"):
        with pytest.raises(UnicodeDecodeError):
            m.bad_utf32_string()
    if hasattr(m, "bad_wchar_string"):
        with pytest.raises(UnicodeDecodeError):
            m.bad_wchar_string()
    if hasattr(m, "has_u8string"):
        with pytest.raises(UnicodeDecodeError):
            m.bad_utf8_u8string()

    assert m.u8_Z() == "Z"
    assert m.u8_eacute() == "é"
    assert m.u16_ibang() == "‽"
    assert m.u32_mathbfA() == "𝐀"
    assert m.wchar_heart() == "♥"
    if hasattr(m, "has_u8string"):
        assert m.u8_char8_Z() == "Z"


def test_single_char_arguments():
    """Tests failures for passing invalid inputs to char-accepting functions"""

    def toobig_message(r):
        return f"Character code point not in range({r:#x})"

    toolong_message = "Expected a character, but multi-character string found"

    assert m.ord_char("a") == 0x61  # simple ASCII
    assert m.ord_char_lv("b") == 0x62
    assert (
        m.ord_char("é") == 0xE9
    )  # requires 2 bytes in utf-8, but can be stuffed in a char
    with pytest.raises(ValueError) as excinfo:
        assert m.ord_char("Ā") == 0x100  # requires 2 bytes, doesn't fit in a char
    assert str(excinfo.value) == toobig_message(0x100)
    with pytest.raises(ValueError) as excinfo:
        assert m.ord_char("ab")
    assert str(excinfo.value) == toolong_message

    assert m.ord_char16("a") == 0x61
    assert m.ord_char16("é") == 0xE9
    assert m.ord_char16_lv("ê") == 0xEA
    assert m.ord_char16("Ā") == 0x100
    assert m.ord_char16("‽") == 0x203D
    assert m.ord_char16("♥") == 0x2665
    assert m.ord_char16_lv("♡") == 0x2661
    with pytest.raises(ValueError) as excinfo:
        assert m.ord_char16("🎂") == 0x1F382  # requires surrogate pair
    assert str(excinfo.value) == toobig_message(0x10000)
    with pytest.raises(ValueError) as excinfo:
        assert m.ord_char16("aa")
    assert str(excinfo.value) == toolong_message

    assert m.ord_char32("a") == 0x61
    assert m.ord_char32("é") == 0xE9
    assert m.ord_char32("Ā") == 0x100
    assert m.ord_char32("‽") == 0x203D
    assert m.ord_char32("♥") == 0x2665
    assert m.ord_char32("🎂") == 0x1F382
    with pytest.raises(ValueError) as excinfo:
        assert m.ord_char32("aa")
    assert str(excinfo.value) == toolong_message

    assert m.ord_wchar("a") == 0x61
    assert m.ord_wchar("é") == 0xE9
    assert m.ord_wchar("Ā") == 0x100
    assert m.ord_wchar("‽") == 0x203D
    assert m.ord_wchar("♥") == 0x2665
    if m.wchar_size == 2:
        with pytest.raises(ValueError) as excinfo:
            assert m.ord_wchar("🎂") == 0x1F382  # requires surrogate pair
        assert str(excinfo.value) == toobig_message(0x10000)
    else:
        assert m.ord_wchar("🎂") == 0x1F382
    with pytest.raises(ValueError) as excinfo:
        assert m.ord_wchar("aa")
    assert str(excinfo.value) == toolong_message

    if hasattr(m, "has_u8string"):
        assert m.ord_char8("a") == 0x61  # simple ASCII
        assert m.ord_char8_lv("b") == 0x62
        assert (
            m.ord_char8("é") == 0xE9
        )  # requires 2 bytes in utf-8, but can be stuffed in a char
        with pytest.raises(ValueError) as excinfo:
            assert m.ord_char8("Ā") == 0x100  # requires 2 bytes, doesn't fit in a char
        assert str(excinfo.value) == toobig_message(0x100)
        with pytest.raises(ValueError) as excinfo:
            assert m.ord_char8("ab")
        assert str(excinfo.value) == toolong_message


def test_bytes_to_string():
    """Tests the ability to pass bytes to C++ string-accepting functions.  Note that this is
    one-way: the only way to return bytes to Python is via the pybind11::bytes class."""
    # Issue #816

    assert m.strlen(b"hi") == 2
    assert m.string_length(b"world") == 5
    assert m.string_length(b"a\x00b") == 3
    assert m.strlen(b"a\x00b") == 1  # C-string limitation

    # passing in a utf8 encoded string should work
    assert m.string_length("💩".encode()) == 4


def test_bytearray_to_string():
    """Tests the ability to pass bytearray to C++ string-accepting functions"""
    assert m.string_length(bytearray(b"Hi")) == 2
    assert m.strlen(bytearray(b"bytearray")) == 9
    assert m.string_length(bytearray()) == 0
    assert m.string_length(bytearray("🦜", "utf-8", "strict")) == 4
    assert m.string_length(bytearray(b"\x80")) == 1


@pytest.mark.skipif(not hasattr(m, "has_string_view"), reason="no <string_view>")
def test_string_view(capture):
    """Tests support for C++17 string_view arguments and return values"""
    assert m.string_view_chars("Hi") == [72, 105]
    assert m.string_view_chars("Hi 🎂") == [72, 105, 32, 0xF0, 0x9F, 0x8E, 0x82]
    assert m.string_view16_chars("Hi 🎂") == [72, 105, 32, 0xD83C, 0xDF82]
    assert m.string_view32_chars("Hi 🎂") == [72, 105, 32, 127874]
    if hasattr(m, "has_u8string"):
        assert m.string_view8_chars("Hi") == [72, 105]
        assert m.string_view8_chars("Hi 🎂") == [72, 105, 32, 0xF0, 0x9F, 0x8E, 0x82]

    assert m.string_view_return() == "utf8 secret 🎂"
    assert m.string_view16_return() == "utf16 secret 🎂"
    assert m.string_view32_return() == "utf32 secret 🎂"
    if hasattr(m, "has_u8string"):
        assert m.string_view8_return() == "utf8 secret 🎂"

    with capture:
        m.string_view_print("Hi")
        m.string_view_print("utf8 🎂")
        m.string_view16_print("utf16 🎂")
        m.string_view32_print("utf32 🎂")
    assert (
        capture
        == """
        Hi 2
        utf8 🎂 9
        utf16 🎂 8
        utf32 🎂 7
    """
    )
    if hasattr(m, "has_u8string"):
        with capture:
            m.string_view8_print("Hi")
            m.string_view8_print("utf8 🎂")
        assert (
            capture
            == """
            Hi 2
            utf8 🎂 9
        """
        )

    with capture:
        m.string_view_print("Hi, ascii")
        m.string_view_print("Hi, utf8 🎂")
        m.string_view16_print("Hi, utf16 🎂")
        m.string_view32_print("Hi, utf32 🎂")
    assert (
        capture
        == """
        Hi, ascii 9
        Hi, utf8 🎂 13
        Hi, utf16 🎂 12
        Hi, utf32 🎂 11
    """
    )
    if hasattr(m, "has_u8string"):
        with capture:
            m.string_view8_print("Hi, ascii")
            m.string_view8_print("Hi, utf8 🎂")
        assert (
            capture
            == """
            Hi, ascii 9
            Hi, utf8 🎂 13
        """
        )

    assert m.string_view_bytes() == b"abc \x80\x80 def"
    assert m.string_view_str() == "abc ‽ def"
    assert m.string_view_from_bytes("abc ‽ def".encode()) == "abc ‽ def"
    if hasattr(m, "has_u8string"):
        assert m.string_view8_str() == "abc ‽ def"
    assert m.string_view_memoryview() == "Have some 🎂".encode()

    assert m.bytes_from_type_with_both_operator_string_and_string_view() == b"success"
    assert m.str_from_type_with_both_operator_string_and_string_view() == "success"


def test_integer_casting():
    """Issue #929 - out-of-range integer values shouldn't be accepted"""
    assert m.i32_str(-1) == "-1"
    assert m.i64_str(-1) == "-1"
    assert m.i32_str(2000000000) == "2000000000"
    assert m.u32_str(2000000000) == "2000000000"
    assert m.i64_str(-999999999999) == "-999999999999"
    assert m.u64_str(999999999999) == "999999999999"

    with pytest.raises(TypeError) as excinfo:
        m.u32_str(-1)
    assert "incompatible function arguments" in str(excinfo.value)
    with pytest.raises(TypeError) as excinfo:
        m.u64_str(-1)
    assert "incompatible function arguments" in str(excinfo.value)
    with pytest.raises(TypeError) as excinfo:
        m.i32_str(-3000000000)
    assert "incompatible function arguments" in str(excinfo.value)
    with pytest.raises(TypeError) as excinfo:
        m.i32_str(3000000000)
    assert "incompatible function arguments" in str(excinfo.value)


def test_int_convert():
    class Int:
        def __int__(self):
            return 42

    class NotInt:
        pass

    class Float:
        def __float__(self):
            return 41.99999

    class Index:
        def __index__(self):
            return 42

    class IntAndIndex:
        def __int__(self):
            return 42

        def __index__(self):
            return 0

    class RaisingTypeErrorOnIndex:
        def __index__(self):
            raise TypeError

        def __int__(self):
            return 42

    class RaisingValueErrorOnIndex:
        def __index__(self):
            raise ValueError

        def __int__(self):
            return 42

    convert, noconvert = m.int_passthrough, m.int_passthrough_noconvert

    def requires_conversion(v):
        pytest.raises(TypeError, noconvert, v)

    def cant_convert(v):
        pytest.raises(TypeError, convert, v)

    assert convert(7) == 7
    assert noconvert(7) == 7
    cant_convert(3.14159)
    # TODO: Avoid DeprecationWarning in `PyLong_AsLong` (and similar)
    # TODO: PyPy 3.8 does not behave like CPython 3.8 here yet (7.3.7)
    if (3, 8) <= sys.version_info < (3, 10) and env.CPYTHON:
        with env.deprecated_call():
            assert convert(Int()) == 42
    else:
        assert convert(Int()) == 42
    requires_conversion(Int())
    cant_convert(NotInt())
    cant_convert(Float())

    # Before Python 3.8, `PyLong_AsLong` does not pick up on `obj.__index__`,
    # but pybind11 "backports" this behavior.
    assert convert(Index()) == 42
    assert noconvert(Index()) == 42
    assert convert(IntAndIndex()) == 0  # Fishy; `int(DoubleThought)` == 42
    assert noconvert(IntAndIndex()) == 0
    assert convert(RaisingTypeErrorOnIndex()) == 42
    requires_conversion(RaisingTypeErrorOnIndex())
    assert convert(RaisingValueErrorOnIndex()) == 42
    requires_conversion(RaisingValueErrorOnIndex())


def test_numpy_int_convert():
    np = pytest.importorskip("numpy")

    convert, noconvert = m.int_passthrough, m.int_passthrough_noconvert

    def require_implicit(v):
        pytest.raises(TypeError, noconvert, v)

    # `np.intc` is an alias that corresponds to a C++ `int`
    assert convert(np.intc(42)) == 42
    assert noconvert(np.intc(42)) == 42

    # The implicit conversion from np.float32 is undesirable but currently accepted.
    # TODO: Avoid DeprecationWarning in `PyLong_AsLong` (and similar)
    # TODO: PyPy 3.8 does not behave like CPython 3.8 here yet (7.3.7)
    # https://github.com/pybind/pybind11/issues/3408
    if (3, 8) <= sys.version_info < (3, 10) and env.CPYTHON:
        with env.deprecated_call():
            assert convert(np.float32(3.14159)) == 3
    else:
        assert convert(np.float32(3.14159)) == 3
    require_implicit(np.float32(3.14159))


def test_tuple(doc):
    """std::pair <-> tuple & std::tuple <-> tuple"""
    assert m.pair_passthrough((True, "test")) == ("test", True)
    assert m.tuple_passthrough((True, "test", 5)) == (5, "test", True)
    # Any sequence can be cast to a std::pair or std::tuple
    assert m.pair_passthrough([True, "test"]) == ("test", True)
    assert m.tuple_passthrough([True, "test", 5]) == (5, "test", True)
    assert m.empty_tuple() == ()

    assert (
        doc(m.pair_passthrough)
        == """
        pair_passthrough(arg0: tuple[bool, str]) -> tuple[str, bool]

        Return a pair in reversed order
    """
    )
    assert (
        doc(m.tuple_passthrough)
        == """
        tuple_passthrough(arg0: tuple[bool, str, int]) -> tuple[int, str, bool]

        Return a triple in reversed order
    """
    )

    assert doc(m.empty_tuple) == """empty_tuple() -> tuple[()]"""

    assert m.rvalue_pair() == ("rvalue", "rvalue")
    assert m.lvalue_pair() == ("lvalue", "lvalue")
    assert m.rvalue_tuple() == ("rvalue", "rvalue", "rvalue")
    assert m.lvalue_tuple() == ("lvalue", "lvalue", "lvalue")
    assert m.rvalue_nested() == ("rvalue", ("rvalue", ("rvalue", "rvalue")))
    assert m.lvalue_nested() == ("lvalue", ("lvalue", ("lvalue", "lvalue")))

    assert m.int_string_pair() == (2, "items")


def test_builtins_cast_return_none():
    """Casters produced with PYBIND11_TYPE_CASTER() should convert nullptr to None"""
    assert m.return_none_string() is None
    assert m.return_none_char() is None
    assert m.return_none_bool() is None
    assert m.return_none_int() is None
    assert m.return_none_float() is None
    assert m.return_none_pair() is None


def test_none_deferred():
    """None passed as various argument types should defer to other overloads"""
    assert not m.defer_none_cstring("abc")
    assert m.defer_none_cstring(None)
    assert not m.defer_none_custom(UserType())
    assert m.defer_none_custom(None)
    assert m.nodefer_none_void(None)


def test_void_caster():
    assert m.load_nullptr_t(None) is None
    assert m.cast_nullptr_t() is None


def test_reference_wrapper():
    """std::reference_wrapper for builtin and user types"""
    assert m.refwrap_builtin(42) == 420
    assert m.refwrap_usertype(UserType(42)) == 42
    assert m.refwrap_usertype_const(UserType(42)) == 42

    with pytest.raises(TypeError) as excinfo:
        m.refwrap_builtin(None)
    assert "incompatible function arguments" in str(excinfo.value)

    with pytest.raises(TypeError) as excinfo:
        m.refwrap_usertype(None)
    assert "incompatible function arguments" in str(excinfo.value)

    assert m.refwrap_lvalue().value == 1
    assert m.refwrap_lvalue_const().value == 1

    a1 = m.refwrap_list(copy=True)
    a2 = m.refwrap_list(copy=True)
    assert [x.value for x in a1] == [2, 3]
    assert [x.value for x in a2] == [2, 3]
    assert a1[0] is not a2[0]
    assert a1[1] is not a2[1]

    b1 = m.refwrap_list(copy=False)
    b2 = m.refwrap_list(copy=False)
    assert [x.value for x in b1] == [1, 2]
    assert [x.value for x in b2] == [1, 2]
    assert b1[0] is b2[0]
    assert b1[1] is b2[1]

    assert m.refwrap_iiw(IncType(5)) == 5
    assert m.refwrap_call_iiw(IncType(10), m.refwrap_iiw) == [10, 10, 10, 10]


def test_complex_cast():
    """std::complex casts"""
    assert m.complex_cast(1) == "1.0"
    assert m.complex_cast(2j) == "(0.0, 2.0)"


def test_bool_caster():
    """Test bool caster implicit conversions."""
    convert, noconvert = m.bool_passthrough, m.bool_passthrough_noconvert

    def require_implicit(v):
        pytest.raises(TypeError, noconvert, v)

    def cant_convert(v):
        pytest.raises(TypeError, convert, v)

    # straight up bool
    assert convert(True) is True
    assert convert(False) is False
    assert noconvert(True) is True
    assert noconvert(False) is False

    # None requires implicit conversion
    require_implicit(None)
    assert convert(None) is False

    class A:
        def __init__(self, x):
            self.x = x

        def __nonzero__(self):
            return self.x

        def __bool__(self):
            return self.x

    class B:
        pass

    # Arbitrary objects are not accepted
    cant_convert(object())
    cant_convert(B())

    # Objects with __nonzero__ / __bool__ defined can be converted
    require_implicit(A(True))
    assert convert(A(True)) is True
    assert convert(A(False)) is False


def test_numpy_bool():
    np = pytest.importorskip("numpy")

    convert, noconvert = m.bool_passthrough, m.bool_passthrough_noconvert

    def cant_convert(v):
        pytest.raises(TypeError, convert, v)

    # np.bool_ is not considered implicit
    assert convert(np.bool_(True)) is True
    assert convert(np.bool_(False)) is False
    assert noconvert(np.bool_(True)) is True
    assert noconvert(np.bool_(False)) is False
    cant_convert(np.zeros(2, dtype="int"))


def test_int_long():
    assert isinstance(m.int_cast(), int)
    assert isinstance(m.long_cast(), int)
    assert isinstance(m.longlong_cast(), int)


def test_void_caster_2():
    assert m.test_void_caster()


def test_const_ref_caster():
    """Verifies that const-ref is propagated through type_caster cast_op.
    The returned ConstRefCasted type is a minimal type that is constructed to
    reference the casting mode used.
    """
    x = False
    assert m.takes(x) == 1
    assert m.takes_move(x) == 1

    assert m.takes_ptr(x) == 3
    assert m.takes_ref(x) == 2
    assert m.takes_ref_wrap(x) == 2

    assert m.takes_const_ptr(x) == 5
    assert m.takes_const_ref(x) == 4
    assert m.takes_const_ref_wrap(x) == 4
