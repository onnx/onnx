# ruff: noqa: SIM201 SIM300 SIM202
from __future__ import annotations

import pytest

from pybind11_tests import enums as m


def test_unscoped_enum():
    assert str(m.UnscopedEnum.EOne) == "UnscopedEnum.EOne"
    assert str(m.UnscopedEnum.ETwo) == "UnscopedEnum.ETwo"
    assert str(m.EOne) == "UnscopedEnum.EOne"
    assert repr(m.UnscopedEnum.EOne) == "<UnscopedEnum.EOne: 1>"
    assert repr(m.UnscopedEnum.ETwo) == "<UnscopedEnum.ETwo: 2>"
    assert repr(m.EOne) == "<UnscopedEnum.EOne: 1>"

    # name property
    assert m.UnscopedEnum.EOne.name == "EOne"
    assert m.UnscopedEnum.EOne.value == 1
    assert m.UnscopedEnum.ETwo.name == "ETwo"
    assert m.UnscopedEnum.ETwo.value == 2
    assert m.EOne is m.UnscopedEnum.EOne
    # name, value readonly
    with pytest.raises(AttributeError):
        m.UnscopedEnum.EOne.name = ""
    with pytest.raises(AttributeError):
        m.UnscopedEnum.EOne.value = 10
    # name, value returns a copy
    # TODO: Neither the name nor value tests actually check against aliasing.
    # Use a mutable type that has reference semantics.
    nonaliased_name = m.UnscopedEnum.EOne.name
    nonaliased_name = "bar"  # noqa: F841
    assert m.UnscopedEnum.EOne.name == "EOne"
    nonaliased_value = m.UnscopedEnum.EOne.value
    nonaliased_value = 10  # noqa: F841
    assert m.UnscopedEnum.EOne.value == 1

    # __members__ property
    assert m.UnscopedEnum.__members__ == {
        "EOne": m.UnscopedEnum.EOne,
        "ETwo": m.UnscopedEnum.ETwo,
        "EThree": m.UnscopedEnum.EThree,
    }
    # __members__ readonly
    with pytest.raises(AttributeError):
        m.UnscopedEnum.__members__ = {}
    # __members__ returns a copy
    nonaliased_members = m.UnscopedEnum.__members__
    nonaliased_members["bar"] = "baz"
    assert m.UnscopedEnum.__members__ == {
        "EOne": m.UnscopedEnum.EOne,
        "ETwo": m.UnscopedEnum.ETwo,
        "EThree": m.UnscopedEnum.EThree,
    }

    for docstring_line in """An unscoped enumeration

Members:

  EOne : Docstring for EOne

  ETwo : Docstring for ETwo

  EThree : Docstring for EThree""".split("\n"):
        assert docstring_line in m.UnscopedEnum.__doc__

    # Unscoped enums will accept ==/!= int comparisons
    y = m.UnscopedEnum.ETwo
    assert y == 2
    assert 2 == y
    assert y != 3
    assert 3 != y
    # Compare with None
    assert y != None  # noqa: E711
    assert not (y == None)  # noqa: E711
    # Compare with an object
    assert y != object()
    assert not (y == object())
    # Compare with string
    assert y != "2"
    assert "2" != y
    assert not ("2" == y)
    assert not (y == "2")

    with pytest.raises(TypeError):
        y < object()  # noqa: B015

    with pytest.raises(TypeError):
        y <= object()  # noqa: B015

    with pytest.raises(TypeError):
        y > object()  # noqa: B015

    with pytest.raises(TypeError):
        y >= object()  # noqa: B015

    with pytest.raises(TypeError):
        y | object()

    with pytest.raises(TypeError):
        y & object()

    with pytest.raises(TypeError):
        y ^ object()

    assert int(m.UnscopedEnum.ETwo) == 2
    assert str(m.UnscopedEnum(2)) == "UnscopedEnum.ETwo"

    # order
    assert m.UnscopedEnum.EOne < m.UnscopedEnum.ETwo
    assert m.UnscopedEnum.EOne < 2
    assert m.UnscopedEnum.ETwo > m.UnscopedEnum.EOne
    assert m.UnscopedEnum.ETwo > 1
    assert m.UnscopedEnum.ETwo <= 2
    assert m.UnscopedEnum.ETwo >= 2
    assert m.UnscopedEnum.EOne <= m.UnscopedEnum.ETwo
    assert m.UnscopedEnum.EOne <= 2
    assert m.UnscopedEnum.ETwo >= m.UnscopedEnum.EOne
    assert m.UnscopedEnum.ETwo >= 1
    assert not (m.UnscopedEnum.ETwo < m.UnscopedEnum.EOne)
    assert not (2 < m.UnscopedEnum.EOne)

    # arithmetic
    assert m.UnscopedEnum.EOne & m.UnscopedEnum.EThree == m.UnscopedEnum.EOne
    assert m.UnscopedEnum.EOne | m.UnscopedEnum.ETwo == m.UnscopedEnum.EThree
    assert m.UnscopedEnum.EOne ^ m.UnscopedEnum.EThree == m.UnscopedEnum.ETwo


def test_scoped_enum():
    assert m.test_scoped_enum(m.ScopedEnum.Three) == "ScopedEnum::Three"
    z = m.ScopedEnum.Two
    assert m.test_scoped_enum(z) == "ScopedEnum::Two"

    # Scoped enums will *NOT* accept ==/!= int comparisons (Will always return False)
    assert not z == 3
    assert not 3 == z
    assert z != 3
    assert 3 != z
    # Compare with None
    assert z != None  # noqa: E711
    assert not (z == None)  # noqa: E711
    # Compare with an object
    assert z != object()
    assert not (z == object())
    # Scoped enums will *NOT* accept >, <, >= and <= int comparisons (Will throw exceptions)
    with pytest.raises(TypeError):
        z > 3  # noqa: B015
    with pytest.raises(TypeError):
        z < 3  # noqa: B015
    with pytest.raises(TypeError):
        z >= 3  # noqa: B015
    with pytest.raises(TypeError):
        z <= 3  # noqa: B015

    # order
    assert m.ScopedEnum.Two < m.ScopedEnum.Three
    assert m.ScopedEnum.Three > m.ScopedEnum.Two
    assert m.ScopedEnum.Two <= m.ScopedEnum.Three
    assert m.ScopedEnum.Two <= m.ScopedEnum.Two
    assert m.ScopedEnum.Two >= m.ScopedEnum.Two
    assert m.ScopedEnum.Three >= m.ScopedEnum.Two


def test_implicit_conversion():
    assert str(m.ClassWithUnscopedEnum.EMode.EFirstMode) == "EMode.EFirstMode"
    assert str(m.ClassWithUnscopedEnum.EFirstMode) == "EMode.EFirstMode"
    assert repr(m.ClassWithUnscopedEnum.EMode.EFirstMode) == "<EMode.EFirstMode: 1>"
    assert repr(m.ClassWithUnscopedEnum.EFirstMode) == "<EMode.EFirstMode: 1>"

    f = m.ClassWithUnscopedEnum.test_function
    first = m.ClassWithUnscopedEnum.EFirstMode
    second = m.ClassWithUnscopedEnum.ESecondMode

    assert f(first) == 1

    assert f(first) == f(first)
    assert not f(first) != f(first)

    assert f(first) != f(second)
    assert not f(first) == f(second)

    assert f(first) == int(f(first))
    assert not f(first) != int(f(first))

    assert f(first) != int(f(second))
    assert not f(first) == int(f(second))

    # noinspection PyDictCreation
    x = {f(first): 1, f(second): 2}
    x[f(first)] = 3
    x[f(second)] = 4
    # Hashing test
    assert repr(x) == "{<EMode.EFirstMode: 1>: 3, <EMode.ESecondMode: 2>: 4}"


def test_binary_operators():
    assert int(m.Flags.Read) == 4
    assert int(m.Flags.Write) == 2
    assert int(m.Flags.Execute) == 1
    assert int(m.Flags.Read | m.Flags.Write | m.Flags.Execute) == 7
    assert int(m.Flags.Read | m.Flags.Write) == 6
    assert int(m.Flags.Read | m.Flags.Execute) == 5
    assert int(m.Flags.Write | m.Flags.Execute) == 3
    assert int(m.Flags.Write | 1) == 3
    assert ~m.Flags.Write == -3

    state = m.Flags.Read | m.Flags.Write
    assert (state & m.Flags.Read) != 0
    assert (state & m.Flags.Write) != 0
    assert (state & m.Flags.Execute) == 0
    assert (state & 1) == 0

    state2 = ~state
    assert state2 == -7
    assert int(state ^ state2) == -1


def test_enum_to_int():
    m.test_enum_to_int(m.Flags.Read)
    m.test_enum_to_int(m.ClassWithUnscopedEnum.EMode.EFirstMode)
    m.test_enum_to_int(m.ScopedCharEnum.Positive)
    m.test_enum_to_int(m.ScopedBoolEnum.TRUE)
    m.test_enum_to_uint(m.Flags.Read)
    m.test_enum_to_uint(m.ClassWithUnscopedEnum.EMode.EFirstMode)
    m.test_enum_to_uint(m.ScopedCharEnum.Positive)
    m.test_enum_to_uint(m.ScopedBoolEnum.TRUE)
    m.test_enum_to_long_long(m.Flags.Read)
    m.test_enum_to_long_long(m.ClassWithUnscopedEnum.EMode.EFirstMode)
    m.test_enum_to_long_long(m.ScopedCharEnum.Positive)
    m.test_enum_to_long_long(m.ScopedBoolEnum.TRUE)


def test_duplicate_enum_name():
    with pytest.raises(ValueError) as excinfo:
        m.register_bad_enum()
    assert str(excinfo.value) == 'SimpleEnum: element "ONE" already exists!'


def test_char_underlying_enum():  # Issue #1331/PR #1334:
    assert type(m.ScopedCharEnum.Positive.__int__()) is int
    assert int(m.ScopedChar16Enum.Zero) == 0
    assert hash(m.ScopedChar32Enum.Positive) == 1
    assert type(m.ScopedCharEnum.Positive.__getstate__()) is int
    assert m.ScopedWCharEnum(1) == m.ScopedWCharEnum.Positive
    with pytest.raises(TypeError):
        # Even if the underlying type is char, only an int can be used to construct the enum:
        m.ScopedCharEnum("0")


def test_bool_underlying_enum():
    assert type(m.ScopedBoolEnum.TRUE.__int__()) is int
    assert int(m.ScopedBoolEnum.FALSE) == 0
    assert hash(m.ScopedBoolEnum.TRUE) == 1
    assert type(m.ScopedBoolEnum.TRUE.__getstate__()) is int
    assert m.ScopedBoolEnum(1) == m.ScopedBoolEnum.TRUE
    # Enum could construct with a bool
    # (bool is a strict subclass of int, and False will be converted to 0)
    assert m.ScopedBoolEnum(False) == m.ScopedBoolEnum.FALSE


def test_docstring_signatures():
    for enum_type in [m.ScopedEnum, m.UnscopedEnum]:
        for attr in enum_type.__dict__.values():
            # Issue #2623/PR #2637: Add argument names to enum_ methods
            assert "arg0" not in (attr.__doc__ or "")


def test_str_signature():
    for enum_type in [m.ScopedEnum, m.UnscopedEnum]:
        assert enum_type.__str__.__doc__.startswith("__str__")
