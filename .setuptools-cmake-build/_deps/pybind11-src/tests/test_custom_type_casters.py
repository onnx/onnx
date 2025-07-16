from __future__ import annotations

import pytest

from pybind11_tests import custom_type_casters as m


def test_noconvert_args(msg):
    a = m.ArgInspector()
    assert (
        msg(a.f("hi"))
        == """
        loading ArgInspector1 argument WITH conversion allowed.  Argument value = hi
    """
    )
    assert (
        msg(a.g("this is a", "this is b"))
        == """
        loading ArgInspector1 argument WITHOUT conversion allowed.  Argument value = this is a
        loading ArgInspector1 argument WITH conversion allowed.  Argument value = this is b
        13
        loading ArgInspector2 argument WITH conversion allowed.  Argument value = (default arg inspector 2)
    """
    )
    assert (
        msg(a.g("this is a", "this is b", 42))
        == """
        loading ArgInspector1 argument WITHOUT conversion allowed.  Argument value = this is a
        loading ArgInspector1 argument WITH conversion allowed.  Argument value = this is b
        42
        loading ArgInspector2 argument WITH conversion allowed.  Argument value = (default arg inspector 2)
    """
    )
    assert (
        msg(a.g("this is a", "this is b", 42, "this is d"))
        == """
        loading ArgInspector1 argument WITHOUT conversion allowed.  Argument value = this is a
        loading ArgInspector1 argument WITH conversion allowed.  Argument value = this is b
        42
        loading ArgInspector2 argument WITH conversion allowed.  Argument value = this is d
    """
    )
    assert (
        a.h("arg 1")
        == "loading ArgInspector2 argument WITHOUT conversion allowed.  Argument value = arg 1"
    )
    assert (
        msg(m.arg_inspect_func("A1", "A2"))
        == """
        loading ArgInspector2 argument WITH conversion allowed.  Argument value = A1
        loading ArgInspector1 argument WITHOUT conversion allowed.  Argument value = A2
    """
    )

    assert m.floats_preferred(4) == 2.0
    assert m.floats_only(4.0) == 2.0
    with pytest.raises(TypeError) as excinfo:
        m.floats_only(4)
    assert (
        msg(excinfo.value)
        == """
        floats_only(): incompatible function arguments. The following argument types are supported:
            1. (f: float) -> float

        Invoked with: 4
    """
    )

    assert m.ints_preferred(4) == 2
    assert m.ints_preferred(True) == 0
    with pytest.raises(TypeError) as excinfo:
        m.ints_preferred(4.0)
    assert (
        msg(excinfo.value)
        == """
        ints_preferred(): incompatible function arguments. The following argument types are supported:
            1. (i: int) -> int

        Invoked with: 4.0
    """
    )

    assert m.ints_only(4) == 2
    with pytest.raises(TypeError) as excinfo:
        m.ints_only(4.0)
    assert (
        msg(excinfo.value)
        == """
        ints_only(): incompatible function arguments. The following argument types are supported:
            1. (i: int) -> int

        Invoked with: 4.0
    """
    )


def test_custom_caster_destruction():
    """Tests that returning a pointer to a type that gets converted with a custom type caster gets
    destroyed when the function has py::return_value_policy::take_ownership policy applied.
    """

    cstats = m.destruction_tester_cstats()
    # This one *doesn't* have take_ownership: the pointer should be used but not destroyed:
    z = m.custom_caster_no_destroy()
    assert cstats.alive() == 1
    assert cstats.default_constructions == 1
    assert z

    # take_ownership applied: this constructs a new object, casts it, then destroys it:
    z = m.custom_caster_destroy()
    assert z
    assert cstats.default_constructions == 2

    # Same, but with a const pointer return (which should *not* inhibit destruction):
    z = m.custom_caster_destroy_const()
    assert z
    assert cstats.default_constructions == 3

    # Make sure we still only have the original object (from ..._no_destroy()) alive:
    assert cstats.alive() == 1


def test_custom_caster_other_lib():
    assert m.other_lib_type(True)
