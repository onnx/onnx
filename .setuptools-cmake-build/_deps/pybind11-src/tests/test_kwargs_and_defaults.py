from __future__ import annotations

import pytest

from pybind11_tests import kwargs_and_defaults as m


def test_function_signatures(doc):
    assert doc(m.kw_func0) == "kw_func0(arg0: int, arg1: int) -> str"
    assert doc(m.kw_func1) == "kw_func1(x: int, y: int) -> str"
    assert doc(m.kw_func2) == "kw_func2(x: int = 100, y: int = 200) -> str"
    assert doc(m.kw_func3) == "kw_func3(data: str = 'Hello world!') -> None"
    assert doc(m.kw_func4) == "kw_func4(myList: list[int] = [13, 17]) -> str"
    assert doc(m.kw_func_udl) == "kw_func_udl(x: int, y: int = 300) -> str"
    assert doc(m.kw_func_udl_z) == "kw_func_udl_z(x: int, y: int = 0) -> str"
    assert doc(m.args_function) == "args_function(*args) -> tuple"
    assert (
        doc(m.args_kwargs_function) == "args_kwargs_function(*args, **kwargs) -> tuple"
    )
    assert (
        doc(m.KWClass.foo0)
        == "foo0(self: m.kwargs_and_defaults.KWClass, arg0: int, arg1: float) -> None"
    )
    assert (
        doc(m.KWClass.foo1)
        == "foo1(self: m.kwargs_and_defaults.KWClass, x: int, y: float) -> None"
    )
    assert (
        doc(m.kw_lb_func0)
        == "kw_lb_func0(custom: m.kwargs_and_defaults.CustomRepr = array([[A, B], [C, D]])) -> None"
    )
    assert (
        doc(m.kw_lb_func1)
        == "kw_lb_func1(custom: m.kwargs_and_defaults.CustomRepr = array([[A, B], [C, D]])) -> None"
    )
    assert (
        doc(m.kw_lb_func2)
        == "kw_lb_func2(custom: m.kwargs_and_defaults.CustomRepr = array([[A, B], [C, D]])) -> None"
    )
    assert (
        doc(m.kw_lb_func3)
        == "kw_lb_func3(custom: m.kwargs_and_defaults.CustomRepr = array([[A, B], [C, D]])) -> None"
    )
    assert (
        doc(m.kw_lb_func4)
        == "kw_lb_func4(custom: m.kwargs_and_defaults.CustomRepr = array([[A, B], [C, D]])) -> None"
    )
    assert (
        doc(m.kw_lb_func5)
        == "kw_lb_func5(custom: m.kwargs_and_defaults.CustomRepr = array([[A, B], [C, D]])) -> None"
    )
    assert (
        doc(m.kw_lb_func6)
        == "kw_lb_func6(custom: m.kwargs_and_defaults.CustomRepr = ) -> None"
    )
    assert (
        doc(m.kw_lb_func7)
        == "kw_lb_func7(str_arg: str = 'First line.\\n  Second line.') -> None"
    )
    assert (
        doc(m.kw_lb_func8)
        == "kw_lb_func8(custom: m.kwargs_and_defaults.CustomRepr = ) -> None"
    )


def test_named_arguments():
    assert m.kw_func0(5, 10) == "x=5, y=10"

    assert m.kw_func1(5, 10) == "x=5, y=10"
    assert m.kw_func1(5, y=10) == "x=5, y=10"
    assert m.kw_func1(y=10, x=5) == "x=5, y=10"

    assert m.kw_func2() == "x=100, y=200"
    assert m.kw_func2(5) == "x=5, y=200"
    assert m.kw_func2(x=5) == "x=5, y=200"
    assert m.kw_func2(y=10) == "x=100, y=10"
    assert m.kw_func2(5, 10) == "x=5, y=10"
    assert m.kw_func2(x=5, y=10) == "x=5, y=10"

    with pytest.raises(TypeError) as excinfo:
        # noinspection PyArgumentList
        m.kw_func2(x=5, y=10, z=12)
    assert excinfo.match(
        r"(?s)^kw_func2\(\): incompatible.*Invoked with: kwargs: ((x=5|y=10|z=12)(, |$)){3}$"
    )

    assert m.kw_func4() == "{13 17}"
    assert m.kw_func4(myList=[1, 2, 3]) == "{1 2 3}"

    assert m.kw_func_udl(x=5, y=10) == "x=5, y=10"
    assert m.kw_func_udl_z(x=5) == "x=5, y=0"


def test_arg_and_kwargs():
    args = "arg1_value", "arg2_value", 3
    assert m.args_function(*args) == args

    args = "a1", "a2"
    kwargs = {"arg3": "a3", "arg4": 4}
    assert m.args_kwargs_function(*args, **kwargs) == (args, kwargs)


def test_mixed_args_and_kwargs(msg):
    mpa = m.mixed_plus_args
    mpk = m.mixed_plus_kwargs
    mpak = m.mixed_plus_args_kwargs
    mpakd = m.mixed_plus_args_kwargs_defaults

    assert mpa(1, 2.5, 4, 99.5, None) == (1, 2.5, (4, 99.5, None))
    assert mpa(1, 2.5) == (1, 2.5, ())
    with pytest.raises(TypeError) as excinfo:
        assert mpa(1)
    assert (
        msg(excinfo.value)
        == """
        mixed_plus_args(): incompatible function arguments. The following argument types are supported:
            1. (arg0: int, arg1: float, *args) -> tuple

        Invoked with: 1
    """
    )
    with pytest.raises(TypeError) as excinfo:
        assert mpa()
    assert (
        msg(excinfo.value)
        == """
        mixed_plus_args(): incompatible function arguments. The following argument types are supported:
            1. (arg0: int, arg1: float, *args) -> tuple

        Invoked with:
    """
    )

    assert mpk(-2, 3.5, pi=3.14159, e=2.71828) == (
        -2,
        3.5,
        {"e": 2.71828, "pi": 3.14159},
    )
    assert mpak(7, 7.7, 7.77, 7.777, 7.7777, minusseven=-7) == (
        7,
        7.7,
        (7.77, 7.777, 7.7777),
        {"minusseven": -7},
    )
    assert mpakd() == (1, 3.14159, (), {})
    assert mpakd(3) == (3, 3.14159, (), {})
    assert mpakd(j=2.71828) == (1, 2.71828, (), {})
    assert mpakd(k=42) == (1, 3.14159, (), {"k": 42})
    assert mpakd(1, 1, 2, 3, 5, 8, then=13, followedby=21) == (
        1,
        1,
        (2, 3, 5, 8),
        {"then": 13, "followedby": 21},
    )
    # Arguments specified both positionally and via kwargs should fail:
    with pytest.raises(TypeError) as excinfo:
        assert mpakd(1, i=1)
    assert (
        msg(excinfo.value)
        == """
        mixed_plus_args_kwargs_defaults(): incompatible function arguments. The following argument types are supported:
            1. (i: int = 1, j: float = 3.14159, *args, **kwargs) -> tuple

        Invoked with: 1; kwargs: i=1
    """
    )
    with pytest.raises(TypeError) as excinfo:
        assert mpakd(1, 2, j=1)
    assert (
        msg(excinfo.value)
        == """
        mixed_plus_args_kwargs_defaults(): incompatible function arguments. The following argument types are supported:
            1. (i: int = 1, j: float = 3.14159, *args, **kwargs) -> tuple

        Invoked with: 1, 2; kwargs: j=1
    """
    )

    # Arguments after a py::args are automatically keyword-only (pybind 2.9+)
    assert m.args_kwonly(2, 2.5, z=22) == (2, 2.5, (), 22)
    assert m.args_kwonly(2, 2.5, "a", "b", "c", z=22) == (2, 2.5, ("a", "b", "c"), 22)
    assert m.args_kwonly(z=22, i=4, j=16) == (4, 16, (), 22)

    with pytest.raises(TypeError) as excinfo:
        assert m.args_kwonly(2, 2.5, 22)  # missing z= keyword
    assert (
        msg(excinfo.value)
        == """
        args_kwonly(): incompatible function arguments. The following argument types are supported:
            1. (i: int, j: float, *args, z: int) -> tuple

        Invoked with: 2, 2.5, 22
    """
    )

    assert m.args_kwonly_kwargs(i=1, k=4, j=10, z=-1, y=9) == (
        1,
        10,
        (),
        -1,
        {"k": 4, "y": 9},
    )
    assert m.args_kwonly_kwargs(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, z=11, y=12) == (
        1,
        2,
        (3, 4, 5, 6, 7, 8, 9, 10),
        11,
        {"y": 12},
    )
    assert (
        m.args_kwonly_kwargs.__doc__
        == "args_kwonly_kwargs(i: int, j: float, *args, z: int, **kwargs) -> tuple\n"
    )

    assert (
        m.args_kwonly_kwargs_defaults.__doc__
        == "args_kwonly_kwargs_defaults(i: int = 1, j: float = 3.14159, *args, z: int = 42, **kwargs) -> tuple\n"
    )
    assert m.args_kwonly_kwargs_defaults() == (1, 3.14159, (), 42, {})
    assert m.args_kwonly_kwargs_defaults(2) == (2, 3.14159, (), 42, {})
    assert m.args_kwonly_kwargs_defaults(z=-99) == (1, 3.14159, (), -99, {})
    assert m.args_kwonly_kwargs_defaults(5, 6, 7, 8) == (5, 6, (7, 8), 42, {})
    assert m.args_kwonly_kwargs_defaults(5, 6, 7, m=8) == (5, 6, (7,), 42, {"m": 8})
    assert m.args_kwonly_kwargs_defaults(5, 6, 7, m=8, z=9) == (5, 6, (7,), 9, {"m": 8})


def test_keyword_only_args(msg):
    assert m.kw_only_all(i=1, j=2) == (1, 2)
    assert m.kw_only_all(j=1, i=2) == (2, 1)

    with pytest.raises(TypeError) as excinfo:
        assert m.kw_only_all(i=1) == (1,)
    assert "incompatible function arguments" in str(excinfo.value)

    with pytest.raises(TypeError) as excinfo:
        assert m.kw_only_all(1, 2) == (1, 2)
    assert "incompatible function arguments" in str(excinfo.value)

    assert m.kw_only_some(1, k=3, j=2) == (1, 2, 3)

    assert m.kw_only_with_defaults(z=8) == (3, 4, 5, 8)
    assert m.kw_only_with_defaults(2, z=8) == (2, 4, 5, 8)
    assert m.kw_only_with_defaults(2, j=7, k=8, z=9) == (2, 7, 8, 9)
    assert m.kw_only_with_defaults(2, 7, z=9, k=8) == (2, 7, 8, 9)

    assert m.kw_only_mixed(1, j=2) == (1, 2)
    assert m.kw_only_mixed(j=2, i=3) == (3, 2)
    assert m.kw_only_mixed(i=2, j=3) == (2, 3)

    assert m.kw_only_plus_more(4, 5, k=6, extra=7) == (4, 5, 6, {"extra": 7})
    assert m.kw_only_plus_more(3, k=5, j=4, extra=6) == (3, 4, 5, {"extra": 6})
    assert m.kw_only_plus_more(2, k=3, extra=4) == (2, -1, 3, {"extra": 4})

    with pytest.raises(TypeError) as excinfo:
        assert m.kw_only_mixed(i=1) == (1,)
    assert "incompatible function arguments" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        m.register_invalid_kw_only(m)
    assert (
        msg(excinfo.value)
        == """
        arg(): cannot specify an unnamed argument after a kw_only() annotation or args() argument
    """
    )

    # https://github.com/pybind/pybind11/pull/3402#issuecomment-963341987
    x = m.first_arg_kw_only(i=1)
    x.method()
    x.method(i=1, j=2)
    assert (
        m.first_arg_kw_only.__init__.__doc__
        == "__init__(self: pybind11_tests.kwargs_and_defaults.first_arg_kw_only, *, i: int = 0) -> None\n"
    )
    assert (
        m.first_arg_kw_only.method.__doc__
        == "method(self: pybind11_tests.kwargs_and_defaults.first_arg_kw_only, *, i: int = 1, j: int = 2) -> None\n"
    )


def test_positional_only_args():
    assert m.pos_only_all(1, 2) == (1, 2)
    assert m.pos_only_all(2, 1) == (2, 1)

    with pytest.raises(TypeError) as excinfo:
        m.pos_only_all(i=1, j=2)
    assert "incompatible function arguments" in str(excinfo.value)

    assert m.pos_only_mix(1, 2) == (1, 2)
    assert m.pos_only_mix(2, j=1) == (2, 1)

    with pytest.raises(TypeError) as excinfo:
        m.pos_only_mix(i=1, j=2)
    assert "incompatible function arguments" in str(excinfo.value)

    assert m.pos_kw_only_mix(1, 2, k=3) == (1, 2, 3)
    assert m.pos_kw_only_mix(1, j=2, k=3) == (1, 2, 3)

    with pytest.raises(TypeError) as excinfo:
        m.pos_kw_only_mix(i=1, j=2, k=3)
    assert "incompatible function arguments" in str(excinfo.value)

    with pytest.raises(TypeError) as excinfo:
        m.pos_kw_only_mix(1, 2, 3)
    assert "incompatible function arguments" in str(excinfo.value)

    with pytest.raises(TypeError) as excinfo:
        m.pos_only_def_mix()
    assert "incompatible function arguments" in str(excinfo.value)

    assert m.pos_only_def_mix(1) == (1, 2, 3)
    assert m.pos_only_def_mix(1, 4) == (1, 4, 3)
    assert m.pos_only_def_mix(1, 4, 7) == (1, 4, 7)
    assert m.pos_only_def_mix(1, 4, k=7) == (1, 4, 7)

    with pytest.raises(TypeError) as excinfo:
        m.pos_only_def_mix(1, j=4)
    assert "incompatible function arguments" in str(excinfo.value)

    # Mix it with args and kwargs:
    assert (
        m.args_kwonly_full_monty.__doc__
        == "args_kwonly_full_monty(arg0: int = 1, arg1: int = 2, /, j: float = 3.14159, *args, z: int = 42, **kwargs) -> tuple\n"
    )
    assert m.args_kwonly_full_monty() == (1, 2, 3.14159, (), 42, {})
    assert m.args_kwonly_full_monty(8) == (8, 2, 3.14159, (), 42, {})
    assert m.args_kwonly_full_monty(8, 9) == (8, 9, 3.14159, (), 42, {})
    assert m.args_kwonly_full_monty(8, 9, 10) == (8, 9, 10.0, (), 42, {})
    assert m.args_kwonly_full_monty(3, 4, 5, 6, 7, m=8, z=9) == (
        3,
        4,
        5.0,
        (
            6,
            7,
        ),
        9,
        {"m": 8},
    )
    assert m.args_kwonly_full_monty(3, 4, 5, 6, 7, m=8, z=9) == (
        3,
        4,
        5.0,
        (
            6,
            7,
        ),
        9,
        {"m": 8},
    )
    assert m.args_kwonly_full_monty(5, j=7, m=8, z=9) == (5, 2, 7.0, (), 9, {"m": 8})
    assert m.args_kwonly_full_monty(i=5, j=7, m=8, z=9) == (
        1,
        2,
        7.0,
        (),
        9,
        {"i": 5, "m": 8},
    )

    # pos_only at the beginning of the argument list was "broken" in how it was displayed (though
    # this is fairly useless in practice).  Related to:
    # https://github.com/pybind/pybind11/pull/3402#issuecomment-963341987
    assert (
        m.first_arg_kw_only.pos_only.__doc__
        == "pos_only(self: pybind11_tests.kwargs_and_defaults.first_arg_kw_only, /, i: int, j: int) -> None\n"
    )


def test_signatures():
    assert m.kw_only_all.__doc__ == "kw_only_all(*, i: int, j: int) -> tuple\n"
    assert m.kw_only_mixed.__doc__ == "kw_only_mixed(i: int, *, j: int) -> tuple\n"
    assert m.pos_only_all.__doc__ == "pos_only_all(i: int, j: int, /) -> tuple\n"
    assert m.pos_only_mix.__doc__ == "pos_only_mix(i: int, /, j: int) -> tuple\n"
    assert (
        m.pos_kw_only_mix.__doc__
        == "pos_kw_only_mix(i: int, /, j: int, *, k: int) -> tuple\n"
    )


def test_args_refcount():
    """Issue/PR #1216 - py::args elements get double-inc_ref()ed when combined with regular
    arguments"""
    refcount = m.arg_refcount_h

    myval = object()
    expected = refcount(myval)
    assert m.arg_refcount_h(myval) == expected
    assert m.arg_refcount_o(myval) == expected + 1
    assert m.arg_refcount_h(myval) == expected
    assert refcount(myval) == expected

    assert m.mixed_plus_args(1, 2.0, "a", myval) == (1, 2.0, ("a", myval))
    assert refcount(myval) == expected

    assert m.mixed_plus_kwargs(3, 4.0, a=1, b=myval) == (3, 4.0, {"a": 1, "b": myval})
    assert refcount(myval) == expected

    assert m.args_function(-1, myval) == (-1, myval)
    assert refcount(myval) == expected

    assert m.mixed_plus_args_kwargs(5, 6.0, myval, a=myval) == (
        5,
        6.0,
        (myval,),
        {"a": myval},
    )
    assert refcount(myval) == expected

    assert m.args_kwargs_function(7, 8, myval, a=1, b=myval) == (
        (7, 8, myval),
        {"a": 1, "b": myval},
    )
    assert refcount(myval) == expected

    exp3 = refcount(myval, myval, myval)
    assert m.args_refcount(myval, myval, myval) == (exp3, exp3, exp3)
    assert refcount(myval) == expected

    # This function takes the first arg as a `py::object` and the rest as a `py::args`.  Unlike the
    # previous case, when we have both positional and `py::args` we need to construct a new tuple
    # for the `py::args`; in the previous case, we could simply inc_ref and pass on Python's input
    # tuple without having to inc_ref the individual elements, but here we can't, hence the extra
    # refs.
    exp3_3 = exp3 + 3
    assert m.mixed_args_refcount(myval, myval, myval) == (exp3_3, exp3_3, exp3_3)

    assert m.class_default_argument() == "<class 'decimal.Decimal'>"
