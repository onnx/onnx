from __future__ import annotations

import sys
import time
from threading import Thread

import pytest

import env  # noqa: F401
from pybind11_tests import callbacks as m
from pybind11_tests import detailed_error_messages_enabled


def test_callbacks():
    from functools import partial

    def func1():
        return "func1"

    def func2(a, b, c, d):
        return "func2", a, b, c, d

    def func3(a):
        return f"func3({a})"

    assert m.test_callback1(func1) == "func1"
    assert m.test_callback2(func2) == ("func2", "Hello", "x", True, 5)
    assert m.test_callback1(partial(func2, 1, 2, 3, 4)) == ("func2", 1, 2, 3, 4)
    assert m.test_callback1(partial(func3, "partial")) == "func3(partial)"
    assert m.test_callback3(lambda i: i + 1) == "func(43) = 44"

    f = m.test_callback4()
    assert f(43) == 44
    f = m.test_callback5()
    assert f(number=43) == 44


def test_bound_method_callback():
    # Bound Python method:
    class MyClass:
        def double(self, val):
            return 2 * val

    z = MyClass()
    assert m.test_callback3(z.double) == "func(43) = 86"

    z = m.CppBoundMethodTest()
    assert m.test_callback3(z.triple) == "func(43) = 129"


def test_keyword_args_and_generalized_unpacking():
    def f(*args, **kwargs):
        return args, kwargs

    assert m.test_tuple_unpacking(f) == (("positional", 1, 2, 3, 4, 5, 6), {})
    assert m.test_dict_unpacking(f) == (
        ("positional", 1),
        {"key": "value", "a": 1, "b": 2},
    )
    assert m.test_keyword_args(f) == ((), {"x": 10, "y": 20})
    assert m.test_unpacking_and_keywords1(f) == ((1, 2), {"c": 3, "d": 4})
    assert m.test_unpacking_and_keywords2(f) == (
        ("positional", 1, 2, 3, 4, 5),
        {"key": "value", "a": 1, "b": 2, "c": 3, "d": 4, "e": 5},
    )

    with pytest.raises(TypeError) as excinfo:
        m.test_unpacking_error1(f)
    assert "Got multiple values for keyword argument" in str(excinfo.value)

    with pytest.raises(TypeError) as excinfo:
        m.test_unpacking_error2(f)
    assert "Got multiple values for keyword argument" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        m.test_arg_conversion_error1(f)
    assert str(excinfo.value) == "Unable to convert call argument " + (
        "'1' of type 'UnregisteredType' to Python object"
        if detailed_error_messages_enabled
        else "'1' to Python object (#define PYBIND11_DETAILED_ERROR_MESSAGES or compile in debug mode for details)"
    )

    with pytest.raises(RuntimeError) as excinfo:
        m.test_arg_conversion_error2(f)
    assert str(excinfo.value) == "Unable to convert call argument " + (
        "'expected_name' of type 'UnregisteredType' to Python object"
        if detailed_error_messages_enabled
        else "'expected_name' to Python object "
        "(#define PYBIND11_DETAILED_ERROR_MESSAGES or compile in debug mode for details)"
    )


def test_lambda_closure_cleanup():
    m.test_lambda_closure_cleanup()
    cstats = m.payload_cstats()
    assert cstats.alive() == 0
    assert cstats.copy_constructions == 1
    assert cstats.move_constructions >= 1


def test_cpp_callable_cleanup():
    alive_counts = m.test_cpp_callable_cleanup()
    assert alive_counts == [0, 1, 2, 1, 2, 1, 0]


def test_cpp_function_roundtrip():
    """Test if passing a function pointer from C++ -> Python -> C++ yields the original pointer"""

    assert (
        m.test_dummy_function(m.dummy_function) == "matches dummy_function: eval(1) = 2"
    )
    assert (
        m.test_dummy_function(m.roundtrip(m.dummy_function))
        == "matches dummy_function: eval(1) = 2"
    )
    assert (
        m.test_dummy_function(m.dummy_function_overloaded)
        == "matches dummy_function: eval(1) = 2"
    )
    assert m.roundtrip(None, expect_none=True) is None
    assert (
        m.test_dummy_function(lambda x: x + 2)
        == "can't convert to function pointer: eval(1) = 3"
    )

    with pytest.raises(TypeError) as excinfo:
        m.test_dummy_function(m.dummy_function2)
    assert "incompatible function arguments" in str(excinfo.value)

    with pytest.raises(TypeError) as excinfo:
        m.test_dummy_function(lambda x, y: x + y)
    assert any(
        s in str(excinfo.value)
        for s in ("missing 1 required positional argument", "takes exactly 2 arguments")
    )


def test_function_signatures(doc):
    assert doc(m.test_callback3) == "test_callback3(arg0: Callable[[int], int]) -> str"
    assert doc(m.test_callback4) == "test_callback4() -> Callable[[int], int]"


def test_movable_object():
    assert m.callback_with_movable(lambda _: None) is True


@pytest.mark.skipif(
    "env.PYPY",
    reason="PyPy segfaults on here. See discussion on #1413.",
)
def test_python_builtins():
    """Test if python builtins like sum() can be used as callbacks"""
    assert m.test_sum_builtin(sum, [1, 2, 3]) == 6
    assert m.test_sum_builtin(sum, []) == 0


@pytest.mark.skipif(sys.platform.startswith("emscripten"), reason="Requires threads")
def test_async_callbacks():
    # serves as state for async callback
    class Item:
        def __init__(self, value):
            self.value = value

    res = []

    # generate stateful lambda that will store result in `res`
    def gen_f():
        s = Item(3)
        return lambda j: res.append(s.value + j)

    # do some work async
    work = [1, 2, 3, 4]
    m.test_async_callback(gen_f(), work)
    # wait until work is done
    from time import sleep

    sleep(0.5)
    assert sum(res) == sum(x + 3 for x in work)


@pytest.mark.skipif(sys.platform.startswith("emscripten"), reason="Requires threads")
def test_async_async_callbacks():
    t = Thread(target=test_async_callbacks)
    t.start()
    t.join()


def test_callback_num_times():
    # Super-simple micro-benchmarking related to PR #2919.
    # Example runtimes (Intel Xeon 2.2GHz, fully optimized):
    #   num_millions  1, repeats  2:  0.1 secs
    #   num_millions 20, repeats 10: 11.5 secs
    one_million = 1000000
    num_millions = 1  # Try 20 for actual micro-benchmarking.
    repeats = 2  # Try 10.
    rates = []
    for rep in range(repeats):
        t0 = time.time()
        m.callback_num_times(lambda: None, num_millions * one_million)
        td = time.time() - t0
        rate = num_millions / td if td else 0
        rates.append(rate)
        if not rep:
            print()
        print(
            f"callback_num_times: {num_millions:d} million / {td:.3f} seconds = {rate:.3f} million / second"
        )
    if len(rates) > 1:
        print("Min    Mean   Max")
        print(f"{min(rates):6.3f} {sum(rates) / len(rates):6.3f} {max(rates):6.3f}")


def test_custom_func():
    assert m.custom_function(4) == 36
    assert m.roundtrip(m.custom_function)(4) == 36


@pytest.mark.skipif(
    m.custom_function2 is None, reason="Current PYBIND11_INTERNALS_VERSION too low"
)
def test_custom_func2():
    assert m.custom_function2(3) == 27
    assert m.roundtrip(m.custom_function2)(3) == 27


def test_callback_docstring():
    assert (
        m.test_tuple_unpacking.__doc__.strip()
        == "test_tuple_unpacking(arg0: Callable) -> object"
    )
