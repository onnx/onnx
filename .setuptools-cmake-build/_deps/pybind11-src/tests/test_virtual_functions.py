from __future__ import annotations

import sys

import pytest

import env  # noqa: F401

m = pytest.importorskip("pybind11_tests.virtual_functions")
from pybind11_tests import ConstructorStats  # noqa: E402


def test_override(capture, msg):
    class ExtendedExampleVirt(m.ExampleVirt):
        def __init__(self, state):
            super().__init__(state + 1)
            self.data = "Hello world"

        def run(self, value):
            print(f"ExtendedExampleVirt::run({value}), calling parent..")
            return super().run(value + 1)

        def run_bool(self):
            print("ExtendedExampleVirt::run_bool()")
            return False

        def get_string1(self):
            return "override1"

        def pure_virtual(self):
            print(f"ExtendedExampleVirt::pure_virtual(): {self.data}")

    class ExtendedExampleVirt2(ExtendedExampleVirt):
        def __init__(self, state):
            super().__init__(state + 1)

        def get_string2(self):
            return "override2"

    ex12 = m.ExampleVirt(10)
    with capture:
        assert m.runExampleVirt(ex12, 20) == 30
    assert (
        capture
        == """
        Original implementation of ExampleVirt::run(state=10, value=20, str1=default1, str2=default2)
    """
    )

    with pytest.raises(RuntimeError) as excinfo:
        m.runExampleVirtVirtual(ex12)
    assert (
        msg(excinfo.value)
        == 'Tried to call pure virtual function "ExampleVirt::pure_virtual"'
    )

    ex12p = ExtendedExampleVirt(10)
    with capture:
        assert m.runExampleVirt(ex12p, 20) == 32
    assert (
        capture
        == """
        ExtendedExampleVirt::run(20), calling parent..
        Original implementation of ExampleVirt::run(state=11, value=21, str1=override1, str2=default2)
    """
    )
    with capture:
        assert m.runExampleVirtBool(ex12p) is False
    assert capture == "ExtendedExampleVirt::run_bool()"
    with capture:
        m.runExampleVirtVirtual(ex12p)
    assert capture == "ExtendedExampleVirt::pure_virtual(): Hello world"

    ex12p2 = ExtendedExampleVirt2(15)
    with capture:
        assert m.runExampleVirt(ex12p2, 50) == 68
    assert (
        capture
        == """
        ExtendedExampleVirt::run(50), calling parent..
        Original implementation of ExampleVirt::run(state=17, value=51, str1=override1, str2=override2)
    """
    )

    cstats = ConstructorStats.get(m.ExampleVirt)
    assert cstats.alive() == 3
    del ex12, ex12p, ex12p2
    assert cstats.alive() == 0
    assert cstats.values() == ["10", "11", "17"]
    assert cstats.copy_constructions == 0
    assert cstats.move_constructions >= 0


def test_alias_delay_initialization1(capture):
    """`A` only initializes its trampoline class when we inherit from it

    If we just create and use an A instance directly, the trampoline initialization is
    bypassed and we only initialize an A() instead (for performance reasons).
    """

    class B(m.A):
        def __init__(self):
            super().__init__()

        def f(self):
            print("In python f()")

    # C++ version
    with capture:
        a = m.A()
        m.call_f(a)
        del a
        pytest.gc_collect()
    assert capture == "A.f()"

    # Python version
    with capture:
        b = B()
        m.call_f(b)
        del b
        pytest.gc_collect()
    assert (
        capture
        == """
        PyA.PyA()
        PyA.f()
        In python f()
        PyA.~PyA()
    """
    )


def test_alias_delay_initialization2(capture):
    """`A2`, unlike the above, is configured to always initialize the alias

    While the extra initialization and extra class layer has small virtual dispatch
    performance penalty, it also allows us to do more things with the trampoline
    class such as defining local variables and performing construction/destruction.
    """

    class B2(m.A2):
        def __init__(self):
            super().__init__()

        def f(self):
            print("In python B2.f()")

    # No python subclass version
    with capture:
        a2 = m.A2()
        m.call_f(a2)
        del a2
        pytest.gc_collect()
        a3 = m.A2(1)
        m.call_f(a3)
        del a3
        pytest.gc_collect()
    assert (
        capture
        == """
        PyA2.PyA2()
        PyA2.f()
        A2.f()
        PyA2.~PyA2()
        PyA2.PyA2()
        PyA2.f()
        A2.f()
        PyA2.~PyA2()
    """
    )

    # Python subclass version
    with capture:
        b2 = B2()
        m.call_f(b2)
        del b2
        pytest.gc_collect()
    assert (
        capture
        == """
        PyA2.PyA2()
        PyA2.f()
        In python B2.f()
        PyA2.~PyA2()
    """
    )


# PyPy: Reference count > 1 causes call with noncopyable instance
# to fail in ncv1.print_nc()
@pytest.mark.xfail("env.PYPY")
@pytest.mark.skipif(
    not hasattr(m, "NCVirt"), reason="NCVirt does not work on Intel/PGI/NVCC compilers"
)
def test_move_support():
    class NCVirtExt(m.NCVirt):
        def get_noncopyable(self, a, b):
            # Constructs and returns a new instance:
            return m.NonCopyable(a * a, b * b)

        def get_movable(self, a, b):
            # Return a referenced copy
            self.movable = m.Movable(a, b)
            return self.movable

    class NCVirtExt2(m.NCVirt):
        def get_noncopyable(self, a, b):
            # Keep a reference: this is going to throw an exception
            self.nc = m.NonCopyable(a, b)
            return self.nc

        def get_movable(self, a, b):
            # Return a new instance without storing it
            return m.Movable(a, b)

    ncv1 = NCVirtExt()
    assert ncv1.print_nc(2, 3) == "36"
    assert ncv1.print_movable(4, 5) == "9"
    ncv2 = NCVirtExt2()
    assert ncv2.print_movable(7, 7) == "14"
    # Don't check the exception message here because it differs under debug/non-debug mode
    with pytest.raises(RuntimeError):
        ncv2.print_nc(9, 9)

    nc_stats = ConstructorStats.get(m.NonCopyable)
    mv_stats = ConstructorStats.get(m.Movable)
    assert nc_stats.alive() == 1
    assert mv_stats.alive() == 1
    del ncv1, ncv2
    assert nc_stats.alive() == 0
    assert mv_stats.alive() == 0
    assert nc_stats.values() == ["4", "9", "9", "9"]
    assert mv_stats.values() == ["4", "5", "7", "7"]
    assert nc_stats.copy_constructions == 0
    assert mv_stats.copy_constructions == 1
    assert nc_stats.move_constructions >= 0
    assert mv_stats.move_constructions >= 0


def test_dispatch_issue(msg):
    """#159: virtual function dispatch has problems with similar-named functions"""

    class PyClass1(m.DispatchIssue):
        def dispatch(self):
            return "Yay.."

    class PyClass2(m.DispatchIssue):
        def dispatch(self):
            with pytest.raises(RuntimeError) as excinfo:
                super().dispatch()
            assert (
                msg(excinfo.value)
                == 'Tried to call pure virtual function "Base::dispatch"'
            )

            return m.dispatch_issue_go(PyClass1())

    b = PyClass2()
    assert m.dispatch_issue_go(b) == "Yay.."


def test_recursive_dispatch_issue():
    """#3357: Recursive dispatch fails to find python function override"""

    class Data(m.Data):
        def __init__(self, value):
            super().__init__()
            self.value = value

    class Adder(m.Adder):
        def __call__(self, first, second, visitor):
            # lambda is a workaround, which adds extra frame to the
            # current CPython thread. Removing lambda reveals the bug
            # [https://github.com/pybind/pybind11/issues/3357]
            (lambda: visitor(Data(first.value + second.value)))()  # noqa: PLC3002

    class StoreResultVisitor:
        def __init__(self):
            self.result = None

        def __call__(self, data):
            self.result = data.value

    store = StoreResultVisitor()

    m.add2(Data(1), Data(2), Adder(), store)
    assert store.result == 3

    # without lambda in Adder class, this function fails with
    # RuntimeError: Tried to call pure virtual function "AdderBase::__call__"
    m.add3(Data(1), Data(2), Data(3), Adder(), store)
    assert store.result == 6


def test_override_ref():
    """#392/397: overriding reference-returning functions"""
    o = m.OverrideTest("asdf")

    # Not allowed (see associated .cpp comment)
    # i = o.str_ref()
    # assert o.str_ref() == "asdf"
    assert o.str_value() == "asdf"

    assert o.A_value().value == "hi"
    a = o.A_ref()
    assert a.value == "hi"
    a.value = "bye"
    assert a.value == "bye"


def test_inherited_virtuals():
    class AR(m.A_Repeat):
        def unlucky_number(self):
            return 99

    class AT(m.A_Tpl):
        def unlucky_number(self):
            return 999

    obj = AR()
    assert obj.say_something(3) == "hihihi"
    assert obj.unlucky_number() == 99
    assert obj.say_everything() == "hi 99"

    obj = AT()
    assert obj.say_something(3) == "hihihi"
    assert obj.unlucky_number() == 999
    assert obj.say_everything() == "hi 999"

    for obj in [m.B_Repeat(), m.B_Tpl()]:
        assert obj.say_something(3) == "B says hi 3 times"
        assert obj.unlucky_number() == 13
        assert obj.lucky_number() == 7.0
        assert obj.say_everything() == "B says hi 1 times 13"

    for obj in [m.C_Repeat(), m.C_Tpl()]:
        assert obj.say_something(3) == "B says hi 3 times"
        assert obj.unlucky_number() == 4444
        assert obj.lucky_number() == 888.0
        assert obj.say_everything() == "B says hi 1 times 4444"

    class CR(m.C_Repeat):
        def lucky_number(self):
            return m.C_Repeat.lucky_number(self) + 1.25

    obj = CR()
    assert obj.say_something(3) == "B says hi 3 times"
    assert obj.unlucky_number() == 4444
    assert obj.lucky_number() == 889.25
    assert obj.say_everything() == "B says hi 1 times 4444"

    class CT(m.C_Tpl):
        pass

    obj = CT()
    assert obj.say_something(3) == "B says hi 3 times"
    assert obj.unlucky_number() == 4444
    assert obj.lucky_number() == 888.0
    assert obj.say_everything() == "B says hi 1 times 4444"

    class CCR(CR):
        def lucky_number(self):
            return CR.lucky_number(self) * 10

    obj = CCR()
    assert obj.say_something(3) == "B says hi 3 times"
    assert obj.unlucky_number() == 4444
    assert obj.lucky_number() == 8892.5
    assert obj.say_everything() == "B says hi 1 times 4444"

    class CCT(CT):
        def lucky_number(self):
            return CT.lucky_number(self) * 1000

    obj = CCT()
    assert obj.say_something(3) == "B says hi 3 times"
    assert obj.unlucky_number() == 4444
    assert obj.lucky_number() == 888000.0
    assert obj.say_everything() == "B says hi 1 times 4444"

    class DR(m.D_Repeat):
        def unlucky_number(self):
            return 123

        def lucky_number(self):
            return 42.0

    for obj in [m.D_Repeat(), m.D_Tpl()]:
        assert obj.say_something(3) == "B says hi 3 times"
        assert obj.unlucky_number() == 4444
        assert obj.lucky_number() == 888.0
        assert obj.say_everything() == "B says hi 1 times 4444"

    obj = DR()
    assert obj.say_something(3) == "B says hi 3 times"
    assert obj.unlucky_number() == 123
    assert obj.lucky_number() == 42.0
    assert obj.say_everything() == "B says hi 1 times 123"

    class DT(m.D_Tpl):
        def say_something(self, times):
            return "DT says:" + (" quack" * times)

        def unlucky_number(self):
            return 1234

        def lucky_number(self):
            return -4.25

    obj = DT()
    assert obj.say_something(3) == "DT says: quack quack quack"
    assert obj.unlucky_number() == 1234
    assert obj.lucky_number() == -4.25
    assert obj.say_everything() == "DT says: quack 1234"

    class DT2(DT):
        def say_something(self, times):
            return "DT2: " + ("QUACK" * times)

        def unlucky_number(self):
            return -3

    class BT(m.B_Tpl):
        def say_something(self, times):
            return "BT" * times

        def unlucky_number(self):
            return -7

        def lucky_number(self):
            return -1.375

    obj = BT()
    assert obj.say_something(3) == "BTBTBT"
    assert obj.unlucky_number() == -7
    assert obj.lucky_number() == -1.375
    assert obj.say_everything() == "BT -7"


@pytest.mark.skipif(sys.platform.startswith("emscripten"), reason="Requires threads")
def test_issue_1454():
    # Fix issue #1454 (crash when acquiring/releasing GIL on another thread in Python 2.7)
    m.test_gil()
    m.test_gil_from_thread()


def test_python_override():
    def func():
        class Test(m.test_override_cache_helper):
            def func(self):
                return 42

        return Test()

    def func2():
        class Test(m.test_override_cache_helper):
            pass

        return Test()

    for _ in range(1500):
        assert m.test_override_cache(func()) == 42
        assert m.test_override_cache(func2()) == 0
