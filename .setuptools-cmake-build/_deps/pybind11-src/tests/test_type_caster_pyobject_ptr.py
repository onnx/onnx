from __future__ import annotations

import pytest

from pybind11_tests import type_caster_pyobject_ptr as m


# For use as a temporary user-defined object, to maximize sensitivity of the tests below.
class ValueHolder:
    def __init__(self, value):
        self.value = value


def test_cast_from_pyobject_ptr():
    assert m.cast_from_pyobject_ptr() == 6758


def test_cast_handle_to_pyobject_ptr():
    assert m.cast_handle_to_pyobject_ptr(ValueHolder(24)) == 76


def test_cast_object_to_pyobject_ptr():
    assert m.cast_object_to_pyobject_ptr(ValueHolder(43)) == 257


def test_cast_list_to_pyobject_ptr():
    assert m.cast_list_to_pyobject_ptr([1, 2, 3, 4, 5]) == 395


def test_return_pyobject_ptr():
    assert m.return_pyobject_ptr() == 2314


def test_pass_pyobject_ptr():
    assert m.pass_pyobject_ptr(ValueHolder(82)) == 118


@pytest.mark.parametrize(
    "call_callback",
    [
        m.call_callback_with_object_return,
        m.call_callback_with_pyobject_ptr_return,
    ],
)
def test_call_callback_with_object_return(call_callback):
    def cb(value):
        if value < 0:
            raise ValueError("Raised from cb")
        return ValueHolder(1000 - value)

    assert call_callback(cb, 287).value == 713

    with pytest.raises(ValueError, match="^Raised from cb$"):
        call_callback(cb, -1)


def test_call_callback_with_pyobject_ptr_arg():
    def cb(obj):
        return 300 - obj.value

    assert m.call_callback_with_pyobject_ptr_arg(cb, ValueHolder(39)) == 261


@pytest.mark.parametrize("set_error", [True, False])
def test_cast_to_python_nullptr(set_error):
    expected = {
        True: r"^Reflective of healthy error handling\.$",
        False: (
            r"^Internal error: pybind11::error_already_set called "
            r"while Python error indicator not set\.$"
        ),
    }[set_error]
    with pytest.raises(RuntimeError, match=expected):
        m.cast_to_pyobject_ptr_nullptr(set_error)


def test_cast_to_python_non_nullptr_with_error_set():
    with pytest.raises(SystemError) as excinfo:
        m.cast_to_pyobject_ptr_non_nullptr_with_error_set()
    assert str(excinfo.value) == "src != nullptr but PyErr_Occurred()"
    assert str(excinfo.value.__cause__) == "Reflective of unhealthy error handling."


def test_pass_list_pyobject_ptr():
    acc = m.pass_list_pyobject_ptr([ValueHolder(842), ValueHolder(452)])
    assert acc == 842452


def test_return_list_pyobject_ptr_take_ownership():
    vec_obj = m.return_list_pyobject_ptr_take_ownership(ValueHolder)
    assert [e.value for e in vec_obj] == [93, 186]


def test_return_list_pyobject_ptr_reference():
    vec_obj = m.return_list_pyobject_ptr_reference(ValueHolder)
    assert [e.value for e in vec_obj] == [93, 186]
    # Commenting out the next `assert` will leak the Python references.
    # An easy way to see evidence of the leaks:
    # Insert `while True:` as the first line of this function and monitor the
    # process RES (Resident Memory Size) with the Unix top command.
    assert m.dec_ref_each_pyobject_ptr(vec_obj) == 2


def test_type_caster_name_via_incompatible_function_arguments_type_error():
    with pytest.raises(TypeError, match=r"1\. \(arg0: object, arg1: int\) -> None"):
        m.pass_pyobject_ptr_and_int(ValueHolder(101), ValueHolder(202))


def test_trampoline_with_pyobject_ptr_return():
    class Drvd(m.WithPyObjectPtrReturn):
        def return_pyobject_ptr(self):
            return ["11", "22", "33"]

    # Basic health check: First make sure this works as expected.
    d = Drvd()
    assert d.return_pyobject_ptr() == ["11", "22", "33"]

    while True:
        # This failed before PR #5156: AddressSanitizer: heap-use-after-free ... in Py_DECREF
        d_repr = m.call_return_pyobject_ptr(d)
        assert d_repr == repr(["11", "22", "33"])
        break  # Comment out for manual leak checking.
