from __future__ import annotations

from pybind11_tests import type_caster_std_function_specializations as m


def test_callback_with_special_return():
    def return_special():
        return m.SpecialReturn()

    def raise_exception():
        raise ValueError("called raise_exception.")

    assert return_special().value == 99
    assert m.call_callback_with_special_return(return_special).value == 199
    assert m.call_callback_with_special_return(raise_exception).value == 200
