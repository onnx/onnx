#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

#include "pybind11_tests.h"

namespace py = pybind11;

namespace {

struct SpecialReturn {
    int value = 99;
};

} // namespace

namespace pybind11 {
namespace detail {
namespace type_caster_std_function_specializations {

template <typename... Args>
struct func_wrapper<SpecialReturn, Args...> : func_wrapper_base {
    using func_wrapper_base::func_wrapper_base;
    SpecialReturn operator()(Args... args) const {
        gil_scoped_acquire acq;
        SpecialReturn result;
        try {
            result = hfunc.f(std::forward<Args>(args)...).template cast<SpecialReturn>();
        } catch (error_already_set &) {
            result.value += 1;
        }
        result.value += 100;
        return result;
    }
};

} // namespace type_caster_std_function_specializations
} // namespace detail
} // namespace pybind11

TEST_SUBMODULE(type_caster_std_function_specializations, m) {
    py::class_<SpecialReturn>(m, "SpecialReturn")
        .def(py::init<>())
        .def_readwrite("value", &SpecialReturn::value);
    m.def("call_callback_with_special_return",
          [](const std::function<SpecialReturn()> &func) { return func(); });
}
