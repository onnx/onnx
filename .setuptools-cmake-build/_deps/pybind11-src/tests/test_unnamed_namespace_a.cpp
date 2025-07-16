#include "pybind11_tests.h"

namespace {
struct any_struct {};
} // namespace

TEST_SUBMODULE(unnamed_namespace_a, m) {
    if (py::detail::get_type_info(typeid(any_struct)) == nullptr) {
        py::class_<any_struct>(m, "unnamed_namespace_a_any_struct");
    } else {
        m.attr("unnamed_namespace_a_any_struct") = py::none();
    }
    m.attr("PYBIND11_INTERNALS_VERSION") = PYBIND11_INTERNALS_VERSION;
    m.attr("defined_WIN32_or__WIN32") =
#if defined(WIN32) || defined(_WIN32)
        true;
#else
        false;
#endif
    m.attr("defined___clang__") =
#if defined(__clang__)
        true;
#else
        false;
#endif
    m.attr("defined__LIBCPP_VERSION") =
#if defined(_LIBCPP_VERSION)
        true;
#else
        false;
#endif
    m.attr("defined___GLIBCXX__") =
#if defined(__GLIBCXX__)
        true;
#else
        false;
#endif
}
