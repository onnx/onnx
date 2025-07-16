#include "pybind11_tests.h"

namespace {
struct any_struct {};
} // namespace

TEST_SUBMODULE(unnamed_namespace_b, m) {
    if (py::detail::get_type_info(typeid(any_struct)) == nullptr) {
        py::class_<any_struct>(m, "unnamed_namespace_b_any_struct");
    } else {
        m.attr("unnamed_namespace_b_any_struct") = py::none();
    }
}
