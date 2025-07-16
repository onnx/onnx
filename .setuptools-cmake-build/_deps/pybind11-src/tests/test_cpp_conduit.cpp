// Copyright (c) 2024 The pybind Community.

#include "pybind11_tests.h"
#include "test_cpp_conduit_traveler_bindings.h"

#include <typeinfo>

namespace pybind11_tests {
namespace test_cpp_conduit {

TEST_SUBMODULE(cpp_conduit, m) {
    m.attr("PYBIND11_PLATFORM_ABI_ID") = py::bytes(PYBIND11_PLATFORM_ABI_ID);
    m.attr("cpp_type_info_capsule_Traveler")
        = py::capsule(&typeid(Traveler), typeid(std::type_info).name());
    m.attr("cpp_type_info_capsule_int") = py::capsule(&typeid(int), typeid(std::type_info).name());

    wrap_traveler(m);
    wrap_lonely_traveler(m);
}

} // namespace test_cpp_conduit
} // namespace pybind11_tests
