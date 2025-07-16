// Copyright (c) 2024 The pybind Community.

#include "test_cpp_conduit_traveler_bindings.h"

namespace pybind11_tests {
namespace test_cpp_conduit {

PYBIND11_MODULE(home_planet_very_lonely_traveler, m) {
    m.def("wrap_very_lonely_traveler", [m]() { wrap_very_lonely_traveler(m); });
}

} // namespace test_cpp_conduit
} // namespace pybind11_tests
