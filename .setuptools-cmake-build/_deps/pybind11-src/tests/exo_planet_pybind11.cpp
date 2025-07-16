// Copyright (c) 2024 The pybind Community.

#if defined(PYBIND11_INTERNALS_VERSION)
#    undef PYBIND11_INTERNALS_VERSION
#endif
#define PYBIND11_INTERNALS_VERSION 900000001

#include "test_cpp_conduit_traveler_bindings.h"

namespace pybind11_tests {
namespace test_cpp_conduit {

PYBIND11_MODULE(exo_planet_pybind11, m) {
    wrap_traveler(m);
    m.def("wrap_very_lonely_traveler", [m]() { wrap_very_lonely_traveler(m); });
}

} // namespace test_cpp_conduit
} // namespace pybind11_tests
