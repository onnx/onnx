// Copyright (c) 2024 The pybind Community.

#pragma once

#include <pybind11/pybind11.h>

#include "test_cpp_conduit_traveler_types.h"

#include <string>

namespace pybind11_tests {
namespace test_cpp_conduit {

namespace py = pybind11;

inline void wrap_traveler(py::module_ m) {
    py::class_<Traveler>(m, "Traveler")
        .def(py::init<std::string>())
        .def_readwrite("luggage", &Traveler::luggage)
        // See issue #3788:
        .def("__getattr__", [](const Traveler &self, const std::string &key) {
            return "Traveler GetAttr: " + key + " luggage: " + self.luggage;
        });

    m.def("get_luggage", [](const Traveler &person) { return person.luggage; });

    py::class_<PremiumTraveler, Traveler>(m, "PremiumTraveler")
        .def(py::init<std::string, int>())
        .def_readwrite("points", &PremiumTraveler::points)
        // See issue #3788:
        .def("__getattr__", [](const PremiumTraveler &self, const std::string &key) {
            return "PremiumTraveler GetAttr: " + key + " points: " + std::to_string(self.points);
        });

    m.def("get_points", [](const PremiumTraveler &person) { return person.points; });
}

inline void wrap_lonely_traveler(py::module_ m) {
    py::class_<LonelyTraveler>(std::move(m), "LonelyTraveler");
}

inline void wrap_very_lonely_traveler(py::module_ m) {
    py::class_<VeryLonelyTraveler, LonelyTraveler>(std::move(m), "VeryLonelyTraveler");
}

} // namespace test_cpp_conduit
} // namespace pybind11_tests
