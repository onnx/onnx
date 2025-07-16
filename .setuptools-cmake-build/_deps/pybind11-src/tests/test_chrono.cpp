/*
    tests/test_chrono.cpp -- test conversions to/from std::chrono types

    Copyright (c) 2016 Trent Houliston <trent@houliston.me> and
                       Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include <pybind11/chrono.h>

#include "pybind11_tests.h"

#include <chrono>

struct different_resolutions {
    using time_point_h = std::chrono::time_point<std::chrono::system_clock, std::chrono::hours>;
    using time_point_m = std::chrono::time_point<std::chrono::system_clock, std::chrono::minutes>;
    using time_point_s = std::chrono::time_point<std::chrono::system_clock, std::chrono::seconds>;
    using time_point_ms
        = std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds>;
    using time_point_us
        = std::chrono::time_point<std::chrono::system_clock, std::chrono::microseconds>;
    time_point_h timestamp_h;
    time_point_m timestamp_m;
    time_point_s timestamp_s;
    time_point_ms timestamp_ms;
    time_point_us timestamp_us;
};

TEST_SUBMODULE(chrono, m) {
    using system_time = std::chrono::system_clock::time_point;
    using steady_time = std::chrono::steady_clock::time_point;

    using timespan = std::chrono::duration<int64_t, std::nano>;
    using timestamp = std::chrono::time_point<std::chrono::system_clock, timespan>;

    // test_chrono_system_clock
    // Return the current time off the wall clock
    m.def("test_chrono1", []() { return std::chrono::system_clock::now(); });

    // test_chrono_system_clock_roundtrip
    // Round trip the passed in system clock time
    m.def("test_chrono2", [](system_time t) { return t; });

    // test_chrono_duration_roundtrip
    // Round trip the passed in duration
    m.def("test_chrono3", [](std::chrono::system_clock::duration d) { return d; });

    // test_chrono_duration_subtraction_equivalence
    // Difference between two passed in time_points
    m.def("test_chrono4", [](system_time a, system_time b) { return a - b; });

    // test_chrono_steady_clock
    // Return the current time off the steady_clock
    m.def("test_chrono5", []() { return std::chrono::steady_clock::now(); });

    // test_chrono_steady_clock_roundtrip
    // Round trip a steady clock timepoint
    m.def("test_chrono6", [](steady_time t) { return t; });

    // test_floating_point_duration
    // Roundtrip a duration in microseconds from a float argument
    m.def("test_chrono7", [](std::chrono::microseconds t) { return t; });
    // Float durations (issue #719)
    m.def("test_chrono_float_diff",
          [](std::chrono::duration<float> a, std::chrono::duration<float> b) { return a - b; });

    m.def("test_nano_timepoint",
          [](timestamp start, timespan delta) -> timestamp { return start + delta; });

    // Test different resolutions
    py::class_<different_resolutions>(m, "different_resolutions")
        .def(py::init<>())
        .def_readwrite("timestamp_h", &different_resolutions::timestamp_h)
        .def_readwrite("timestamp_m", &different_resolutions::timestamp_m)
        .def_readwrite("timestamp_s", &different_resolutions::timestamp_s)
        .def_readwrite("timestamp_ms", &different_resolutions::timestamp_ms)
        .def_readwrite("timestamp_us", &different_resolutions::timestamp_us);
}
