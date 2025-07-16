/*
    tests/test_constants_and_functions.cpp -- global constants and functions, enumerations, raw
    byte strings

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"

enum MyEnum { EFirstEntry = 1, ESecondEntry };

std::string test_function1() { return "test_function()"; }

std::string test_function2(MyEnum k) { return "test_function(enum=" + std::to_string(k) + ")"; }

std::string test_function3(int i) { return "test_function(" + std::to_string(i) + ")"; }

py::str test_function4() { return "test_function()"; }
py::str test_function4(char *) { return "test_function(char *)"; }
py::str test_function4(int, float) { return "test_function(int, float)"; }
py::str test_function4(float, int) { return "test_function(float, int)"; }

py::bytes return_bytes() {
    const char *data = "\x01\x00\x02\x00";
    return std::string(data, 4);
}

std::string print_bytes(const py::bytes &bytes) {
    std::string ret = "bytes[";
    const auto value = static_cast<std::string>(bytes);
    for (char c : value) {
        ret += std::to_string(static_cast<int>(c)) + ' ';
    }
    ret.back() = ']';
    return ret;
}

// Test that we properly handle C++17 exception specifiers (which are part of the function
// signature in C++17).  These should all still work before C++17, but don't affect the function
// signature.
namespace test_exc_sp {
// [workaround(intel)] Unable to use noexcept instead of noexcept(true)
// Make the f1 test basically the same as the f2 test in C++17 mode for the Intel compiler as
// it fails to compile with a plain noexcept (tested with icc (ICC) 2021.1 Beta 20200827).
#if defined(__INTEL_COMPILER) && defined(PYBIND11_CPP17)
int f1(int x) noexcept(true) { return x + 1; }
#else
int f1(int x) noexcept { return x + 1; }
#endif
int f2(int x) noexcept(true) { return x + 2; }
int f3(int x) noexcept(false) { return x + 3; }
PYBIND11_WARNING_PUSH
PYBIND11_WARNING_DISABLE_GCC("-Wdeprecated")
#if defined(__clang_major__) && __clang_major__ >= 5
PYBIND11_WARNING_DISABLE_CLANG("-Wdeprecated-dynamic-exception-spec")
#else
PYBIND11_WARNING_DISABLE_CLANG("-Wdeprecated")
#endif
// NOLINTNEXTLINE(modernize-use-noexcept)
int f4(int x) throw() { return x + 4; } // Deprecated equivalent to noexcept(true)
PYBIND11_WARNING_POP
struct C {
    int m1(int x) noexcept { return x - 1; }
    int m2(int x) const noexcept { return x - 2; }
    int m3(int x) noexcept(true) { return x - 3; }
    int m4(int x) const noexcept(true) { return x - 4; }
    int m5(int x) noexcept(false) { return x - 5; }
    int m6(int x) const noexcept(false) { return x - 6; }
    PYBIND11_WARNING_PUSH
    PYBIND11_WARNING_DISABLE_GCC("-Wdeprecated")
    PYBIND11_WARNING_DISABLE_CLANG("-Wdeprecated")
    // NOLINTNEXTLINE(modernize-use-noexcept)
    int m7(int x) throw() { return x - 7; }
    // NOLINTNEXTLINE(modernize-use-noexcept)
    int m8(int x) const throw() { return x - 8; }
    PYBIND11_WARNING_POP
};
} // namespace test_exc_sp

TEST_SUBMODULE(constants_and_functions, m) {
    // test_constants
    m.attr("some_constant") = py::int_(14);

    // test_function_overloading
    m.def("test_function", &test_function1);
    m.def("test_function", &test_function2);
    m.def("test_function", &test_function3);

#if defined(PYBIND11_OVERLOAD_CAST)
    m.def("test_function", py::overload_cast<>(&test_function4));
    m.def("test_function", py::overload_cast<char *>(&test_function4));
    m.def("test_function", py::overload_cast<int, float>(&test_function4));
    m.def("test_function", py::overload_cast<float, int>(&test_function4));
#else
    m.def("test_function", static_cast<py::str (*)()>(&test_function4));
    m.def("test_function", static_cast<py::str (*)(char *)>(&test_function4));
    m.def("test_function", static_cast<py::str (*)(int, float)>(&test_function4));
    m.def("test_function", static_cast<py::str (*)(float, int)>(&test_function4));
#endif

    py::enum_<MyEnum>(m, "MyEnum")
        .value("EFirstEntry", EFirstEntry)
        .value("ESecondEntry", ESecondEntry)
        .export_values();

    // test_bytes
    m.def("return_bytes", &return_bytes);
    m.def("print_bytes", &print_bytes);

    // test_exception_specifiers
    using namespace test_exc_sp;
    py::class_<C>(m, "C")
        .def(py::init<>())
        .def("m1", &C::m1)
        .def("m2", &C::m2)
        .def("m3", &C::m3)
        .def("m4", &C::m4)
        .def("m5", &C::m5)
        .def("m6", &C::m6)
        .def("m7", &C::m7)
        .def("m8", &C::m8);
    m.def("f1", f1);
    m.def("f2", f2);

    PYBIND11_WARNING_PUSH
    PYBIND11_WARNING_DISABLE_INTEL(878) // incompatible exception specifications
    m.def("f3", f3);
    PYBIND11_WARNING_POP

    m.def("f4", f4);

    // test_function_record_leaks
    m.def("register_large_capture_with_invalid_arguments", [](py::module_ m) {
        // This should always be enough to trigger the alternative branch
        // where `sizeof(capture) > sizeof(rec->data)`
        uint64_t capture[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
#if defined(__GNUC__) && __GNUC__ == 4 // CentOS7
        py::detail::silence_unused_warnings(capture);
#endif
        m.def(
            "should_raise", [capture](int) { return capture[9] + 33; }, py::kw_only(), py::arg());
    });
    m.def("register_with_raising_repr", [](py::module_ m, const py::object &default_value) {
        m.def(
            "should_raise",
            [](int, int, const py::object &) { return 42; },
            "some docstring",
            py::arg_v("x", 42),
            py::arg_v("y", 42, "<the answer>"),
            py::arg_v("z", default_value));
    });

    // test noexcept(true) lambda (#4565)
    m.def("l1", []() noexcept(true) { return 0; });
}
