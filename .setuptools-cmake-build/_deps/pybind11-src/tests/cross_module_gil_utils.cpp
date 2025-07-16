/*
    tests/cross_module_gil_utils.cpp -- tools for acquiring GIL from a different module

    Copyright (c) 2019 Google LLC

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/
#if defined(PYBIND11_INTERNALS_VERSION)
#    undef PYBIND11_INTERNALS_VERSION
#endif
#define PYBIND11_INTERNALS_VERSION 21814642 // Ensure this module has its own `internals` instance.
#include <pybind11/pybind11.h>

#include <cstdint>
#include <string>
#include <thread>

// This file mimics a DSO that makes pybind11 calls but does not define a
// PYBIND11_MODULE. The purpose is to test that such a DSO can create a
// py::gil_scoped_acquire when the running thread is in a GIL-released state.
//
// Note that we define a Python module here for convenience, but in general
// this need not be the case. The typical scenario would be a DSO that implements
// shared logic used internally by multiple pybind11 modules.

namespace {

namespace py = pybind11;

void gil_acquire() { py::gil_scoped_acquire gil; }

std::string gil_multi_acquire_release(unsigned bits) {
    if ((bits & 0x1u) != 0u) {
        py::gil_scoped_acquire gil;
    }
    if ((bits & 0x2u) != 0u) {
        py::gil_scoped_release gil;
    }
    if ((bits & 0x4u) != 0u) {
        py::gil_scoped_acquire gil;
    }
    if ((bits & 0x8u) != 0u) {
        py::gil_scoped_release gil;
    }
    return PYBIND11_INTERNALS_ID;
}

struct CustomAutoGIL {
    CustomAutoGIL() : gstate(PyGILState_Ensure()) {}
    ~CustomAutoGIL() { PyGILState_Release(gstate); }

    PyGILState_STATE gstate;
};
struct CustomAutoNoGIL {
    CustomAutoNoGIL() : save(PyEval_SaveThread()) {}
    ~CustomAutoNoGIL() { PyEval_RestoreThread(save); }

    PyThreadState *save;
};

template <typename Acquire, typename Release>
void gil_acquire_inner() {
    Acquire acquire_outer;
    Acquire acquire_inner;
    Release release;
}

template <typename Acquire, typename Release>
void gil_acquire_nested() {
    Acquire acquire_outer;
    Acquire acquire_inner;
    Release release;
    auto thread = std::thread(&gil_acquire_inner<Acquire, Release>);
    thread.join();
}

constexpr char kModuleName[] = "cross_module_gil_utils";

struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, kModuleName, nullptr, 0, nullptr, nullptr, nullptr, nullptr, nullptr};

} // namespace

#define ADD_FUNCTION(Name, ...)                                                                   \
    PyModule_AddObject(m, Name, PyLong_FromVoidPtr(reinterpret_cast<void *>(&__VA_ARGS__)));

extern "C" PYBIND11_EXPORT PyObject *PyInit_cross_module_gil_utils() {

    PyObject *m = PyModule_Create(&moduledef);

    if (m != nullptr) {
        static_assert(sizeof(&gil_acquire) == sizeof(void *),
                      "Function pointer must have the same size as void*");
#ifdef Py_GIL_DISABLED
        PyUnstable_Module_SetGIL(m, Py_MOD_GIL_NOT_USED);
#endif
        ADD_FUNCTION("gil_acquire_funcaddr", gil_acquire)
        ADD_FUNCTION("gil_multi_acquire_release_funcaddr", gil_multi_acquire_release)
        ADD_FUNCTION("gil_acquire_inner_custom_funcaddr",
                     gil_acquire_inner<CustomAutoGIL, CustomAutoNoGIL>)
        ADD_FUNCTION("gil_acquire_nested_custom_funcaddr",
                     gil_acquire_nested<CustomAutoGIL, CustomAutoNoGIL>)
        ADD_FUNCTION("gil_acquire_inner_pybind11_funcaddr",
                     gil_acquire_inner<py::gil_scoped_acquire, py::gil_scoped_release>)
        ADD_FUNCTION("gil_acquire_nested_pybind11_funcaddr",
                     gil_acquire_nested<py::gil_scoped_acquire, py::gil_scoped_release>)
    }

    return m;
}
