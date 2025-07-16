/*
    Copyright (c) 2022 Google LLC

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include <pybind11/pybind11.h>

// This file mimics a DSO that makes pybind11 calls but does not define a PYBIND11_MODULE,
// so that the first call of cross_module_error_already_set() triggers the first call of
// pybind11::detail::get_internals().

namespace {

namespace py = pybind11;

void interleaved_error_already_set() {
    py::set_error(PyExc_RuntimeError, "1st error.");
    try {
        throw py::error_already_set();
    } catch (const py::error_already_set &) {
        // The 2nd error could be conditional in a real application.
        py::set_error(PyExc_RuntimeError, "2nd error.");
    } // Here the 1st error is destroyed before the 2nd error is fetched.
    // The error_already_set dtor triggers a pybind11::detail::get_internals()
    // call via pybind11::gil_scoped_acquire.
    if (PyErr_Occurred()) {
        throw py::error_already_set();
    }
}

constexpr char kModuleName[] = "cross_module_interleaved_error_already_set";

struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, kModuleName, nullptr, 0, nullptr, nullptr, nullptr, nullptr, nullptr};

} // namespace

extern "C" PYBIND11_EXPORT PyObject *PyInit_cross_module_interleaved_error_already_set() {
    PyObject *m = PyModule_Create(&moduledef);
    if (m != nullptr) {
        static_assert(sizeof(&interleaved_error_already_set) == sizeof(void *),
                      "Function pointer must have the same size as void *");
#ifdef Py_GIL_DISABLED
        PyUnstable_Module_SetGIL(m, Py_MOD_GIL_NOT_USED);
#endif
        PyModule_AddObject(
            m,
            "funcaddr",
            PyLong_FromVoidPtr(reinterpret_cast<void *>(&interleaved_error_already_set)));
    }
    return m;
}
