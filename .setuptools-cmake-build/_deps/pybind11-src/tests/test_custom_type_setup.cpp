/*
    tests/test_custom_type_setup.cpp -- Tests `pybind11::custom_type_setup`

    Copyright (c) Google LLC

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include <pybind11/pybind11.h>

#include "pybind11_tests.h"

namespace py = pybind11;

namespace {

struct OwnsPythonObjects {
    py::object value = py::none();
};
} // namespace

TEST_SUBMODULE(custom_type_setup, m) {
    py::class_<OwnsPythonObjects> cls(
        m, "OwnsPythonObjects", py::custom_type_setup([](PyHeapTypeObject *heap_type) {
            auto *type = &heap_type->ht_type;
            type->tp_flags |= Py_TPFLAGS_HAVE_GC;
            type->tp_traverse = [](PyObject *self_base, visitproc visit, void *arg) {
                auto &self = py::cast<OwnsPythonObjects &>(py::handle(self_base));
                Py_VISIT(self.value.ptr());
                return 0;
            };
            type->tp_clear = [](PyObject *self_base) {
                auto &self = py::cast<OwnsPythonObjects &>(py::handle(self_base));
                self.value = py::none();
                return 0;
            };
        }));
    cls.def(py::init<>());
    cls.def_readwrite("value", &OwnsPythonObjects::value);
}
