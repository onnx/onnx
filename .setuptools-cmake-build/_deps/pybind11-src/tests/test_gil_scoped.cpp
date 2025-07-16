/*
    tests/test_gil_scoped.cpp -- acquire and release gil

    Copyright (c) 2017 Borja Zarco (Google LLC) <bzarco@google.com>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include <pybind11/functional.h>

#include "pybind11_tests.h"

#include <string>
#include <thread>

#define CROSS_MODULE(Function)                                                                    \
    auto cm = py::module_::import("cross_module_gil_utils");                                      \
    auto target = reinterpret_cast<void (*)()>(PyLong_AsVoidPtr(cm.attr(Function).ptr()));

class VirtClass {
public:
    virtual ~VirtClass() = default;
    VirtClass() = default;
    VirtClass(const VirtClass &) = delete;
    virtual void virtual_func() {}
    virtual void pure_virtual_func() = 0;
};

class PyVirtClass : public VirtClass {
    void virtual_func() override { PYBIND11_OVERRIDE(void, VirtClass, virtual_func, ); }
    void pure_virtual_func() override {
        PYBIND11_OVERRIDE_PURE(void, VirtClass, pure_virtual_func, );
    }
};

TEST_SUBMODULE(gil_scoped, m) {
    m.attr("defined_THREAD_SANITIZER") =
#if defined(THREAD_SANITIZER)
        true;
#else
        false;
#endif

    m.def("intentional_deadlock",
          []() { std::thread([]() { py::gil_scoped_acquire gil_acquired; }).join(); });

    py::class_<VirtClass, PyVirtClass>(m, "VirtClass")
        .def(py::init<>())
        .def("virtual_func", &VirtClass::virtual_func)
        .def("pure_virtual_func", &VirtClass::pure_virtual_func);

    m.def("test_callback_py_obj", [](py::object &func) { func(); });
    m.def("test_callback_std_func", [](const std::function<void()> &func) { func(); });
    m.def("test_callback_virtual_func", [](VirtClass &virt) { virt.virtual_func(); });
    m.def("test_callback_pure_virtual_func", [](VirtClass &virt) { virt.pure_virtual_func(); });
    m.def("test_cross_module_gil_released", []() {
        CROSS_MODULE("gil_acquire_funcaddr")
        py::gil_scoped_release gil_release;
        target();
    });
    m.def("test_cross_module_gil_acquired", []() {
        CROSS_MODULE("gil_acquire_funcaddr")
        py::gil_scoped_acquire gil_acquire;
        target();
    });
    m.def("test_cross_module_gil_inner_custom_released", []() {
        CROSS_MODULE("gil_acquire_inner_custom_funcaddr")
        py::gil_scoped_release gil_release;
        target();
    });
    m.def("test_cross_module_gil_inner_custom_acquired", []() {
        CROSS_MODULE("gil_acquire_inner_custom_funcaddr")
        py::gil_scoped_acquire gil_acquire;
        target();
    });
    m.def("test_cross_module_gil_inner_pybind11_released", []() {
        CROSS_MODULE("gil_acquire_inner_pybind11_funcaddr")
        py::gil_scoped_release gil_release;
        target();
    });
    m.def("test_cross_module_gil_inner_pybind11_acquired", []() {
        CROSS_MODULE("gil_acquire_inner_pybind11_funcaddr")
        py::gil_scoped_acquire gil_acquire;
        target();
    });
    m.def("test_cross_module_gil_nested_custom_released", []() {
        CROSS_MODULE("gil_acquire_nested_custom_funcaddr")
        py::gil_scoped_release gil_release;
        target();
    });
    m.def("test_cross_module_gil_nested_custom_acquired", []() {
        CROSS_MODULE("gil_acquire_nested_custom_funcaddr")
        py::gil_scoped_acquire gil_acquire;
        target();
    });
    m.def("test_cross_module_gil_nested_pybind11_released", []() {
        CROSS_MODULE("gil_acquire_nested_pybind11_funcaddr")
        py::gil_scoped_release gil_release;
        target();
    });
    m.def("test_cross_module_gil_nested_pybind11_acquired", []() {
        CROSS_MODULE("gil_acquire_nested_pybind11_funcaddr")
        py::gil_scoped_acquire gil_acquire;
        target();
    });
    m.def("test_release_acquire", [](const py::object &obj) {
        py::gil_scoped_release gil_released;
        py::gil_scoped_acquire gil_acquired;
        return py::str(obj);
    });
    m.def("test_nested_acquire", [](const py::object &obj) {
        py::gil_scoped_release gil_released;
        py::gil_scoped_acquire gil_acquired_outer;
        py::gil_scoped_acquire gil_acquired_inner;
        return py::str(obj);
    });
    m.def("test_multi_acquire_release_cross_module", [](unsigned bits) {
        py::set internals_ids;
        internals_ids.add(PYBIND11_INTERNALS_ID);
        {
            py::gil_scoped_release gil_released;
            auto thread_f = [bits, &internals_ids]() {
                py::gil_scoped_acquire gil_acquired;
                auto cm = py::module_::import("cross_module_gil_utils");
                auto target = reinterpret_cast<std::string (*)(unsigned)>(
                    PyLong_AsVoidPtr(cm.attr("gil_multi_acquire_release_funcaddr").ptr()));
                std::string cm_internals_id = target(bits >> 3);
                internals_ids.add(cm_internals_id);
            };
            if ((bits & 0x1u) != 0u) {
                thread_f();
            }
            if ((bits & 0x2u) != 0u) {
                std::thread non_python_thread(thread_f);
                non_python_thread.join();
            }
            if ((bits & 0x4u) != 0u) {
                thread_f();
            }
        }
        return internals_ids;
    });
}
