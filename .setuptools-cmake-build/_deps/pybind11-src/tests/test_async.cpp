/*
    tests/test_async.cpp -- __await__ support

    Copyright (c) 2019 Google Inc.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"

TEST_SUBMODULE(async_module, m) {
    struct DoesNotSupportAsync {};
    py::class_<DoesNotSupportAsync>(m, "DoesNotSupportAsync").def(py::init<>());
    struct SupportsAsync {};
    py::class_<SupportsAsync>(m, "SupportsAsync")
        .def(py::init<>())
        .def("__await__", [](const SupportsAsync &self) -> py::object {
            static_cast<void>(self);
            py::object loop = py::module_::import("asyncio.events").attr("get_event_loop")();
            py::object f = loop.attr("create_future")();
            f.attr("set_result")(5);
            return f.attr("__await__")();
        });
}
