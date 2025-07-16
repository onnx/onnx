/*
    tests/test_iostream.cpp -- Usage of scoped_output_redirect

    Copyright (c) 2017 Henry F. Schreiner

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include <pybind11/iostream.h>

#include "pybind11_tests.h"

#include <atomic>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

void noisy_function(const std::string &msg, bool flush) {

    std::cout << msg;
    if (flush) {
        std::cout << std::flush;
    }
}

void noisy_funct_dual(const std::string &msg, const std::string &emsg) {
    std::cout << msg;
    std::cerr << emsg;
}

// object to manage C++ thread
// simply repeatedly write to std::cerr until stopped
// redirect is called at some point to test the safety of scoped_estream_redirect
struct TestThread {
    TestThread() : stop_{false} {
        auto thread_f = [this] {
            static std::mutex cout_mutex;
            while (!stop_) {
                {
                    // #HelpAppreciated: Work on iostream.h thread safety.
                    // Without this lock, the clang ThreadSanitizer (tsan) reliably reports a
                    // data race, and this test is predictably flakey on Windows.
                    // For more background see the discussion under
                    // https://github.com/pybind/pybind11/pull/2982 and
                    // https://github.com/pybind/pybind11/pull/2995.
                    const std::lock_guard<std::mutex> lock(cout_mutex);
                    std::cout << "x" << std::flush;
                }
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
        };
        t_ = new std::thread(std::move(thread_f));
    }

    ~TestThread() { delete t_; }

    void stop() { stop_ = true; }

    void join() const {
        py::gil_scoped_release gil_lock;
        t_->join();
    }

    void sleep() {
        py::gil_scoped_release gil_lock;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    std::thread *t_{nullptr};
    std::atomic<bool> stop_;
};

TEST_SUBMODULE(iostream, m) {

    add_ostream_redirect(m);

    // test_evals

    m.def("captured_output_default", [](const std::string &msg) {
        py::scoped_ostream_redirect redir;
        std::cout << msg << std::flush;
    });

    m.def("captured_output", [](const std::string &msg) {
        py::scoped_ostream_redirect redir(std::cout, py::module_::import("sys").attr("stdout"));
        std::cout << msg << std::flush;
    });

    m.def("guard_output",
          &noisy_function,
          py::call_guard<py::scoped_ostream_redirect>(),
          py::arg("msg"),
          py::arg("flush") = true);

    m.def("captured_err", [](const std::string &msg) {
        py::scoped_ostream_redirect redir(std::cerr, py::module_::import("sys").attr("stderr"));
        std::cerr << msg << std::flush;
    });

    m.def("noisy_function", &noisy_function, py::arg("msg"), py::arg("flush") = true);

    m.def("dual_guard",
          &noisy_funct_dual,
          py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>(),
          py::arg("msg"),
          py::arg("emsg"));

    m.def("raw_output", [](const std::string &msg) { std::cout << msg << std::flush; });

    m.def("raw_err", [](const std::string &msg) { std::cerr << msg << std::flush; });

    m.def("captured_dual", [](const std::string &msg, const std::string &emsg) {
        py::scoped_ostream_redirect redirout(std::cout, py::module_::import("sys").attr("stdout"));
        py::scoped_ostream_redirect redirerr(std::cerr, py::module_::import("sys").attr("stderr"));
        std::cout << msg << std::flush;
        std::cerr << emsg << std::flush;
    });

    py::class_<TestThread>(m, "TestThread")
        .def(py::init<>())
        .def("stop", &TestThread::stop)
        .def("join", &TestThread::join)
        .def("sleep", &TestThread::sleep);
}
