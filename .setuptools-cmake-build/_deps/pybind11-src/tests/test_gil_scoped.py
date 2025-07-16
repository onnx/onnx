from __future__ import annotations

import multiprocessing
import sys
import threading
import time

import pytest

import env
from pybind11_tests import gil_scoped as m


class ExtendedVirtClass(m.VirtClass):
    def virtual_func(self):
        pass

    def pure_virtual_func(self):
        pass


def test_callback_py_obj():
    m.test_callback_py_obj(lambda: None)


def test_callback_std_func():
    m.test_callback_std_func(lambda: None)


def test_callback_virtual_func():
    extended = ExtendedVirtClass()
    m.test_callback_virtual_func(extended)


def test_callback_pure_virtual_func():
    extended = ExtendedVirtClass()
    m.test_callback_pure_virtual_func(extended)


def test_cross_module_gil_released():
    """Makes sure that the GIL can be acquired by another module from a GIL-released state."""
    m.test_cross_module_gil_released()  # Should not raise a SIGSEGV


def test_cross_module_gil_acquired():
    """Makes sure that the GIL can be acquired by another module from a GIL-acquired state."""
    m.test_cross_module_gil_acquired()  # Should not raise a SIGSEGV


def test_cross_module_gil_inner_custom_released():
    """Makes sure that the GIL can be acquired/released by another module
    from a GIL-released state using custom locking logic."""
    m.test_cross_module_gil_inner_custom_released()


def test_cross_module_gil_inner_custom_acquired():
    """Makes sure that the GIL can be acquired/acquired by another module
    from a GIL-acquired state using custom locking logic."""
    m.test_cross_module_gil_inner_custom_acquired()


def test_cross_module_gil_inner_pybind11_released():
    """Makes sure that the GIL can be acquired/released by another module
    from a GIL-released state using pybind11 locking logic."""
    m.test_cross_module_gil_inner_pybind11_released()


def test_cross_module_gil_inner_pybind11_acquired():
    """Makes sure that the GIL can be acquired/acquired by another module
    from a GIL-acquired state using pybind11 locking logic."""
    m.test_cross_module_gil_inner_pybind11_acquired()


@pytest.mark.skipif(sys.platform.startswith("emscripten"), reason="Requires threads")
def test_cross_module_gil_nested_custom_released():
    """Makes sure that the GIL can be nested acquired/released by another module
    from a GIL-released state using custom locking logic."""
    m.test_cross_module_gil_nested_custom_released()


@pytest.mark.skipif(sys.platform.startswith("emscripten"), reason="Requires threads")
def test_cross_module_gil_nested_custom_acquired():
    """Makes sure that the GIL can be nested acquired/acquired by another module
    from a GIL-acquired state using custom locking logic."""
    m.test_cross_module_gil_nested_custom_acquired()


@pytest.mark.skipif(sys.platform.startswith("emscripten"), reason="Requires threads")
def test_cross_module_gil_nested_pybind11_released():
    """Makes sure that the GIL can be nested acquired/released by another module
    from a GIL-released state using pybind11 locking logic."""
    m.test_cross_module_gil_nested_pybind11_released()


@pytest.mark.skipif(sys.platform.startswith("emscripten"), reason="Requires threads")
def test_cross_module_gil_nested_pybind11_acquired():
    """Makes sure that the GIL can be nested acquired/acquired by another module
    from a GIL-acquired state using pybind11 locking logic."""
    m.test_cross_module_gil_nested_pybind11_acquired()


def test_release_acquire():
    assert m.test_release_acquire(0xAB) == "171"


def test_nested_acquire():
    assert m.test_nested_acquire(0xAB) == "171"


@pytest.mark.skipif(sys.platform.startswith("emscripten"), reason="Requires threads")
def test_multi_acquire_release_cross_module():
    for bits in range(16 * 8):
        internals_ids = m.test_multi_acquire_release_cross_module(bits)
        assert len(internals_ids) == 2 if bits % 8 else 1


# Intentionally putting human review in the loop here, to guard against accidents.
VARS_BEFORE_ALL_BASIC_TESTS = dict(vars())  # Make a copy of the dict (critical).
ALL_BASIC_TESTS = (
    test_callback_py_obj,
    test_callback_std_func,
    test_callback_virtual_func,
    test_callback_pure_virtual_func,
    test_cross_module_gil_released,
    test_cross_module_gil_acquired,
    test_cross_module_gil_inner_custom_released,
    test_cross_module_gil_inner_custom_acquired,
    test_cross_module_gil_inner_pybind11_released,
    test_cross_module_gil_inner_pybind11_acquired,
    test_cross_module_gil_nested_custom_released,
    test_cross_module_gil_nested_custom_acquired,
    test_cross_module_gil_nested_pybind11_released,
    test_cross_module_gil_nested_pybind11_acquired,
    test_release_acquire,
    test_nested_acquire,
    test_multi_acquire_release_cross_module,
)


def test_all_basic_tests_completeness():
    num_found = 0
    for key, value in VARS_BEFORE_ALL_BASIC_TESTS.items():
        if not key.startswith("test_"):
            continue
        assert value in ALL_BASIC_TESTS
        num_found += 1
    assert len(ALL_BASIC_TESTS) == num_found


def _intentional_deadlock():
    m.intentional_deadlock()


ALL_BASIC_TESTS_PLUS_INTENTIONAL_DEADLOCK = ALL_BASIC_TESTS + (_intentional_deadlock,)


def _run_in_process(target, *args, **kwargs):
    test_fn = target if len(args) == 0 else args[0]
    # Do not need to wait much, 10s should be more than enough.
    timeout = 0.1 if test_fn is _intentional_deadlock else 10
    process = multiprocessing.Process(target=target, args=args, kwargs=kwargs)
    process.daemon = True
    try:
        t_start = time.time()
        process.start()
        if timeout >= 100:  # For debugging.
            print(
                "\nprocess.pid STARTED", process.pid, (sys.argv, target, args, kwargs)
            )
            print(f"COPY-PASTE-THIS: gdb {sys.argv[0]} -p {process.pid}", flush=True)
        process.join(timeout=timeout)
        if timeout >= 100:
            print("\nprocess.pid JOINED", process.pid, flush=True)
        t_delta = time.time() - t_start
        if process.exitcode == 66 and m.defined_THREAD_SANITIZER:  # Issue #2754
            # WOULD-BE-NICE-TO-HAVE: Check that the message below is actually in the output.
            # Maybe this could work:
            # https://gist.github.com/alexeygrigorev/01ce847f2e721b513b42ea4a6c96905e
            pytest.skip(
                "ThreadSanitizer: starting new threads after multi-threaded fork is not supported."
            )
        elif test_fn is _intentional_deadlock:
            assert process.exitcode is None
            return 0

        if process.exitcode is None:
            assert t_delta > 0.9 * timeout
            msg = "DEADLOCK, most likely, exactly what this test is meant to detect."
            if env.PYPY and env.WIN:
                pytest.skip(msg)
            raise RuntimeError(msg)
        return process.exitcode
    finally:
        if process.is_alive():
            process.terminate()


def _run_in_threads(test_fn, num_threads, parallel):
    threads = []
    for _ in range(num_threads):
        thread = threading.Thread(target=test_fn)
        thread.daemon = True
        thread.start()
        if parallel:
            threads.append(thread)
        else:
            thread.join()
    for thread in threads:
        thread.join()


@pytest.mark.skipif(sys.platform.startswith("emscripten"), reason="Requires threads")
@pytest.mark.parametrize("test_fn", ALL_BASIC_TESTS_PLUS_INTENTIONAL_DEADLOCK)
def test_run_in_process_one_thread(test_fn):
    """Makes sure there is no GIL deadlock when running in a thread.

    It runs in a separate process to be able to stop and assert if it deadlocks.
    """
    assert _run_in_process(_run_in_threads, test_fn, num_threads=1, parallel=False) == 0


@pytest.mark.skipif(sys.platform.startswith("emscripten"), reason="Requires threads")
@pytest.mark.parametrize("test_fn", ALL_BASIC_TESTS_PLUS_INTENTIONAL_DEADLOCK)
def test_run_in_process_multiple_threads_parallel(test_fn):
    """Makes sure there is no GIL deadlock when running in a thread multiple times in parallel.

    It runs in a separate process to be able to stop and assert if it deadlocks.
    """
    assert _run_in_process(_run_in_threads, test_fn, num_threads=8, parallel=True) == 0


@pytest.mark.skipif(sys.platform.startswith("emscripten"), reason="Requires threads")
@pytest.mark.parametrize("test_fn", ALL_BASIC_TESTS_PLUS_INTENTIONAL_DEADLOCK)
def test_run_in_process_multiple_threads_sequential(test_fn):
    """Makes sure there is no GIL deadlock when running in a thread multiple times sequentially.

    It runs in a separate process to be able to stop and assert if it deadlocks.
    """
    assert _run_in_process(_run_in_threads, test_fn, num_threads=8, parallel=False) == 0


@pytest.mark.skipif(sys.platform.startswith("emscripten"), reason="Requires threads")
@pytest.mark.parametrize("test_fn", ALL_BASIC_TESTS_PLUS_INTENTIONAL_DEADLOCK)
def test_run_in_process_direct(test_fn):
    """Makes sure there is no GIL deadlock when using processes.

    This test is for completion, but it was never an issue.
    """
    assert _run_in_process(test_fn) == 0
