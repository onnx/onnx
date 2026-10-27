# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(
    sys.platform == "win32", reason="The target fixture uses a POSIX shell"
)
@pytest.mark.parametrize("program_mode", [None, "ONLY"])
@pytest.mark.parametrize("explicit_interpreter", [False, True])
def test_cross_compilation_finds_host_python(
    tmp_path: Path, program_mode: str | None, explicit_interpreter: bool
) -> None:
    cmake = shutil.which("cmake")
    make = shutil.which("make")
    if cmake is None or make is None:
        pytest.skip("CMake and make are needed for the configure test")

    source_root = Path(__file__).resolve().parents[2]
    sysroot = tmp_path / "riscv64-sysroot"
    target_bin = sysroot / "usr" / "bin"
    target_bin.mkdir(parents=True)
    target_marker = tmp_path / "target-python-executed"
    target_python = target_bin / "python3"
    target_python.write_text(
        '#!/bin/sh\nprintf "target Python must not run on the build host\\n" >&2\n'
        'touch "$ONNX_TEST_TARGET_PYTHON_MARKER"\nexit 126\n',
        encoding="utf-8",
    )
    target_python.chmod(0o755)

    project = tmp_path / "source"
    project.mkdir()
    (project / "CMakeLists.txt").write_text(
        """cmake_minimum_required(VERSION 3.26)
project(onnx_host_python_test LANGUAGES NONE)
cmake_policy(SET CMP0094 NEW)
if(NOT CMAKE_CROSSCOMPILING OR NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "riscv64")
  message(FATAL_ERROR "The fixture must cross-compile for RISC-V")
endif()
if(DEFINED CMAKE_FIND_ROOT_PATH_MODE_PROGRAM)
  set(expected_program_mode "${CMAKE_FIND_ROOT_PATH_MODE_PROGRAM}")
endif()
include("${ONNX_SOURCE_DIR}/cmake/FindHostPython.cmake")
if(DEFINED expected_program_mode)
  if(NOT CMAKE_FIND_ROOT_PATH_MODE_PROGRAM STREQUAL expected_program_mode)
    message(FATAL_ERROR "The caller's program search mode was changed")
  endif()
elseif(DEFINED CMAKE_FIND_ROOT_PATH_MODE_PROGRAM)
  message(FATAL_ERROR "An originally undefined program search mode was not restored")
endif()
if(DEFINED EXPECTED_HOST_PYTHON AND NOT Python_EXECUTABLE STREQUAL EXPECTED_HOST_PYTHON)
  message(FATAL_ERROR "The explicit Python_EXECUTABLE was changed")
endif()
if(NOT TARGET Python::Interpreter)
  message(FATAL_ERROR "The host interpreter target was not defined")
endif()
cmake_path(IS_PREFIX CMAKE_FIND_ROOT_PATH "${Python_EXECUTABLE}" NORMALIZE is_target_python)
if(is_target_python)
  message(FATAL_ERROR "A target-sysroot Python was selected")
endif()
execute_process(COMMAND "${Python_EXECUTABLE}" -c "print('ONNX_HOST_PYTHON_OK')"
                RESULT_VARIABLE python_result OUTPUT_VARIABLE python_output)
if(NOT python_result EQUAL 0 OR NOT python_output MATCHES "ONNX_HOST_PYTHON_OK")
  message(FATAL_ERROR "The selected interpreter cannot execute on the build host")
endif()
""",
        encoding="utf-8",
    )
    command = [
        cmake,
        "-G",
        "Unix Makefiles",
        "-S",
        str(project),
        "-B",
        str(tmp_path / "build"),
        f"-DONNX_SOURCE_DIR={source_root}",
        "-DCMAKE_SYSTEM_NAME=Linux",
        "-DCMAKE_SYSTEM_PROCESSOR=riscv64",
        f"-DCMAKE_FIND_ROOT_PATH={sysroot}",
        f"-DCMAKE_PROGRAM_PATH={Path(sys.executable).parent}",
        f"-DCMAKE_MAKE_PROGRAM={make}",
        "-DPython_FIND_FRAMEWORK=NEVER",
        "-DPython_FIND_VIRTUALENV=STANDARD",
    ]
    if program_mode is not None:
        command.append(f"-DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM={program_mode}")
    if explicit_interpreter:
        command.extend(
            [
                f"-DPython_EXECUTABLE={sys.executable}",
                f"-DEXPECTED_HOST_PYTHON={sys.executable}",
            ]
        )
    environment = os.environ.copy()
    environment["ONNX_TEST_TARGET_PYTHON_MARKER"] = str(target_marker)
    # CMake and its arguments are controlled by this fixture.
    completed = subprocess.run(  # noqa: S603
        command, capture_output=True, text=True, check=False, env=environment
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert not target_marker.exists(), (
        "Target Python was executed during host discovery"
    )
