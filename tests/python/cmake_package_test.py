# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX cross-sysroot fixture")
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize(
    ("lite", "target"),
    [
        (False, "protobuf::libprotobuf"),
        (True, "protobuf::libprotobuf-lite"),
        (True, "protobuf::libprotobuf"),
    ],
)
@pytest.mark.parametrize("discovery", ["config", "module", "existing", "broken_config"])
def test_onnx_package_discovers_protobuf(
    tmp_path: Path, shared: bool, lite: bool, target: str, discovery: str
) -> None:
    cmake = shutil.which("cmake")
    compiler = shutil.which("c++")
    make = shutil.which("make")
    if cmake is None or compiler is None or make is None:
        pytest.skip("CMake, make and a C++ compiler are needed for the package test")

    source_root = Path(__file__).resolve().parents[2]
    prefix = tmp_path / "sysroot" / "usr"
    onnx_package = prefix / "lib" / "cmake" / "ONNX"
    onnx_package.mkdir(parents=True)
    template = (source_root / "cmake" / "ONNXConfig.cmake.in").read_text(
        encoding="utf-8"
    )
    config = template.replace("@ONNX_VERSION@", "1.24.0")
    config = config.replace(
        "@ONNX_USE_PROTOBUF_SHARED_LIBS@", "ON" if shared else "OFF"
    )
    config = config.replace("@ONNX_USE_LITE_PROTO@", "ON" if lite else "OFF")
    config = config.replace("@LINKED_PROTOBUF_TARGET@", target)
    (onnx_package / "ONNXConfig.cmake").write_text(config, encoding="utf-8")
    (onnx_package / "ONNXTargets.cmake").write_text(
        "add_library(ONNX::onnx INTERFACE IMPORTED)\n"
        f'set_property(TARGET ONNX::onnx PROPERTY INTERFACE_LINK_LIBRARIES "{target}")\n',
        encoding="utf-8",
    )

    project = tmp_path / "source"
    project.mkdir()
    module_dir = project / "modules"
    module_dir.mkdir()
    dependency = f"add_library({target} INTERFACE IMPORTED)\nset(Protobuf_FOUND TRUE)\n"
    if discovery in {"config", "broken_config"}:
        protobuf_package = prefix / "lib" / "cmake" / "Protobuf"
        protobuf_package.mkdir()
        if discovery == "broken_config":
            dependency = (
                "include(CMakeFindDependencyMacro)\n"
                "find_dependency(ONNXMissingProtobufDependency CONFIG)\n"
            )
            (module_dir / "FindProtobuf.cmake").write_text(
                'message(FATAL_ERROR "MODULE fallback must not mask a broken CONFIG")\n',
                encoding="utf-8",
            )
        (protobuf_package / "ProtobufConfig.cmake").write_text(
            dependency, encoding="utf-8"
        )
    elif discovery == "module":
        (module_dir / "FindProtobuf.cmake").write_text(dependency, encoding="utf-8")
    else:
        # A parent project may already have provided the selected library.
        (module_dir / "FindProtobuf.cmake").write_text(
            'message(FATAL_ERROR "An existing Protobuf target must be reused")\n',
            encoding="utf-8",
        )
    preexisting = dependency if discovery == "existing" else ""
    (project / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.26)\n"
        "project(onnx_package_test LANGUAGES CXX)\n"
        'list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/modules")\n'
        + preexisting
        + "find_package(ONNX REQUIRED CONFIG)\n"
        "add_executable(consumer main.cc)\n"
        "target_link_libraries(consumer PRIVATE ONNX::onnx)\n",
        encoding="utf-8",
    )
    (project / "main.cc").write_text("int main() { return 0; }\n", encoding="utf-8")
    # Execute only the local CMake binary with this test's generated fixtures.
    completed = subprocess.run(  # noqa: S603
        [
            cmake,
            "-G",
            "Unix Makefiles",
            "-S",
            str(project),
            "-B",
            str(tmp_path / "build"),
            f"-DONNX_DIR={onnx_package}",
            f"-DCMAKE_PREFIX_PATH={prefix}",
            f"-DCMAKE_FIND_ROOT_PATH={prefix.parent}",
            "-DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=ONLY",
            "-DCMAKE_SYSTEM_NAME=Linux",
            "-DCMAKE_SYSTEM_PROCESSOR=riscv64",
            "-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY",
            f"-DCMAKE_CXX_COMPILER={compiler}",
            f"-DCMAKE_MAKE_PROGRAM={make}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if discovery == "broken_config":
        assert completed.returncode != 0
        assert "ONNXMissingProtobufDependency" in completed.stdout + completed.stderr
        assert (
            "MODULE fallback must not mask" not in completed.stdout + completed.stderr
        )
    else:
        assert completed.returncode == 0, completed.stdout + completed.stderr
