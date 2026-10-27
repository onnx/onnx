# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def _write_protobuf_fixture(prefix: Path) -> None:
    include_dir = prefix / "include" / "google" / "protobuf"
    (include_dir / "stubs").mkdir(parents=True)
    (include_dir / "service.h").touch()
    (include_dir / "stubs" / "common.h").write_text(
        "#define GOOGLE_PROTOBUF_VERSION 3021012\n", encoding="utf-8"
    )
    (prefix / "lib").mkdir()
    # The tests configure targets without linking these fixture libraries.
    (prefix / "lib" / "libprotobuf.a").write_bytes(b"!<arch>\n")


def _write_host_protoc(tmp_path: Path) -> Path:
    host_protoc = tmp_path / "host" / "bin" / "protoc"
    host_protoc.parent.mkdir(parents=True)
    host_protoc.write_text(
        '#!/bin/sh\nprintf "libprotoc 3.21.12\\n"\n', encoding="utf-8"
    )
    host_protoc.chmod(0o755)
    return host_protoc


def _configure_command(tmp_path: Path, sysroot: Path, library_mode: str) -> list[str]:
    cmake = shutil.which("cmake")
    compiler = shutil.which("c++")
    make = shutil.which("make")
    if cmake is None or compiler is None or make is None:
        pytest.skip("CMake, make and a C++ compiler are needed for the configure test")
    source_dir = Path(__file__).resolve().parents[2]
    toolchain = tmp_path / "riscv64.cmake"
    toolchain.write_text(
        "set(CMAKE_SYSTEM_NAME Linux)\n"
        "set(CMAKE_SYSTEM_PROCESSOR riscv64)\n"
        "set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)\n"
        f'set(CMAKE_CXX_COMPILER "{Path(compiler).as_posix()}")\n'
        f'set(CMAKE_FIND_ROOT_PATH "{sysroot.as_posix()}")\n'
        "set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)\n"
        f"set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY {library_mode})\n"
        f"set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE {library_mode})\n"
        "set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)\n",
        encoding="utf-8",
    )
    return [
        cmake,
        "-G",
        "Unix Makefiles",
        "-S",
        str(source_dir),
        "-B",
        str(tmp_path / "build"),
        f"-DCMAKE_MAKE_PROGRAM={make}",
        f"-DCMAKE_TOOLCHAIN_FILE={toolchain}",
        "-DONNX_BUILD_PYTHON=OFF",
        "-DONNX_BUILD_TESTS=OFF",
        "-DONNX_INSTALL=OFF",
    ]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX cross-toolchain fixture")
@pytest.mark.parametrize("library_mode", ["ONLY", "BOTH"])
@pytest.mark.parametrize("cached_protoc", ["host", "target"])
def test_external_protobuf_uses_host_protoc_when_cross_compiling(
    tmp_path: Path, library_mode: str, cached_protoc: str
) -> None:
    host_protoc = _write_host_protoc(tmp_path)
    # Deliberately offer host libraries beside protoc. BOTH allows this prefix
    # to be found if ONNX incorrectly seeds target lookups from the host tool.
    _write_protobuf_fixture(tmp_path / "host")
    sysroot = tmp_path / "riscv64-sysroot"
    _write_protobuf_fixture(sysroot)
    (sysroot / "bin").mkdir()
    target_protoc = sysroot / "bin" / "protoc"
    target_protoc.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    target_protoc.chmod(0o755)

    command = _configure_command(tmp_path, sysroot, library_mode)
    command.extend(
        [
            f"-DONNX_CUSTOM_PROTOC_EXECUTABLE={host_protoc}",
            "-DProtobuf_PROTOC_EXECUTABLE="
            + str(host_protoc if cached_protoc == "host" else target_protoc),
        ]
    )
    result = subprocess.run(command, capture_output=True, text=True, check=False)  # noqa: S603
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"ONNX_PROTOC_EXECUTABLE: {host_protoc}" in result.stdout
    cache = (tmp_path / "build" / "CMakeCache.txt").read_text(encoding="utf-8")
    assert f"Protobuf_INCLUDE_DIR:PATH={sysroot.as_posix()}/include" in cache
    assert (
        f"Protobuf_LIBRARY_RELEASE:FILEPATH={sysroot.as_posix()}/lib/libprotobuf.a"
        in cache
    )


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX cross-toolchain fixture")
def test_custom_protoc_keeps_protobuf_build_fallback(tmp_path: Path) -> None:
    host_protoc = _write_host_protoc(tmp_path)
    sysroot = tmp_path / "empty-riscv64-sysroot"
    sysroot.mkdir()
    absl_source = tmp_path / "absl"
    protobuf_source = tmp_path / "protobuf"
    absl_source.mkdir()
    protobuf_source.mkdir()
    (absl_source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.26)\nproject(absl_fixture LANGUAGES NONE)\n",
        encoding="utf-8",
    )
    (protobuf_source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.26)\n"
        "project(protobuf_fixture LANGUAGES NONE)\n"
        "add_library(libprotobuf INTERFACE)\n"
        "add_library(protobuf::libprotobuf ALIAS libprotobuf)\n",
        encoding="utf-8",
    )
    command = _configure_command(tmp_path, sysroot, "ONLY")
    command.extend(
        [
            f"-DONNX_CUSTOM_PROTOC_EXECUTABLE={host_protoc}",
            "-DCMAKE_DISABLE_FIND_PACKAGE_Protobuf=TRUE",
            f"-DFETCHCONTENT_SOURCE_DIR_ABSL={absl_source}",
            f"-DFETCHCONTENT_SOURCE_DIR_PROTOBUF={protobuf_source}",
            "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
        ]
    )
    result = subprocess.run(command, capture_output=True, text=True, check=False)  # noqa: S603
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"ONNX_PROTOC_EXECUTABLE: {host_protoc}" in result.stdout
    assert "Download and build Protobuf from" in result.stdout
