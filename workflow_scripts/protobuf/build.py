# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the external Protobuf CI dependency with CMake."
    )
    parser.add_argument(
        "--build-dir", type=Path, default=Path(".setuptools-cmake-build/protobuf")
    )
    parser.add_argument(
        "--arch", help="CMake generator architecture, e.g. x64 or ARM64"
    )
    args = parser.parse_args()
    cmake = shutil.which("cmake")
    if cmake is None:
        parser.error("CMake must be installed and available on PATH")
    build_dir = args.build_dir.resolve()
    prefix = build_dir / "install"

    def run_cmake(*options: str | Path) -> None:
        subprocess.run([cmake, *map(str, options)], check=True)  # noqa: S603

    # Makefiles and Visual Studio support building dependency subprojects.
    generator = (
        os.environ.get("CMAKE_GENERATOR", "Visual Studio 17 2022")
        if os.name == "nt"
        else "Unix Makefiles"
    )
    command = [
        "-S",
        str(Path(__file__).resolve().parents[2]),
        "-B",
        str(build_dir),
        "-G",
        generator,
        f"-DCMAKE_INSTALL_PREFIX={prefix}",
        "-DONNX_BUILD_CUSTOM_PROTOBUF=ON",
        "-DONNX_INSTALL=OFF",
    ]
    if args.arch:
        command.extend(["-A", args.arch])
    run_cmake(*command)
    for dependency in ("absl", "protobuf"):
        dependency_dir = build_dir / "_deps" / f"{dependency}-build"
        run_cmake("--build", dependency_dir, "--config", "Release", "--parallel", "2")
        run_cmake("--install", dependency_dir, "--config", "Release")
    if github_env := os.environ.get("GITHUB_ENV"):
        existing = os.environ.get("CMAKE_PREFIX_PATH", "")
        value = str(prefix) + (os.pathsep + existing if existing else "")
        with open(github_env, "a", encoding="utf-8") as stream:
            stream.write(f"CMAKE_PREFIX_PATH={value}\n")
    if github_path := os.environ.get("GITHUB_PATH"):
        with open(github_path, "a", encoding="utf-8") as stream:
            stream.write(f"{prefix / 'bin'}\n")


if __name__ == "__main__":
    main()
