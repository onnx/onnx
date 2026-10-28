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
    parser.add_argument("--install-prefix", type=Path)
    parser.add_argument("--config", default="Release")
    parser.add_argument(
        "--arch", help="CMake generator architecture, e.g. x64 or ARM64"
    )
    parser.add_argument("--parallel", type=int, default=2)
    args = parser.parse_args()
    cmake = shutil.which("cmake")
    if cmake is None:
        parser.error("CMake must be installed and available on PATH")
    build_dir = args.build_dir.resolve()
    prefix = (args.install_prefix or build_dir / "install").resolve()
    command = [
        cmake,
        "-S",
        str(Path(__file__).resolve().parents[2]),
        "-B",
        str(build_dir),
        f"-DCMAKE_INSTALL_PREFIX={prefix}",
        f"-DCMAKE_BUILD_TYPE={args.config}",
        "-DONNX_BUILD_PROTOBUF_ONLY=ON",
    ]
    if args.arch:
        command.extend(["-A", args.arch])
    subprocess.run(command, check=True)  # noqa: S603
    subprocess.run(  # noqa: S603
        [
            cmake,
            "--build",
            str(build_dir),
            "--config",
            args.config,
            "--target",
            "install",
            "--parallel",
            str(args.parallel),
        ],
        check=True,
    )
    protoc = prefix / "bin" / ("protoc.exe" if os.name == "nt" else "protoc")
    subprocess.run([str(protoc), "--version"], check=True)  # noqa: S603
    # Environment changes in a subprocess do not propagate to its caller.
    # GitHub Actions consumes these files before starting the next step.
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
