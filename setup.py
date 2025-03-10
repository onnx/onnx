# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

# NOTE: Put all metadata in pyproject.toml.
# Set the environment variable `ONNX_PREVIEW_BUILD=1` to build the dev preview release.
from __future__ import annotations

import contextlib
import datetime
import glob
import logging
import multiprocessing
import os
import platform
import shlex
import shutil
import subprocess
import sys
import sysconfig
import textwrap
from typing import ClassVar

import setuptools
import setuptools.command.build_ext
import setuptools.command.build_py
import setuptools.command.develop

TOP_DIR = os.path.realpath(os.path.dirname(__file__))
CMAKE_BUILD_DIR = os.path.join(TOP_DIR, ".setuptools-cmake-build")

WINDOWS = os.name == "nt"

CMAKE = shutil.which("cmake3") or shutil.which("cmake")

################################################################################
# Global variables for controlling the build variant
################################################################################

# Default value is set to TRUE\1 to keep the settings same as the current ones.
# However going forward the recommended way to is to set this to False\0
ONNX_ML = os.getenv("ONNX_ML") != "0"
ONNX_VERIFY_PROTO3 = os.getenv("ONNX_VERIFY_PROTO3") == "1"
ONNX_NAMESPACE = os.getenv("ONNX_NAMESPACE", "onnx")
ONNX_BUILD_TESTS = os.getenv("ONNX_BUILD_TESTS") == "1"
ONNX_DISABLE_EXCEPTIONS = os.getenv("ONNX_DISABLE_EXCEPTIONS") == "1"
ONNX_DISABLE_STATIC_REGISTRATION = os.getenv("ONNX_DISABLE_STATIC_REGISTRATION") == "1"
ONNX_PREVIEW_BUILD = os.getenv("ONNX_PREVIEW_BUILD") == "1"

USE_MSVC_STATIC_RUNTIME = os.getenv("USE_MSVC_STATIC_RUNTIME", "0") == "1"
DEBUG = os.getenv("DEBUG", "0") == "1"
COVERAGE = os.getenv("COVERAGE", "0") == "1"

# Customize the wheel plat-name; sometimes useful for MacOS builds.
# See https://github.com/onnx/onnx/pull/6117
ONNX_WHEEL_PLATFORM_NAME = os.getenv("ONNX_WHEEL_PLATFORM_NAME")

################################################################################
# Version
################################################################################

try:
    _git_version = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=TOP_DIR)
        .decode("ascii")
        .strip()
    )
except (OSError, subprocess.CalledProcessError):
    _git_version = ""

with open(os.path.join(TOP_DIR, "VERSION_NUMBER"), encoding="utf-8") as version_file:
    _version = version_file.read().strip()
    if ONNX_PREVIEW_BUILD:
        # Create the preview build for weekly releases
        todays_date = datetime.date.today().strftime("%Y%m%d")
        _version += ".dev" + todays_date
    VERSION_INFO = {"version": _version, "git_version": _git_version}

################################################################################
# Utilities
################################################################################


@contextlib.contextmanager
def cd(path):
    if not os.path.isabs(path):
        raise RuntimeError(f"Can only cd to absolute path, got: {path}")
    orig_path = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(orig_path)


def get_python_execute():
    if WINDOWS:
        return sys.executable
    # Try to search more accurate path, because sys.executable may return a wrong one,
    # as discussed in https://github.com/python/cpython/issues/84399
    python_dir = os.path.abspath(
        os.path.join(sysconfig.get_path("include"), "..", "..")
    )
    if os.path.isdir(python_dir):
        python_bin = os.path.join(python_dir, "bin", "python3")
        if os.path.isfile(python_bin):
            return python_bin
        python_bin = os.path.join(python_dir, "bin", "python")
        if os.path.isfile(python_bin):
            return python_bin
    return sys.executable


# Support for Reproducible Builds
# https://reproducible-builds.org/docs/source-date-epoch/
# https://github.com/pypa/setuptools/issues/2133#issuecomment-1691158410

timestamp = os.environ.get("SOURCE_DATE_EPOCH")
if timestamp is not None:
    import stat
    import tarfile
    import time
    from distutils import archive_util

    timestamp = float(max(int(timestamp), 0))

    class Time:
        @staticmethod
        def time():
            return timestamp
        @staticmethod
        def localtime(_=None):
            return time.localtime(timestamp)

    class TarInfoMode:
        def __get__(self, obj, objtype=None):
            return obj._mode
        def __set__(self, obj, stmd):
            ifmt = stat.S_IFMT(stmd)
            mode = stat.S_IMODE(stmd) & 0o7755
            obj._mode = ifmt | mode

    class TarInfoAttr:
        def __init__(self, value):
            self.value = value
        def __get__(self, obj, objtype=None):
            return self.value
        def __set__(self, obj, value):
            pass

    class TarInfo(tarfile.TarInfo):
        mode = TarInfoMode()
        mtime = TarInfoAttr(timestamp)
        uid = TarInfoAttr(0)
        gid = TarInfoAttr(0)
        uname = TarInfoAttr('')
        gname = TarInfoAttr('')

    def make_tarball(*args, **kwargs):
        tarinfo_orig = tarfile.TarFile.tarinfo
        try:
            tarfile.time = Time()
            tarfile.TarFile.tarinfo = TarInfo
            return archive_util.make_tarball(*args, **kwargs)
        finally:
            tarfile.time = time
            tarfile.TarFile.tarinfo = tarinfo_orig

    archive_util.ARCHIVE_FORMATS['gztar'] = (
        make_tarball, *archive_util.ARCHIVE_FORMATS['gztar'][1:],
    )

################################################################################
# Customized commands
################################################################################




def create_version(directory: str):
    """Create version.py based on VERSION_INFO."""
    version_file_path = os.path.join(directory, "onnx", "version.py")
    os.makedirs(os.path.dirname(version_file_path), exist_ok=True)

    with open(version_file_path, "w", encoding="utf-8") as f:
        f.write(
            textwrap.dedent(
                f"""\
                # This file is generated by setup.py. DO NOT EDIT!


                version = "{VERSION_INFO["version"]}"
                git_version = "{VERSION_INFO["git_version"]}"
                """
            )
        )


class CmakeBuild(setuptools.Command):
    """Compiles everything when `python setup.py build` is run using cmake.

    Custom args can be passed to cmake by specifying the `CMAKE_ARGS`
    environment variable.

    The number of CPUs used by `make` can be specified by passing `-j<ncpus>`
    to `setup.py build`.  By default all CPUs are used.
    """

    user_options: ClassVar[list] = [
        ("jobs=", "j", "Specifies the number of jobs to use with make")
    ]
    jobs: None | str | int = None

    def initialize_options(self):
        self.jobs = None

    def finalize_options(self):
        self.set_undefined_options("build", ("parallel", "jobs"))
        if self.jobs is None and os.getenv("MAX_JOBS") is not None:
            self.jobs = os.getenv("MAX_JOBS")
        if self.jobs is None:
            self.jobs = multiprocessing.cpu_count()

    def run(self):
        assert CMAKE, "Could not find cmake in PATH"

        os.makedirs(CMAKE_BUILD_DIR, exist_ok=True)

        with cd(CMAKE_BUILD_DIR):
            build_type = "Release"
            # configure
            cmake_args = [
                CMAKE,
                f"-DPython3_EXECUTABLE={get_python_execute()}",
                "-DONNX_BUILD_PYTHON=ON",
                f"-DONNX_NAMESPACE={ONNX_NAMESPACE}",
            ]
            if COVERAGE:
                cmake_args.append("-DONNX_COVERAGE=ON")
            if COVERAGE or DEBUG:
                # in order to get accurate coverage information, the
                # build needs to turn off optimizations
                build_type = "Debug"
            cmake_args.append(f"-DCMAKE_BUILD_TYPE={build_type}")
            if WINDOWS:
                if USE_MSVC_STATIC_RUNTIME:
                    cmake_args.append("-DONNX_USE_MSVC_STATIC_RUNTIME=ON")
                if platform.architecture()[0] == "64bit":
                    if "arm" in platform.machine().lower():
                        cmake_args.extend(["-A", "ARM64"])
                    else:
                        cmake_args.extend(["-A", "x64", "-T", "host=x64"])
                else:  # noqa: PLR5501
                    if "arm" in platform.machine().lower():
                        cmake_args.extend(["-A", "ARM"])
                    else:
                        cmake_args.extend(["-A", "Win32", "-T", "host=x86"])
            if ONNX_ML:
                cmake_args.append("-DONNX_ML=1")
            if ONNX_VERIFY_PROTO3:
                cmake_args.append("-DONNX_VERIFY_PROTO3=1")
            if ONNX_BUILD_TESTS:
                cmake_args.append("-DONNX_BUILD_TESTS=ON")
            if ONNX_DISABLE_EXCEPTIONS:
                cmake_args.append("-DONNX_DISABLE_EXCEPTIONS=ON")
            if ONNX_DISABLE_STATIC_REGISTRATION:
                cmake_args.append("-DONNX_DISABLE_STATIC_REGISTRATION=ON")
            if "CMAKE_ARGS" in os.environ:
                extra_cmake_args = shlex.split(os.environ["CMAKE_ARGS"])
                # prevent crossfire with downstream scripts
                del os.environ["CMAKE_ARGS"]
                logging.info("Extra cmake args: %s", extra_cmake_args)
                cmake_args.extend(extra_cmake_args)
            cmake_args.append(TOP_DIR)
            logging.info("Using cmake args: %s", cmake_args)
            if "-DONNX_DISABLE_EXCEPTIONS=ON" in cmake_args:
                raise RuntimeError(
                    "-DONNX_DISABLE_EXCEPTIONS=ON option is only available for c++ builds. Python binding require exceptions to be enabled."
                )
            subprocess.check_call(cmake_args)

            build_args = [
                CMAKE,
                "--build",
                os.curdir,
                f"--parallel {self.jobs}",
            ]
            if WINDOWS:
                build_args.extend(
                    [
                        "--config",
                        build_type,
                        "--verbose",
                    ]
                )
            subprocess.check_call(build_args)


class BuildPy(setuptools.command.build_py.build_py):
    def run(self):
        if self.editable_mode:
            dst_dir = TOP_DIR
        else:
            dst_dir = self.build_lib
        create_version(dst_dir)
        return super().run()


class Develop(setuptools.command.develop.develop):
    def run(self):
        create_version(TOP_DIR)
        return super().run()


class BuildExt(setuptools.command.build_ext.build_ext):
    def run(self):
        self.run_command("cmake_build")
        return super().run()

    def build_extensions(self):
        # We override this method entirely because the actual building is done
        # by cmake_build. Here we just copy the built extensions to the final
        # destination.
        build_lib = self.build_lib
        extension_dst_dir = os.path.join(build_lib, "onnx")
        os.makedirs(extension_dst_dir, exist_ok=True)

        for ext in self.extensions:
            fullname = self.get_ext_fullname(ext.name)
            filename = os.path.basename(self.get_ext_filename(fullname))

            lib_dir = CMAKE_BUILD_DIR
            if WINDOWS:
                # Windows compiled extensions are stored in Release/Debug subfolders
                debug_lib_dir = os.path.join(CMAKE_BUILD_DIR, "Debug")
                release_lib_dir = os.path.join(CMAKE_BUILD_DIR, "Release")
                if os.path.exists(debug_lib_dir):
                    lib_dir = debug_lib_dir
                elif os.path.exists(release_lib_dir):
                    lib_dir = release_lib_dir
            src = os.path.join(lib_dir, filename)
            dst = os.path.join(extension_dst_dir, filename)
            self.copy_file(src, dst)

        # Copy over the generated python files to build/source dir depending on editable mode
        if self.editable_mode:
            dst_dir = TOP_DIR
        else:
            dst_dir = build_lib

        generated_py_files = glob.glob(os.path.join(CMAKE_BUILD_DIR, "onnx", "*.py"))
        generated_pyi_files = glob.glob(os.path.join(CMAKE_BUILD_DIR, "onnx", "*.pyi"))
        assert generated_py_files, "Bug: No generated python files found"
        assert generated_pyi_files, "Bug: No generated python stubs found"
        for src in (*generated_py_files, *generated_pyi_files):
            dst = os.path.join(dst_dir, os.path.relpath(src, CMAKE_BUILD_DIR))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            self.copy_file(src, dst)


CMD_CLASS = {
    "cmake_build": CmakeBuild,
    "build_py": BuildPy,
    "build_ext": BuildExt,
    "develop": Develop,
}

################################################################################
# Extensions
################################################################################

EXT_MODULES = [setuptools.Extension(name="onnx.onnx_cpp2py_export", sources=[])]


################################################################################
# Final
################################################################################

setuptools.setup(
    ext_modules=EXT_MODULES,
    cmdclass=CMD_CLASS,
    version=VERSION_INFO["version"],
    options=(
        {"bdist_wheel": {"plat_name": ONNX_WHEEL_PLATFORM_NAME}}
        if ONNX_WHEEL_PLATFORM_NAME is not None
        else {}
    ),
)
