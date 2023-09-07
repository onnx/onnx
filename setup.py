# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

# NOTE: Put all metadata in pyproject.toml.
# Set the environment variable `ONNX_PREVIEW_BUILD=1` to build the dev preview release.

import contextlib
import datetime
import glob
import itertools
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

# Customize the wheel plat-name, usually needed for MacOS builds.
# See usage in .github/workflows/release_mac.yml
ONNX_WHEEL_PLATFORM_NAME = os.getenv("ONNX_WHEEL_PLATFORM_NAME")

################################################################################
# Pre Check
################################################################################

assert CMAKE, "Could not find cmake in PATH"

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
        # Create the dev build for weekly releases
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


def get_ext_suffix():
    if sys.version_info < (3, 8) and sys.platform == "win32":
        # Workaround for https://bugs.python.org/issue39825
        # Reference: https://github.com/pytorch/pytorch/commit/4b96fc060b0cb810965b5c8c08bc862a69965667
        import distutils

        return distutils.sysconfig.get_config_var("EXT_SUFFIX")
    return sysconfig.get_config_var("EXT_SUFFIX")


################################################################################
# Customized commands
################################################################################


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

    def initialize_options(self):
        self.jobs = None

    def finalize_options(self):
        self.set_undefined_options("build", ("parallel", "jobs"))
        if self.jobs is None and os.getenv("MAX_JOBS") is not None:
            self.jobs = os.getenv("MAX_JOBS")
        self.jobs = multiprocessing.cpu_count() if self.jobs is None else int(self.jobs)

    def run(self):
        os.makedirs(CMAKE_BUILD_DIR, exist_ok=True)

        with cd(CMAKE_BUILD_DIR):
            build_type = "Release"
            # configure
            cmake_args = [
                CMAKE,
                f"-DPYTHON_INCLUDE_DIR={sysconfig.get_path('include')}",
                f"-DPYTHON_EXECUTABLE={sys.executable}",
                "-DBUILD_ONNX_PYTHON=ON",
                "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                f"-DONNX_NAMESPACE={ONNX_NAMESPACE}",
                f"-DPY_EXT_SUFFIX={get_ext_suffix() or ''}",
            ]
            if COVERAGE:
                cmake_args.append("-DONNX_COVERAGE=ON")
            if COVERAGE or DEBUG:
                # in order to get accurate coverage information, the
                # build needs to turn off optimizations
                build_type = "Debug"
            cmake_args.append(f"-DCMAKE_BUILD_TYPE={build_type}")
            if WINDOWS:
                cmake_args.extend(
                    [
                        # we need to link with libpython on windows, so
                        # passing python version to window in order to
                        # find python in cmake
                        f"-DPY_VERSION={'{}.{}'.format(*sys.version_info[:2])}",
                    ]
                )
                if USE_MSVC_STATIC_RUNTIME:
                    cmake_args.append("-DONNX_USE_MSVC_STATIC_RUNTIME=ON")
                if platform.architecture()[0] == "64bit":
                    if "arm" in platform.machine().lower():
                        cmake_args.extend(["-A", "ARM64"])
                    else:
                        cmake_args.extend(["-A", "x64", "-T", "host=x64"])
                else:
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
            if (
                "PYTHONPATH" in os.environ
                and "pip-build-env" in os.environ["PYTHONPATH"]
            ):
                # When the users use `pip install -e .` to install onnx and
                # the cmake executable is a python entry script, there will be
                # `Fix ModuleNotFoundError: No module named 'cmake'` from the cmake script.
                # This is caused by the additional PYTHONPATH environment variable added by pip,
                # which makes cmake python entry script not able to find correct python cmake packages.
                # Actually, sys.path is well enough for `pip install -e .`.
                # Therefore, we delete the PYTHONPATH variable.
                del os.environ["PYTHONPATH"]
            subprocess.check_call(cmake_args)

            build_args = [CMAKE, "--build", os.curdir]
            if WINDOWS:
                build_args.extend(["--config", build_type])
                build_args.extend(["--", f"/maxcpucount:{self.jobs}"])
            else:
                build_args.extend(["--", "-j", str(self.jobs)])
            subprocess.check_call(build_args)


class BuildPy(setuptools.command.build_py.build_py):
    def run(self):
        self.create_version()
        return super().run()

    def create_version(self):
        # We do not make create_version into its own command because we need to use self.build_lib
        if self.editable_mode:
            dst_dir = TOP_DIR
        else:
            dst_dir = self.build_lib
        version_file_path = os.path.join(dst_dir, "onnx", "version.py")
        os.makedirs(os.path.dirname(version_file_path), exist_ok=True)

        with open(version_file_path, "w", encoding="utf-8") as f:
            f.write(
                textwrap.dedent(
                    f"""\
                    # This file is generated by setup.py. DO NOT EDIT!


                    version = "{VERSION_INFO['version']}"
                    git_version = "{VERSION_INFO['git_version']}"
                    """
                )
            )


class BuildExt(setuptools.command.build_ext.build_ext):
    def run(self):
        self.run_command("cmake_build")
        return super().run()

    def build_extensions(self):
        build_lib = os.path.realpath(self.build_lib)
        dst_dir = os.path.join(build_lib, "onnx")
        os.makedirs(dst_dir, exist_ok=True)
        lib_dir = CMAKE_BUILD_DIR

        for ext in self.extensions:
            fullname = self.get_ext_fullname(ext.name)
            filename = os.path.basename(self.get_ext_filename(fullname))

            if WINDOWS:
                debug_lib_dir = os.path.join(lib_dir, "Debug")
                release_lib_dir = os.path.join(lib_dir, "Release")
                if os.path.exists(debug_lib_dir):
                    lib_dir = debug_lib_dir
                elif os.path.exists(release_lib_dir):
                    lib_dir = release_lib_dir
            src = os.path.join(lib_dir, filename)
            dst = os.path.join(dst_dir, filename)
            self.copy_file(src, dst)

        # Copy over the generated python files to build/source dir depending on editable mode
        if self.editable_mode:
            dst_dir = TOP_DIR
        else:
            dst_dir = build_lib
        generated_python_files = itertools.chain(
            glob.glob(os.path.join(lib_dir, "onnx", "*.py")),
            glob.glob(os.path.join(lib_dir, "onnx", "*.pyi")),
        )
        for src in generated_python_files:
            dst = os.path.join(dst_dir, os.path.relpath(src, lib_dir))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            self.copy_file(src, dst)


CMD_CLASS = {
    "cmake_build": CmakeBuild,
    "build_py": BuildPy,
    "build_ext": BuildExt,
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
    options={"bdist_wheel": {"plat_name": ONNX_WHEEL_PLATFORM_NAME}}
    if ONNX_WHEEL_PLATFORM_NAME is not None
    else {},
)
