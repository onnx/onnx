# SPDX-License-Identifier: Apache-2.0

import glob
import multiprocessing
import os
import platform
import shlex
import subprocess
import sys
from collections import namedtuple
from contextlib import contextmanager
from datetime import date
from distutils import log, sysconfig
from distutils.spawn import find_executable
from textwrap import dedent

import setuptools
import setuptools.command.build_ext
import setuptools.command.build_py
import setuptools.command.develop

TOP_DIR = os.path.realpath(os.path.dirname(__file__))
SRC_DIR = os.path.join(TOP_DIR, "onnx")
TP_DIR = os.path.join(TOP_DIR, "third_party")
CMAKE_BUILD_DIR = os.path.join(TOP_DIR, ".setuptools-cmake-build")
PACKAGE_NAME = "onnx"

WINDOWS = os.name == "nt"

CMAKE = find_executable("cmake3") or find_executable("cmake")
MAKE = find_executable("make")

install_requires = []
setup_requires = []
tests_require = []
extras_require = {}

################################################################################
# Global variables for controlling the build variant
################################################################################

# Default value is set to TRUE\1 to keep the settings same as the current ones.
# However going forward the recommended way to is to set this to False\0
ONNX_ML = not bool(os.getenv("ONNX_ML") == "0")
ONNX_VERIFY_PROTO3 = bool(os.getenv("ONNX_VERIFY_PROTO3") == "1")
ONNX_NAMESPACE = os.getenv("ONNX_NAMESPACE", "onnx")
ONNX_BUILD_TESTS = bool(os.getenv("ONNX_BUILD_TESTS") == "1")
ONNX_DISABLE_EXCEPTIONS = bool(os.getenv("ONNX_DISABLE_EXCEPTIONS") == "1")

USE_MSVC_STATIC_RUNTIME = bool(os.getenv("USE_MSVC_STATIC_RUNTIME", "0") == "1")
DEBUG = bool(os.getenv("DEBUG", "0") == "1")
COVERAGE = bool(os.getenv("COVERAGE", "0") == "1")

################################################################################
# Version
################################################################################

try:
    git_version = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=TOP_DIR)
        .decode("ascii")
        .strip()
    )
except (OSError, subprocess.CalledProcessError):
    git_version = None

with open(os.path.join(TOP_DIR, "VERSION_NUMBER")) as version_file:
    VERSION_NUMBER = version_file.read().strip()
    if "--weekly_build" in sys.argv:
        today_number = date.today().strftime("%Y%m%d")
        VERSION_NUMBER += ".dev" + today_number
        PACKAGE_NAME = "onnx-weekly"
        sys.argv.remove("--weekly_build")
    VersionInfo = namedtuple("VersionInfo", ["version", "git_version"])(
        version=VERSION_NUMBER, git_version=git_version
    )

################################################################################
# Pre Check
################################################################################

assert CMAKE, "Could not find cmake executable!"

################################################################################
# Utilities
################################################################################


@contextmanager
def cd(path):
    if not os.path.isabs(path):
        raise RuntimeError(f"Can only cd to absolute path, got: {path}")
    orig_path = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(orig_path)


################################################################################
# Customized commands
################################################################################


class ONNXCommand(setuptools.Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass


class create_version(ONNXCommand):
    def run(self):
        with open(os.path.join(SRC_DIR, "version.py"), "w") as f:
            f.write(
                dedent(
                    """\
            # This file is generated by setup.py. DO NOT EDIT!


            version = "{version}"
            git_version = "{git_version}"
            """.format(
                        **dict(VersionInfo._asdict())
                    )
                )
            )


class cmake_build(setuptools.Command):
    """
    Compiles everything when `python setup.py build` is run using cmake.

    Custom args can be passed to cmake by specifying the `CMAKE_ARGS`
    environment variable.

    The number of CPUs used by `make` can be specified by passing `-j<ncpus>`
    to `setup.py build`.  By default all CPUs are used.
    """

    user_options = [("jobs=", "j", "Specifies the number of jobs to use with make")]

    built = False

    def initialize_options(self):
        self.jobs = None

    def finalize_options(self):
        self.set_undefined_options("build", ("parallel", "jobs"))
        if self.jobs is None and os.getenv("MAX_JOBS") is not None:
            self.jobs = os.getenv("MAX_JOBS")
        self.jobs = multiprocessing.cpu_count() if self.jobs is None else int(self.jobs)

    def run(self):
        if cmake_build.built:
            return
        cmake_build.built = True
        if not os.path.exists(CMAKE_BUILD_DIR):
            os.makedirs(CMAKE_BUILD_DIR)

        with cd(CMAKE_BUILD_DIR):
            build_type = "Release"
            # configure
            cmake_args = [
                CMAKE,
                f"-DPYTHON_INCLUDE_DIR={sysconfig.get_python_inc()}",
                f"-DPYTHON_EXECUTABLE={sys.executable}",
                "-DBUILD_ONNX_PYTHON=ON",
                "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                f"-DONNX_NAMESPACE={ONNX_NAMESPACE}",
                f"-DPY_EXT_SUFFIX={sysconfig.get_config_var('EXT_SUFFIX') or ''}",
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
                    cmake_args.extend(["-A", "x64", "-T", "host=x64"])
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
            if "CMAKE_ARGS" in os.environ:
                extra_cmake_args = shlex.split(os.environ["CMAKE_ARGS"])
                # prevent crossfire with downstream scripts
                del os.environ["CMAKE_ARGS"]
                log.info(f"Extra cmake args: {extra_cmake_args}")
                cmake_args.extend(extra_cmake_args)
            cmake_args.append(TOP_DIR)
            log.info(f"Using cmake args: {cmake_args}")
            if "-DONNX_DISABLE_EXCEPTIONS=ON" in cmake_args:
                raise RuntimeError(
                    "-DONNX_DISABLE_EXCEPTIONS=ON option is only available for c++ builds. Python binding require exceptions to be enabled."
                )
            subprocess.check_call(cmake_args)

            build_args = [CMAKE, "--build", os.curdir]
            if WINDOWS:
                build_args.extend(["--config", build_type])
                build_args.extend(["--", f"/maxcpucount:{self.jobs}"])
            else:
                build_args.extend(["--", "-j", str(self.jobs)])
            subprocess.check_call(build_args)


class build_py(setuptools.command.build_py.build_py):
    def run(self):
        self.run_command("create_version")
        self.run_command("cmake_build")

        generated_python_files = glob.glob(
            os.path.join(CMAKE_BUILD_DIR, "onnx", "*.py")
        ) + glob.glob(os.path.join(CMAKE_BUILD_DIR, "onnx", "*.pyi"))

        for src in generated_python_files:
            dst = os.path.join(TOP_DIR, os.path.relpath(src, CMAKE_BUILD_DIR))
            self.copy_file(src, dst)

        return setuptools.command.build_py.build_py.run(self)


class develop(setuptools.command.develop.develop):
    def run(self):
        self.run_command("build_py")
        setuptools.command.develop.develop.run(self)


class build_ext(setuptools.command.build_ext.build_ext):
    def run(self):
        self.run_command("cmake_build")
        setuptools.command.build_ext.build_ext.run(self)

    def build_extensions(self):
        for ext in self.extensions:
            fullname = self.get_ext_fullname(ext.name)
            filename = os.path.basename(self.get_ext_filename(fullname))

            lib_path = CMAKE_BUILD_DIR
            if os.name == "nt":
                debug_lib_dir = os.path.join(lib_path, "Debug")
                release_lib_dir = os.path.join(lib_path, "Release")
                if os.path.exists(debug_lib_dir):
                    lib_path = debug_lib_dir
                elif os.path.exists(release_lib_dir):
                    lib_path = release_lib_dir
            src = os.path.join(lib_path, filename)
            dst = os.path.join(os.path.realpath(self.build_lib), "onnx", filename)
            self.copy_file(src, dst)


class mypy_type_check(ONNXCommand):
    description = "Run MyPy type checker"

    def run(self):
        """Run command."""
        onnx_script = os.path.realpath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "tools/mypy-onnx.py"
            )
        )
        returncode = subprocess.call([sys.executable, onnx_script])
        sys.exit(returncode)


cmdclass = {
    "create_version": create_version,
    "cmake_build": cmake_build,
    "build_py": build_py,
    "develop": develop,
    "build_ext": build_ext,
    "typecheck": mypy_type_check,
}

################################################################################
# Extensions
################################################################################

ext_modules = [setuptools.Extension(name="onnx.onnx_cpp2py_export", sources=[])]

################################################################################
# Packages
################################################################################

# Add package directories here if you want to package them with the source
include_dirs = [
    "onnx.backend.test.cpp*",
    "onnx.backend.test.data.*",
    "onnx.common",
    "onnx.defs.*",
    "onnx.examples*",
    "onnx.shape_inference",
    "onnx.test.cpp",
    "onnx.version_converter*",
]

packages = setuptools.find_packages() + setuptools.find_namespace_packages(
    include=include_dirs
)

requirements_file = "requirements.txt"
requirements_path = os.path.join(os.getcwd(), requirements_file)
if not os.path.exists(requirements_path):
    this = os.path.dirname(__file__)
    requirements_path = os.path.join(this, requirements_file)
if not os.path.exists(requirements_path):
    raise FileNotFoundError("Unable to find " + requirements_file)
with open(requirements_path) as f:
    install_requires = f.read().splitlines()

################################################################################
# Test
################################################################################

setup_requires.append("pytest-runner")
tests_require.append("pytest")
tests_require.append("nbval")
tests_require.append("tabulate")

extras_require["lint"] = [
    "clang-format==13.0.0",
    "flake8==5.0.2",
    "mypy==0.782",
    "types-protobuf==3.18.4",
    "black>=22.3",
    "isort[colors]>=5.10",
]

################################################################################
# Final
################################################################################

setuptools.setup(
    name=PACKAGE_NAME,
    version=VersionInfo.version,
    description="Open Neural Network Exchange",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    packages=packages,
    license="Apache License v2.0",
    include_package_data=True,
    package_data={"onnx": ["py.typed", "*.pyi"]},
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    extras_require=extras_require,
    author="ONNX",
    author_email="onnx-technical-discuss@lists.lfaidata.foundation",
    url="https://github.com/onnx/onnx",
    entry_points={
        "console_scripts": [
            "check-model = onnx.bin.checker:check_model",
            "check-node = onnx.bin.checker:check_node",
            "backend-test-tools = onnx.backend.test.cmd_tools:main",
        ]
    },
)
