# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

# NOTE: Put all metadata in pyproject.toml.
# Set the environment variable `ONNX_PREVIEW_BUILD=1` to build the dev preview release.
from __future__ import annotations

import base64
import contextlib
import csv
import datetime
import glob
import hashlib
import io
import json
import logging
import multiprocessing
import os
import platform
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import textwrap
import zipfile
from typing import ClassVar

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:
    _bdist_wheel = None

import setuptools
import setuptools.command.build_ext
import setuptools.command.build_py
import setuptools.command.develop

TOP_DIR = os.path.realpath(os.path.dirname(__file__))
_onnx_cmake_build_dir = os.getenv("ONNX_CMAKE_BUILD_DIR")
CMAKE_BUILD_DIR = os.path.join(
    TOP_DIR,
    _onnx_cmake_build_dir.strip()
    if _onnx_cmake_build_dir and _onnx_cmake_build_dir.strip()
    else ".setuptools-cmake-build",
)

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
USE_NINJA = os.getenv("USE_NINJA", "1") != "0"
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
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=TOP_DIR)  # noqa: S607
        .decode("ascii")
        .strip()
    )
except (OSError, subprocess.CalledProcessError):
    _git_version = ""

with open(os.path.join(TOP_DIR, "VERSION_NUMBER"), encoding="utf-8") as version_file:
    _version = version_file.read().strip()
    if ONNX_PREVIEW_BUILD:
        # Create the preview build for weekly releases
        todays_date = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%d")
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
    """Get the Python executable path for CMake configuration.

    Prefer sys.executable as it represents the currently running Python.
    Only fall back to directory traversal if sys.executable is invalid.
    """
    if WINDOWS:
        return sys.executable

    # First, check if sys.executable is valid and usable
    if os.path.isfile(sys.executable) and os.access(sys.executable, os.X_OK):
        return sys.executable

    # Fallback: Try to search for Python based on include path
    # This addresses https://github.com/python/cpython/issues/84399
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
            if USE_NINJA and not WINDOWS and shutil.which("ninja"):
                cmake_args.append("-DCMAKE_GENERATOR=Ninja")

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
                extra_cmake_args = shlex.split(
                    os.environ["CMAKE_ARGS"],
                    posix=not WINDOWS,
                )
                # prevent crossfire with downstream scripts
                del os.environ["CMAKE_ARGS"]
                logging.info("Extra cmake args: %s", extra_cmake_args)  # noqa: LOG015
                cmake_args.extend(extra_cmake_args)
            cmake_args.append(TOP_DIR)
            logging.info("Using cmake args: %s", cmake_args)  # noqa: LOG015
            if "-DONNX_DISABLE_EXCEPTIONS=ON" in cmake_args:
                raise RuntimeError(
                    "-DONNX_DISABLE_EXCEPTIONS=ON option is only available for c++ builds. Python binding require exceptions to be enabled."
                )
            subprocess.check_call(cmake_args)  # noqa: S603

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
            subprocess.check_call(build_args)  # noqa: S603


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


def _annotate_sbom_occurrences(sbom_data: bytes, binary_paths: list[str]) -> bytes:
    """Patch each component in the SBOM with evidence.occurrences listing the
    compiled binary files inside the wheel that contain the bundled code.
    """
    if not binary_paths:
        return sbom_data
    bom = json.loads(sbom_data)
    occurrences = [{"location": p} for p in sorted(binary_paths)]
    for comp in bom.get("components", []):
        comp["evidence"] = {"occurrences": occurrences}
    return json.dumps(bom, indent=2).encode("utf-8")


def _inject_sboms_into_wheel(wheel_path: str, sbom_dir: str) -> None:
    """Rewrite a wheel to add CycloneDX SBOMs into its dist-info/sboms/ directory.

    The RECORD file is updated with the correct hashes so pip can verify the
    wheel normally.  The original wheel file is replaced atomically.
    """
    sbom_files = sorted(glob.glob(os.path.join(sbom_dir, "*.cdx.json")))
    if not sbom_files:
        return

    wheel_basename = os.path.basename(wheel_path)
    tmp_path = wheel_path + ".tmp"
    try:
        with (
            zipfile.ZipFile(wheel_path, "r") as src_zf,
            zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as dst_zf,
        ):
            all_infos = src_zf.infolist()
            record_paths = [
                info.filename
                for info in all_infos
                if info.filename.endswith(".dist-info/RECORD")
            ]
            if len(record_paths) != 1:
                raise ValueError(  # noqa: TRY301
                    f"Expected exactly one .dist-info/RECORD in {wheel_basename}, "
                    f"found {len(record_paths)}"
                )
            record_arcname = record_paths[0]
            dist_info_prefix = record_arcname.rsplit("/", 1)[0]
            record_bytes = src_zf.read(record_arcname)

            # Discover compiled extension modules (.so / .pyd) present in the wheel.
            # These are the files that physically contain the statically linked C++ code.
            binary_paths = [
                info.filename
                for info in all_infos
                if info.filename.endswith((".so", ".pyd"))
            ]

            # Parse RECORD, dropping the self-referential row for RECORD itself
            record_rows = [
                row
                for row in csv.reader(io.StringIO(record_bytes.decode("utf-8")))
                if row and row[0] != record_arcname
            ]

            # Prepare SBOM data and compute RECORD entries for them
            sbom_entries: list[tuple[str, bytes]] = []
            for sbom_path in sbom_files:
                with open(sbom_path, "rb") as f:
                    data = f.read()
                data = _annotate_sbom_occurrences(data, binary_paths)
                digest = (
                    base64.urlsafe_b64encode(hashlib.sha256(data).digest())
                    .rstrip(b"=")
                    .decode()
                )
                arcname = f"{dist_info_prefix}/sboms/{os.path.basename(sbom_path)}"
                record_rows.append([arcname, f"sha256={digest}", str(len(data))])
                sbom_entries.append((arcname, data))
                logging.info(  # noqa: LOG015
                    "SBOM to embed: %s (%d bytes)",
                    os.path.basename(sbom_path),
                    len(data),
                )

            # Build updated RECORD content (RECORD entry itself always has empty hash/size)
            record_buf = io.StringIO()
            csv.writer(record_buf, lineterminator="\n").writerows(record_rows)
            record_buf.write(f"{record_arcname},,\n")
            record_content = record_buf.getvalue().encode("utf-8")

            for info in all_infos:
                if info.filename == record_arcname:
                    continue
                dst_zf.writestr(info, src_zf.read(info.filename))
            for arcname, data in sbom_entries:
                dst_zf.writestr(arcname, data)
            dst_zf.writestr(record_arcname, record_content)

        shutil.move(tmp_path, wheel_path)
        logging.info("Embedded SBOMs into %s", wheel_basename)  # noqa: LOG015

    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


if _bdist_wheel is not None:

    class BdistWheelWithSBOM(_bdist_wheel):  # type: ignore[misc,valid-type]
        """bdist_wheel subclass that embeds a CycloneDX SBOM into the dist-info directory.

        The SBOM describes the C++ libraries (protobuf, abseil-cpp, nanobind)
        that are compiled into onnx_cpp2py_export and bundled inside the wheel,
        per PEP 770 / the dist-info/sboms/ specification.

        If SBOM generation fails for any reason the wheel is built normally
        without SBOMs, so the build is never blocked.
        """

        def run(self) -> None:
            existing = set(glob.glob(os.path.join(self.dist_dir, "*.whl")))
            sbom_dir = self._try_generate_sboms()
            super().run()
            if sbom_dir is not None:
                try:
                    new_wheels = sorted(
                        set(glob.glob(os.path.join(self.dist_dir, "*.whl"))) - existing
                    )
                    for wheel_path in new_wheels:
                        _inject_sboms_into_wheel(wheel_path, sbom_dir)
                    sbom_output_dir = os.path.join(TOP_DIR, "dist")
                    os.makedirs(sbom_output_dir, exist_ok=True)
                    for sbom_path in sorted(
                        glob.glob(os.path.join(sbom_dir, "*.cdx.json"))
                    ):
                        shutil.copy2(sbom_path, sbom_output_dir)
                finally:
                    shutil.rmtree(sbom_dir, ignore_errors=True)

        def _try_generate_sboms(self) -> str | None:
            """Return path to a temp dir containing the generated SBOM, or None on failure."""
            tmp = tempfile.mkdtemp(prefix="onnx-sbom-")
            try:
                self._generate_sboms(tmp)
            except Exception as exc:  # noqa: BLE001 — any failure must not block the build
                logging.warning(  # noqa: LOG015
                    "SBOM generation failed (%s); wheel will be built without SBOMs",
                    exc,
                )
                shutil.rmtree(tmp, ignore_errors=True)
                return None
            else:
                return tmp

        def _generate_sboms(self, tmp_dir: str) -> None:
            subject_name = "onnx-weekly" if ONNX_PREVIEW_BUILD else "onnx"
            subject_version = VERSION_INFO["version"]
            subprocess.check_call(  # noqa: S603 — all args are controlled internal values
                [
                    sys.executable,
                    os.path.join(TOP_DIR, "tools", "extract_cmake_fetchcontent.py"),
                    "--cmake",
                    os.path.join(TOP_DIR, "CMakeLists.txt"),
                    "--output",
                    os.path.join(tmp_dir, "onnx-bundled.cdx.json"),
                    "--subject-name",
                    subject_name,
                    "--subject-version",
                    subject_version,
                ]
            )


CMD_CLASS = {
    "cmake_build": CmakeBuild,
    "build_py": BuildPy,
    "build_ext": BuildExt,
    "develop": Develop,
}
if _bdist_wheel is not None:
    CMD_CLASS["bdist_wheel"] = BdistWheelWithSBOM

################################################################################
# Extensions
################################################################################

# Enable limited ABI build
# nanobind supports limited ABI for Python 3.12 and later.
# https://blog.trailofbits.com/2022/11/15/python-wheels-abi-abi3audit/

# 1. The Py_LIMITED_API macro is defined in the extension
# 2. py_limited_api in Extension tags the extension as abi3
# 3. bdist_wheel_options tags the wheel as abi3

NO_GIL = hasattr(sys, "_is_gil_enabled") and not sys._is_gil_enabled()
PY_312_OR_NEWER = sys.version_info >= (3, 12)
USE_LIMITED_API = not NO_GIL and PY_312_OR_NEWER and platform.system() != "FreeBSD"

macros = []
if USE_LIMITED_API:
    macros.append(("Py_LIMITED_API", "0x030C0000"))

EXT_MODULES = [
    setuptools.Extension(
        name="onnx.onnx_cpp2py_export",
        sources=[],
        py_limited_api=USE_LIMITED_API,
        define_macros=macros,
    )
]

################################################################################
# Final
################################################################################

bdist_wheel_options = {}

if USE_LIMITED_API:
    bdist_wheel_options["py_limited_api"] = "cp312"

if ONNX_WHEEL_PLATFORM_NAME is not None:
    bdist_wheel_options["plat_name"] = ONNX_WHEEL_PLATFORM_NAME

setuptools.setup(
    ext_modules=EXT_MODULES,
    cmdclass=CMD_CLASS,
    version=VERSION_INFO["version"],
    options=({"bdist_wheel": bdist_wheel_options}),
)
