#!/usr/bin/env python3

# Setup script for PyPI; use CMakeFile.txt to build extension modules
from __future__ import annotations

import contextlib
import os
import re
import shutil
import string
import subprocess
import sys
from collections.abc import Generator
from pathlib import Path
from tempfile import TemporaryDirectory

import setuptools.command.sdist

DIR = Path(__file__).parent.absolute()
VERSION_REGEX = re.compile(
    r"^\s*#\s*define\s+PYBIND11_VERSION_([A-Z]+)\s+(.*)$", re.MULTILINE
)
VERSION_FILE = Path("pybind11/_version.py")
COMMON_FILE = Path("include/pybind11/detail/common.h")


def build_expected_version_hex(matches: dict[str, str]) -> str:
    patch_level_serial = matches["PATCH"]
    serial = None
    major = int(matches["MAJOR"])
    minor = int(matches["MINOR"])
    flds = patch_level_serial.split(".")
    if flds:
        patch = int(flds[0])
        if len(flds) == 1:
            level = "0"
            serial = 0
        elif len(flds) == 2:
            level_serial = flds[1]
            for level in ("a", "b", "c", "dev"):
                if level_serial.startswith(level):
                    serial = int(level_serial[len(level) :])
                    break
    if serial is None:
        msg = f'Invalid PYBIND11_VERSION_PATCH: "{patch_level_serial}"'
        raise RuntimeError(msg)
    version_hex_str = f"{major:02x}{minor:02x}{patch:02x}{level[:1]}{serial:x}"
    return f"0x{version_hex_str.upper()}"


# PYBIND11_GLOBAL_SDIST will build a different sdist, with the python-headers
# files, and the sys.prefix files (CMake and headers).

global_sdist = os.environ.get("PYBIND11_GLOBAL_SDIST", False)

setup_py = Path(
    "tools/setup_global.py.in" if global_sdist else "tools/setup_main.py.in"
)
extra_cmd = 'cmdclass["sdist"] = SDist\n'

to_src = (
    (Path("pyproject.toml"), Path("tools/pyproject.toml")),
    (Path("setup.py"), setup_py),
)


# Read the listed version
loc: dict[str, str] = {}
code = compile(VERSION_FILE.read_text(encoding="utf-8"), "pybind11/_version.py", "exec")
exec(code, loc)
version = loc["__version__"]

# Verify that the version matches the one in C++
matches = dict(VERSION_REGEX.findall(COMMON_FILE.read_text(encoding="utf8")))
cpp_version = "{MAJOR}.{MINOR}.{PATCH}".format(**matches)
if version != cpp_version:
    msg = f"Python version {version} does not match C++ version {cpp_version}!"
    raise RuntimeError(msg)

version_hex = matches.get("HEX", "MISSING")
exp_version_hex = build_expected_version_hex(matches)
if version_hex != exp_version_hex:
    msg = f"PYBIND11_VERSION_HEX {version_hex} does not match expected value {exp_version_hex}!"
    raise RuntimeError(msg)


# TODO: use literals & overload (typing extensions or Python 3.8)
def get_and_replace(filename: Path, binary: bool = False, **opts: str) -> bytes | str:
    if binary:
        contents = filename.read_bytes()
        return string.Template(contents.decode()).substitute(opts).encode()

    return string.Template(filename.read_text()).substitute(opts)


# Use our input files instead when making the SDist (and anything that depends
# on it, like a wheel)
class SDist(setuptools.command.sdist.sdist):
    def make_release_tree(self, base_dir: str, files: list[str]) -> None:
        super().make_release_tree(base_dir, files)

        for to, src in to_src:
            txt = get_and_replace(src, binary=True, version=version, extra_cmd="")

            dest = Path(base_dir) / to

            # This is normally linked, so unlink before writing!
            dest.unlink()
            dest.write_bytes(txt)  # type: ignore[arg-type]


# Remove the CMake install directory when done
@contextlib.contextmanager
def remove_output(*sources: str) -> Generator[None, None, None]:
    try:
        yield
    finally:
        for src in sources:
            shutil.rmtree(src)


with remove_output("pybind11/include", "pybind11/share"):
    # Generate the files if they are not present.
    with TemporaryDirectory() as tmpdir:
        cmd = ["cmake", "-S", ".", "-B", tmpdir] + [
            "-DCMAKE_INSTALL_PREFIX=pybind11",
            "-DBUILD_TESTING=OFF",
            "-DPYBIND11_NOPYTHON=ON",
            "-Dprefix_for_pc_file=${pcfiledir}/../../",
        ]
        if "CMAKE_ARGS" in os.environ:
            fcommand = [
                c
                for c in os.environ["CMAKE_ARGS"].split()
                if "DCMAKE_INSTALL_PREFIX" not in c
            ]
            cmd += fcommand
        subprocess.run(cmd, check=True, cwd=DIR, stdout=sys.stdout, stderr=sys.stderr)
        subprocess.run(
            ["cmake", "--install", tmpdir],
            check=True,
            cwd=DIR,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

    txt = get_and_replace(setup_py, version=version, extra_cmd=extra_cmd)
    code = compile(txt, setup_py, "exec")
    exec(code, {"SDist": SDist})
