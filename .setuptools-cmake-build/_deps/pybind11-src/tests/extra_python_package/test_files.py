from __future__ import annotations

import contextlib
import os
import string
import subprocess
import sys
import tarfile
import zipfile

# These tests must be run explicitly
# They require CMake 3.15+ (--install)

DIR = os.path.abspath(os.path.dirname(__file__))
MAIN_DIR = os.path.dirname(os.path.dirname(DIR))

PKGCONFIG = """\
prefix=${{pcfiledir}}/../../
includedir=${{prefix}}/include

Name: pybind11
Description: Seamless operability between C++11 and Python
Version: {VERSION}
Cflags: -I${{includedir}}
"""


main_headers = {
    "include/pybind11/attr.h",
    "include/pybind11/buffer_info.h",
    "include/pybind11/cast.h",
    "include/pybind11/chrono.h",
    "include/pybind11/common.h",
    "include/pybind11/complex.h",
    "include/pybind11/eigen.h",
    "include/pybind11/embed.h",
    "include/pybind11/eval.h",
    "include/pybind11/functional.h",
    "include/pybind11/gil.h",
    "include/pybind11/gil_safe_call_once.h",
    "include/pybind11/iostream.h",
    "include/pybind11/numpy.h",
    "include/pybind11/operators.h",
    "include/pybind11/options.h",
    "include/pybind11/pybind11.h",
    "include/pybind11/pytypes.h",
    "include/pybind11/stl.h",
    "include/pybind11/stl_bind.h",
    "include/pybind11/type_caster_pyobject_ptr.h",
    "include/pybind11/typing.h",
}

detail_headers = {
    "include/pybind11/detail/class.h",
    "include/pybind11/detail/common.h",
    "include/pybind11/detail/cpp_conduit.h",
    "include/pybind11/detail/descr.h",
    "include/pybind11/detail/init.h",
    "include/pybind11/detail/internals.h",
    "include/pybind11/detail/type_caster_base.h",
    "include/pybind11/detail/typeid.h",
    "include/pybind11/detail/value_and_holder.h",
    "include/pybind11/detail/exception_translation.h",
}

eigen_headers = {
    "include/pybind11/eigen/common.h",
    "include/pybind11/eigen/matrix.h",
    "include/pybind11/eigen/tensor.h",
}

stl_headers = {
    "include/pybind11/stl/filesystem.h",
}

cmake_files = {
    "share/cmake/pybind11/FindPythonLibsNew.cmake",
    "share/cmake/pybind11/pybind11Common.cmake",
    "share/cmake/pybind11/pybind11Config.cmake",
    "share/cmake/pybind11/pybind11ConfigVersion.cmake",
    "share/cmake/pybind11/pybind11GuessPythonExtSuffix.cmake",
    "share/cmake/pybind11/pybind11NewTools.cmake",
    "share/cmake/pybind11/pybind11Targets.cmake",
    "share/cmake/pybind11/pybind11Tools.cmake",
}

pkgconfig_files = {
    "share/pkgconfig/pybind11.pc",
}

py_files = {
    "__init__.py",
    "__main__.py",
    "_version.py",
    "commands.py",
    "py.typed",
    "setup_helpers.py",
}

headers = main_headers | detail_headers | eigen_headers | stl_headers
src_files = headers | cmake_files | pkgconfig_files
all_files = src_files | py_files


sdist_files = {
    "pybind11",
    "pybind11/include",
    "pybind11/include/pybind11",
    "pybind11/include/pybind11/detail",
    "pybind11/include/pybind11/eigen",
    "pybind11/include/pybind11/stl",
    "pybind11/share",
    "pybind11/share/cmake",
    "pybind11/share/cmake/pybind11",
    "pybind11/share/pkgconfig",
    "pyproject.toml",
    "setup.cfg",
    "setup.py",
    "LICENSE",
    "MANIFEST.in",
    "README.rst",
    "PKG-INFO",
    "SECURITY.md",
}

local_sdist_files = {
    ".egg-info",
    ".egg-info/PKG-INFO",
    ".egg-info/SOURCES.txt",
    ".egg-info/dependency_links.txt",
    ".egg-info/not-zip-safe",
    ".egg-info/top_level.txt",
}


def read_tz_file(tar: tarfile.TarFile, name: str) -> bytes:
    start = tar.getnames()[0] + "/"
    inner_file = tar.extractfile(tar.getmember(f"{start}{name}"))
    assert inner_file
    with contextlib.closing(inner_file) as f:
        return f.read()


def normalize_line_endings(value: bytes) -> bytes:
    return value.replace(os.linesep.encode("utf-8"), b"\n")


def test_build_sdist(monkeypatch, tmpdir):
    monkeypatch.chdir(MAIN_DIR)

    subprocess.run(
        [sys.executable, "-m", "build", "--sdist", f"--outdir={tmpdir}"], check=True
    )

    (sdist,) = tmpdir.visit("*.tar.gz")

    with tarfile.open(str(sdist), "r:gz") as tar:
        start = tar.getnames()[0] + "/"
        version = start[9:-1]
        simpler = {n.split("/", 1)[-1] for n in tar.getnames()[1:]}

        setup_py = read_tz_file(tar, "setup.py")
        pyproject_toml = read_tz_file(tar, "pyproject.toml")
        pkgconfig = read_tz_file(tar, "pybind11/share/pkgconfig/pybind11.pc")
        cmake_cfg = read_tz_file(
            tar, "pybind11/share/cmake/pybind11/pybind11Config.cmake"
        )

    assert (
        'set(pybind11_INCLUDE_DIR "${PACKAGE_PREFIX_DIR}/include")'
        in cmake_cfg.decode("utf-8")
    )

    files = {f"pybind11/{n}" for n in all_files}
    files |= sdist_files
    files |= {f"pybind11{n}" for n in local_sdist_files}
    files.add("pybind11.egg-info/entry_points.txt")
    files.add("pybind11.egg-info/requires.txt")
    assert simpler == files

    with open(os.path.join(MAIN_DIR, "tools", "setup_main.py.in"), "rb") as f:
        contents = (
            string.Template(f.read().decode("utf-8"))
            .substitute(version=version, extra_cmd="")
            .encode("utf-8")
        )
    assert setup_py == contents

    with open(os.path.join(MAIN_DIR, "tools", "pyproject.toml"), "rb") as f:
        contents = f.read()
    assert pyproject_toml == contents

    simple_version = ".".join(version.split(".")[:3])
    pkgconfig_expected = PKGCONFIG.format(VERSION=simple_version).encode("utf-8")
    assert normalize_line_endings(pkgconfig) == pkgconfig_expected


def test_build_global_dist(monkeypatch, tmpdir):
    monkeypatch.chdir(MAIN_DIR)
    monkeypatch.setenv("PYBIND11_GLOBAL_SDIST", "1")
    subprocess.run(
        [sys.executable, "-m", "build", "--sdist", "--outdir", str(tmpdir)], check=True
    )

    (sdist,) = tmpdir.visit("*.tar.gz")

    with tarfile.open(str(sdist), "r:gz") as tar:
        start = tar.getnames()[0] + "/"
        version = start[16:-1]
        simpler = {n.split("/", 1)[-1] for n in tar.getnames()[1:]}

        setup_py = read_tz_file(tar, "setup.py")
        pyproject_toml = read_tz_file(tar, "pyproject.toml")
        pkgconfig = read_tz_file(tar, "pybind11/share/pkgconfig/pybind11.pc")
        cmake_cfg = read_tz_file(
            tar, "pybind11/share/cmake/pybind11/pybind11Config.cmake"
        )

    assert (
        'set(pybind11_INCLUDE_DIR "${PACKAGE_PREFIX_DIR}/include")'
        in cmake_cfg.decode("utf-8")
    )

    files = {f"pybind11/{n}" for n in all_files}
    files |= sdist_files
    files |= {f"pybind11_global{n}" for n in local_sdist_files}
    assert simpler == files

    with open(os.path.join(MAIN_DIR, "tools", "setup_global.py.in"), "rb") as f:
        contents = (
            string.Template(f.read().decode())
            .substitute(version=version, extra_cmd="")
            .encode("utf-8")
        )
        assert setup_py == contents

    with open(os.path.join(MAIN_DIR, "tools", "pyproject.toml"), "rb") as f:
        contents = f.read()
        assert pyproject_toml == contents

    simple_version = ".".join(version.split(".")[:3])
    pkgconfig_expected = PKGCONFIG.format(VERSION=simple_version).encode("utf-8")
    assert normalize_line_endings(pkgconfig) == pkgconfig_expected


def tests_build_wheel(monkeypatch, tmpdir):
    monkeypatch.chdir(MAIN_DIR)

    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", ".", "-w", str(tmpdir)], check=True
    )

    (wheel,) = tmpdir.visit("*.whl")

    files = {f"pybind11/{n}" for n in all_files}
    files |= {
        "dist-info/LICENSE",
        "dist-info/METADATA",
        "dist-info/RECORD",
        "dist-info/WHEEL",
        "dist-info/entry_points.txt",
        "dist-info/top_level.txt",
    }

    with zipfile.ZipFile(str(wheel)) as z:
        names = z.namelist()

    trimmed = {n for n in names if "dist-info" not in n}
    trimmed |= {f"dist-info/{n.split('/', 1)[-1]}" for n in names if "dist-info" in n}
    assert files == trimmed


def tests_build_global_wheel(monkeypatch, tmpdir):
    monkeypatch.chdir(MAIN_DIR)
    monkeypatch.setenv("PYBIND11_GLOBAL_SDIST", "1")

    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", ".", "-w", str(tmpdir)], check=True
    )

    (wheel,) = tmpdir.visit("*.whl")

    files = {f"data/data/{n}" for n in src_files}
    files |= {f"data/headers/{n[8:]}" for n in headers}
    files |= {
        "dist-info/LICENSE",
        "dist-info/METADATA",
        "dist-info/WHEEL",
        "dist-info/top_level.txt",
        "dist-info/RECORD",
    }

    with zipfile.ZipFile(str(wheel)) as z:
        names = z.namelist()

    beginning = names[0].split("/", 1)[0].rsplit(".", 1)[0]
    trimmed = {n[len(beginning) + 1 :] for n in names}

    assert files == trimmed
