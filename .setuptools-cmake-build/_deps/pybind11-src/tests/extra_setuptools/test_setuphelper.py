from __future__ import annotations

import os
import subprocess
import sys
from textwrap import dedent

import pytest

DIR = os.path.abspath(os.path.dirname(__file__))
MAIN_DIR = os.path.dirname(os.path.dirname(DIR))
WIN = sys.platform.startswith("win32") or sys.platform.startswith("cygwin")


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("std", [11, 0])
def test_simple_setup_py(monkeypatch, tmpdir, parallel, std):
    monkeypatch.chdir(tmpdir)
    monkeypatch.syspath_prepend(MAIN_DIR)

    (tmpdir / "setup.py").write_text(
        dedent(
            f"""\
            import sys
            sys.path.append({MAIN_DIR!r})

            from setuptools import setup, Extension
            from pybind11.setup_helpers import build_ext, Pybind11Extension

            std = {std}

            ext_modules = [
                Pybind11Extension(
                    "simple_setup",
                    sorted(["main.cpp"]),
                    cxx_std=std,
                ),
            ]

            cmdclass = dict()
            if std == 0:
                cmdclass["build_ext"] = build_ext


            parallel = {parallel}
            if parallel:
                from pybind11.setup_helpers import ParallelCompile
                ParallelCompile().install()

            setup(
                name="simple_setup_package",
                cmdclass=cmdclass,
                ext_modules=ext_modules,
            )
            """
        ),
        encoding="ascii",
    )

    (tmpdir / "main.cpp").write_text(
        dedent(
            """\
            #include <pybind11/pybind11.h>

            int f(int x) {
                return x * 3;
            }
            PYBIND11_MODULE(simple_setup, m) {
                m.def("f", &f);
            }
            """
        ),
        encoding="ascii",
    )

    out = subprocess.check_output(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
    )
    if not WIN:
        assert b"-g0" in out
    out = subprocess.check_output(
        [sys.executable, "setup.py", "build_ext", "--inplace", "--force"],
        env=dict(os.environ, CFLAGS="-g"),
    )
    if not WIN:
        assert b"-g0" not in out

    # Debug helper printout, normally hidden
    print(out)
    for item in tmpdir.listdir():
        print(item.basename)

    assert (
        len([f for f in tmpdir.listdir() if f.basename.startswith("simple_setup")]) == 1
    )
    assert len(list(tmpdir.listdir())) == 4  # two files + output + build_dir

    (tmpdir / "test.py").write_text(
        dedent(
            """\
            import simple_setup
            assert simple_setup.f(3) == 9
            """
        ),
        encoding="ascii",
    )

    subprocess.check_call(
        [sys.executable, "test.py"], stdout=sys.stdout, stderr=sys.stderr
    )


def test_intree_extensions(monkeypatch, tmpdir):
    monkeypatch.syspath_prepend(MAIN_DIR)

    from pybind11.setup_helpers import intree_extensions

    monkeypatch.chdir(tmpdir)
    root = tmpdir
    root.ensure_dir()
    subdir = root / "dir"
    subdir.ensure_dir()
    src = subdir / "ext.cpp"
    src.ensure()
    relpath = src.relto(tmpdir)
    (ext,) = intree_extensions([relpath])
    assert ext.name == "ext"
    subdir.ensure("__init__.py")
    (ext,) = intree_extensions([relpath])
    assert ext.name == "dir.ext"


def test_intree_extensions_package_dir(monkeypatch, tmpdir):
    monkeypatch.syspath_prepend(MAIN_DIR)

    from pybind11.setup_helpers import intree_extensions

    monkeypatch.chdir(tmpdir)
    root = tmpdir / "src"
    root.ensure_dir()
    subdir = root / "dir"
    subdir.ensure_dir()
    src = subdir / "ext.cpp"
    src.ensure()
    (ext,) = intree_extensions([src.relto(tmpdir)], package_dir={"": "src"})
    assert ext.name == "dir.ext"
    (ext,) = intree_extensions([src.relto(tmpdir)], package_dir={"foo": "src"})
    assert ext.name == "foo.dir.ext"
    subdir.ensure("__init__.py")
    (ext,) = intree_extensions([src.relto(tmpdir)], package_dir={"": "src"})
    assert ext.name == "dir.ext"
    (ext,) = intree_extensions([src.relto(tmpdir)], package_dir={"foo": "src"})
    assert ext.name == "foo.dir.ext"
