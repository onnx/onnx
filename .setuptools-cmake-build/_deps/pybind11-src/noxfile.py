from __future__ import annotations

import argparse

import nox

nox.needs_version = ">=2024.3.2"
nox.options.sessions = ["lint", "tests", "tests_packaging"]
nox.options.default_venv_backend = "uv|virtualenv"


@nox.session(reuse_venv=True)
def lint(session: nox.Session) -> None:
    """
    Lint the codebase (except for clang-format/tidy).
    """
    session.install("pre-commit")
    session.run("pre-commit", "run", "-a", *session.posargs)


@nox.session
def tests(session: nox.Session) -> None:
    """
    Run the tests (requires a compiler).
    """
    tmpdir = session.create_tmp()
    session.install("cmake")
    session.install("-r", "tests/requirements.txt")
    session.run(
        "cmake",
        "-S.",
        f"-B{tmpdir}",
        "-DPYBIND11_WERROR=ON",
        "-DDOWNLOAD_CATCH=ON",
        "-DDOWNLOAD_EIGEN=ON",
        *session.posargs,
    )
    session.run("cmake", "--build", tmpdir)
    session.run("cmake", "--build", tmpdir, "--config=Release", "--target", "check")


@nox.session
def tests_packaging(session: nox.Session) -> None:
    """
    Run the packaging tests.
    """

    session.install("-r", "tests/requirements.txt", "pip")
    session.run("pytest", "tests/extra_python_package", *session.posargs)


@nox.session(reuse_venv=True)
def docs(session: nox.Session) -> None:
    """
    Build the docs. Pass --non-interactive to avoid serving.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", dest="builder", default="html", help="Build target (default: html)"
    )
    args, posargs = parser.parse_known_args(session.posargs)
    serve = args.builder == "html" and session.interactive

    extra_installs = ["sphinx-autobuild"] if serve else []
    session.install("-r", "docs/requirements.txt", *extra_installs)
    session.chdir("docs")

    shared_args = (
        "-n",  # nitpicky mode
        "-T",  # full tracebacks
        f"-b={args.builder}",
        ".",
        f"_build/{args.builder}",
        *posargs,
    )

    if serve:
        session.run(
            "sphinx-autobuild", "--open-browser", "--ignore=.build", *shared_args
        )
    else:
        session.run("sphinx-build", "--keep-going", *shared_args)


@nox.session(reuse_venv=True)
def make_changelog(session: nox.Session) -> None:
    """
    Inspect the closed issues and make entries for a changelog.
    """
    session.install("ghapi", "rich")
    session.run("python", "tools/make_changelog.py")


@nox.session(reuse_venv=True)
def build(session: nox.Session) -> None:
    """
    Build SDists and wheels.
    """

    session.install("build")
    session.log("Building normal files")
    session.run("python", "-m", "build", *session.posargs)
    session.log("Building pybind11-global files (PYBIND11_GLOBAL_SDIST=1)")
    session.run(
        "python", "-m", "build", *session.posargs, env={"PYBIND11_GLOBAL_SDIST": "1"}
    )
