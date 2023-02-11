"""Test with different environment configuration with nox.

Documentation:
    https://nox.thea.codes/
"""
from __future__ import annotations

import os
import pathlib

import nox

ONNX_BUILD_ENV_VARS = {
    "CMAKE_ARGS": os.environ.get(
        "CMAKE_ARGS", "-DONNX_WERROR=ON -DONNX_USE_PROTOBUF_SHARED_LIBS=ON"
    ),
    "ONNX_NAMESPACE": "ONNX_NAMESPACE_FOO_BAR_FOR_CI",
    "DEBUG": os.environ.get("DEBUG", "0"),
    "ONNX_ML": os.environ.get("ONNX_ML", "0"),
}


PROTOBUFF_INSTALL_DIR = os.environ.get("PROTOBUFF_INSTALL_DIR", None)
PROTOBUFF_BIN_PATHS: list[str] | None = (
    [
        str((pathlib.Path(PROTOBUFF_INSTALL_DIR) / "bin").absolute()),
        str((pathlib.Path(PROTOBUFF_INSTALL_DIR) / "lib").absolute()),
        str((pathlib.Path(PROTOBUFF_INSTALL_DIR) / "include").absolute()),
    ]
    if PROTOBUFF_INSTALL_DIR
    else None
)

if PROTOBUFF_BIN_PATHS:
    if "PATH" in os.environ:
        ONNX_BUILD_ENV_VARS["PATH"] += ":" + ":".join(PROTOBUFF_BIN_PATHS)
    else:
        ONNX_BUILD_ENV_VARS["PATH"] = ":".join(PROTOBUFF_BIN_PATHS)


@nox.session()
def test_onnx(session: nox.Session):
    """Build and run ONNX python tests."""
    session.run("pip", "install", "-r", "requirements-release.txt", silent=True)
    session.run("pip", "install", ".", env=ONNX_BUILD_ENV_VARS)
    session.run("pytest", *session.posargs)


@nox.session()
def test_cpp(session: nox.Session):
    """ONNX C++ API tests."""
    session.run("pip", "install", "-r", "requirements-release.txt", silent=True)
    session.run(
        "python",
        "setup.py",
        "install",
        env={**ONNX_BUILD_ENV_VARS, "ONNX_BUILD_TESTS": "1"},
        silent=True,
    )
    session.run(
        "./.setuptools-cmake-build/onnx_gtests",
        env={"LD_LIBRARY_PATH": "./.setuptools-cmake-build/:$LD_LIBRARY_PATH"},
        external=True,
    )


@nox.session()
def test_backend_test_generation(session: nox.Session):
    """Test backend test data."""
    session.run("pip", "install", "-r", "requirements-release.txt", silent=True)
    session.run("pip", "install", ".", env=ONNX_BUILD_ENV_VARS)
    session.run("python", "onnx/backend/test/cmd_tools.py", "generate-data", "--clean")
    session.run("bash", "tools/check_generated_backend_test_data.sh", external=True)
