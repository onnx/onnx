"""Test with different environment configuration with nox.

Documentation:
    https://nox.thea.codes/
"""

import os

import nox

DEFAULT_ENV_VARS = {
    "ONNX_BUILD_TESTS": "1",
    "CMAKE_ARGS": os.environ.get(
        "CMAKE_ARGS", "-DONNX_WERROR=ON -DONNX_USE_PROTOBUF_SHARED_LIBS=ON"
    ),
    "ONNX_NAMESPACE": "ONNX_NAMESPACE_FOO_BAR_FOR_CI",
    "DEBUG": os.environ.get("DEBUG", "0"),
    "ONNX_ML": os.environ.get("ONNX_ML", "0"),
}


@nox.session()
def test_onnx(session: nox.Session):
    """Build and run ONNX python tests."""
    session.run("pip", "install", "-r", "requirements-release.txt")
    session.run("pip", "install", ".", env=DEFAULT_ENV_VARS)
    session.run("pytest", *session.posargs)


@nox.session()
def test_cpp(session: nox.Session):
    """ONNX C++ API tests."""
    session.run("pip", "install", "-r", "requirements-release.txt")
    session.run("python", "setup.py", "install", env=DEFAULT_ENV_VARS, silent=True)
    session.run(
        "./.setuptools-cmake-build/onnx_gtests",
        env={"LD_LIBRARY_PATH": "./.setuptools-cmake-build/:$LD_LIBRARY_PATH"},
        external=True,
    )


@nox.session()
def test_backend_test_generation(session: nox.Session):
    """Test backend test data."""
    session.run("pip", "install", "-r", "requirements-release.txt")
    session.run("pip", "install", ".", env=DEFAULT_ENV_VARS)
    session.run("python", "onnx/backend/test/cmd_tools.py", "generate-data", "--clean")
    session.run("bash", "tools/check_generated_backend_test_data.sh", external=True)
