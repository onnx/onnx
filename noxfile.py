"""Test with different environment configuration with nox.

Documentation:
    https://nox.thea.codes/
"""

import nox


@nox.session()
def test_onnx(session: nox.Session):
    """Build and test ONNX."""
    session.install("-r", "requirements-release.txt")
    session.install(".")
    session.run("pytest", *session.posargs)


@nox.session()
def test_cpp(session: nox.Session):
    """ONNX C++ API tests."""
    session.install("-r", "requirements-release.txt")
    session.run("python", "setup.py", "install")
    session.run(
        "bash",
        "./.setuptools-cmake-build/onnx_gtests",
        env={"LD_LIBRARY_PATH": "./.setuptools-cmake-build/:$LD_LIBRARY_PATH"},
        external=True,
    )


@nox.session()
def test_backend_test_generation(session: nox.Session):
    """Test backend test data."""
    session.install("-r", "requirements-release.txt")
    session.install(".")
    session.run("python", "onnx/backend/test/cmd_tools.py", "generate-data", "--clean")
    session.run("bash", "tools/check_generated_backend_test_data.sh", external=True)


@nox.session()
def check_generated_files(session: nox.Session):
    """Check auto-gen files up-to-date."""
    session.install("-r", "requirements-release.txt")
    session.install(".")
    session.run("python", "onnx/defs/gen_doc.py")
    session.run("python", "onnx/gen_proto.py", "-l")
    session.run("python", "onnx/gen_proto.py", "-l", "--ml")
    session.run("python", "onnx/backend/test/stat_coverage.py")
    session.run("bash", "tools/check_generated_diff.sh", external=True)
