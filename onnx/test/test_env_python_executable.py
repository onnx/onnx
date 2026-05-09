# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import pathlib
import platform
import sys
import sysconfig
import tempfile
import unittest
from unittest.mock import patch


# Extract get_python_execute function from setup.py for testing
def get_python_execute() -> str:
    """Get the Python executable path for CMake configuration.

    Prefer sys.executable as it represents the currently running Python.
    Only fall back to directory traversal if sys.executable is invalid.
    """
    WINDOWS = os.name == "nt"

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


class TestGetPythonExecutable(unittest.TestCase):
    """Test suite for get_python_execute() function from setup.py."""

    def test_windows_returns_sys_executable(self) -> None:
        """On Windows, get_python_execute() should always return sys.executable."""
        with patch("os.name", "nt"):
            result = get_python_execute()
            self.assertEqual(result, sys.executable)

    def test_valid_sys_executable_is_preferred(self) -> None:
        """When sys.executable is valid, it should be returned (non-Windows)."""
        with patch("os.name", "posix"):
            # sys.executable should be valid in most test environments
            result = get_python_execute()
            # Should return sys.executable since it's valid
            self.assertEqual(result, sys.executable)

    @unittest.skipIf(
        sys.platform == "win32" and sys.version_info < (3, 11),
        "On Windows this test requires Python >= 3.11 due to sysconfig/sys.abiflags behavior on older interpreters",
    )
    def test_invalid_sys_executable_falls_back(self) -> None:
        """When sys.executable is invalid, should fall back to directory search."""
        with (
            patch("os.name", "posix"),
            patch("sys.executable", "/nonexistent/python"),
            patch("os.path.isfile") as mock_isfile,
            patch("os.access") as mock_access,
            patch("os.path.isdir") as mock_isdir,
        ):
            # Mock sys.executable as invalid
            mock_isfile.return_value = False
            mock_access.return_value = False
            mock_isdir.return_value = False

            result = get_python_execute()
            # Should fall back to sys.executable as last resort
            self.assertEqual(result, "/nonexistent/python")

    def test_fallback_finds_python3_in_bin(self) -> None:
        """Test fallback finds python3 in bin directory when sys.executable is invalid."""
        # Create a temporary directory structure for testing
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_python_dir = pathlib.Path(tmpdir) / "python_install"
            mock_bin_dir = mock_python_dir / "bin"
            # Include path typically goes: <prefix>/include/pythonX.Y
            # Going up two dirs (.. ..) gets us back to prefix
            mock_include_dir = mock_python_dir / "include" / "python3.12"
            mock_bin_dir.mkdir(parents=True)
            mock_include_dir.mkdir(parents=True)
            mock_python3 = mock_bin_dir / "python3"
            # Create actual executable file
            mock_python3.touch(mode=0o755)

            with (
                patch("os.name", "posix"),
                patch("sys.executable", "/invalid/python"),
                patch("sysconfig.get_path") as mock_get_path,
            ):
                # Setup mocks - return the include/python3.12 path
                mock_get_path.return_value = str(mock_include_dir)

                result = get_python_execute()
                self.assertEqual(result, str(mock_python3))

    def test_fallback_finds_python_in_bin(self) -> None:
        """Test fallback finds python (not python3) when python3 doesn't exist."""
        # Create a temporary directory structure for testing
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_python_dir = pathlib.Path(tmpdir) / "python_install"
            mock_bin_dir = mock_python_dir / "bin"
            # Include path typically goes: <prefix>/include/pythonX.Y
            # Going up two dirs (.. ..) gets us back to prefix
            mock_include_dir = mock_python_dir / "include" / "python3.12"
            mock_bin_dir.mkdir(parents=True)
            mock_include_dir.mkdir(parents=True)
            mock_python = mock_bin_dir / "python"
            # Create actual executable file (python3 doesn't exist)
            mock_python.touch(mode=0o755)

            with (
                patch("os.name", "posix"),
                patch("sys.executable", "/invalid/python"),
                patch("sysconfig.get_path") as mock_get_path,
            ):
                # Setup mocks - return the include/python3.12 path
                mock_get_path.return_value = str(mock_include_dir)

                result = get_python_execute()
                self.assertEqual(result, str(mock_python))

    @unittest.skipIf(
        sys.platform == "win32" and sys.version_info < (3, 11),
        "On Windows this test requires Python >= 3.11 due to sysconfig/sys.abiflags behavior on older interpreters",
    )
    def test_executable_permission_check(self) -> None:
        """Test that executable permission is verified for sys.executable."""
        with (
            patch("os.name", "posix"),
            patch("sys.executable", "/path/to/python"),
            patch("os.path.isfile") as mock_isfile,
            patch("os.access") as mock_access,
            patch("os.path.isdir") as mock_isdir,
        ):
            # File exists but is not executable
            mock_isfile.return_value = True
            mock_access.return_value = False
            mock_isdir.return_value = False

            result = get_python_execute()
            # Should fall back to sys.executable even though it's not executable
            self.assertEqual(result, sys.executable)

    def test_real_environment(self) -> None:
        """Test with real environment to ensure it works in practice."""
        result = get_python_execute()

        # Result should be a non-empty string
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

        # On Windows, should be exactly sys.executable
        if platform.system() == "Windows":
            self.assertEqual(result, sys.executable)
        # On non-Windows, in a normal environment, should return sys.executable
        # since it should be valid
        elif os.path.isfile(sys.executable) and os.access(sys.executable, os.X_OK):
            self.assertEqual(result, sys.executable)

    def test_virtual_environment_detection(self) -> None:
        """Test that virtual environment Python is correctly detected."""
        with patch("os.name", "posix"):
            result = get_python_execute()

            # In a venv or virtualenv, sys.executable should point to the venv Python
            # and the function should return it
            if hasattr(sys, "prefix") and hasattr(sys, "base_prefix"):
                in_venv = sys.prefix != sys.base_prefix
                if in_venv and os.path.isfile(sys.executable):
                    self.assertEqual(
                        result,
                        sys.executable,
                        "Virtual environment Python should be detected",
                    )

    @unittest.skipIf(
        platform.system() == "Windows",
        "Fallback mechanism only applies to POSIX systems",
    )
    def test_cpython_issue_84399_fallback(self) -> None:
        """Test that the fallback handles cpython issue #84399 edge case.

        This test is skipped on Windows because the Windows implementation
        always returns sys.executable without any fallback logic.
        The fallback mechanism is only relevant for POSIX systems.
        """
        # This test verifies that the fallback mechanism is still present
        # for the edge case mentioned in https://github.com/python/cpython/issues/84399
        with (
            patch("sys.executable", "/usr/bin/python-invalid"),
            patch("os.path.isfile") as mock_isfile,
            patch("os.access") as mock_access,
            patch("sysconfig.get_path") as mock_get_path,
            patch("os.path.isdir") as mock_isdir,
            patch("os.path.abspath") as mock_abspath,
            patch("os.path.join", side_effect=lambda *args: "/".join(args)),
        ):
            # sys.executable points to invalid path
            mock_isfile.return_value = False
            mock_access.return_value = False
            mock_get_path.return_value = "/usr/include/python3.12"
            mock_isdir.return_value = True
            mock_abspath.return_value = "/usr"

            # Mock finding python3 in /usr/bin
            def isfile_check(path: str) -> bool:
                return path == "/usr/bin/python3"

            mock_isfile.side_effect = isfile_check

            result = get_python_execute()
            self.assertEqual(result, "/usr/bin/python3")


if __name__ == "__main__":
    unittest.main()
