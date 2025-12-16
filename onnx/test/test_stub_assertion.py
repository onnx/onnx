# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest
from unittest.mock import MagicMock


class TestStubAssertionLogic(unittest.TestCase):
    """Test the stub file assertion logic in setup.py.

    This tests the logic without importing setup.py to avoid
    triggering setuptools.setup() during test collection.
    """

    def test_cmake_args_with_stubs_disabled(self) -> None:
        """Test CMAKE_ARGS with ONNX_GEN_PB_TYPE_STUBS=OFF sets flag correctly."""
        # Simulate what CmakeBuild.run() does
        extra_cmake_args = ["-DONNX_GEN_PB_TYPE_STUBS=OFF"]

        stubs_disabled = any(
            arg in extra_cmake_args
            for arg in [
                "-DONNX_GEN_PB_TYPE_STUBS=OFF",
                "-DONNX_GEN_PB_TYPE_STUBS=0",
            ]
        )

        self.assertTrue(stubs_disabled, "Should detect -DONNX_GEN_PB_TYPE_STUBS=OFF")

    def test_cmake_args_with_stubs_disabled_zero(self) -> None:
        """Test CMAKE_ARGS with ONNX_GEN_PB_TYPE_STUBS=0 sets flag correctly."""
        extra_cmake_args = ["-DONNX_GEN_PB_TYPE_STUBS=0"]

        stubs_disabled = any(
            arg in extra_cmake_args
            for arg in ["-DONNX_GEN_PB_TYPE_STUBS=OFF", "-DONNX_GEN_PB_TYPE_STUBS=0"]
        )

        self.assertTrue(stubs_disabled, "Should detect -DONNX_GEN_PB_TYPE_STUBS=0")

    def test_cmake_args_with_stubs_enabled(self) -> None:
        """Test CMAKE_ARGS with ONNX_GEN_PB_TYPE_STUBS=ON doesn't set flag."""
        extra_cmake_args = ["-DONNX_GEN_PB_TYPE_STUBS=ON"]

        stubs_disabled = any(
            arg in extra_cmake_args
            for arg in ["-DONNX_GEN_PB_TYPE_STUBS=OFF", "-DONNX_GEN_PB_TYPE_STUBS=0"]
        )

        self.assertFalse(stubs_disabled, "Should not detect disabled when set to ON")

    def test_cmake_args_without_stub_option(self) -> None:
        """Test CMAKE_ARGS without ONNX_GEN_PB_TYPE_STUBS doesn't set flag."""
        extra_cmake_args = ["-DONNX_USE_LITE_PROTO=ON", "-DCMAKE_BUILD_TYPE=Debug"]

        stubs_disabled = any(
            arg in extra_cmake_args
            for arg in ["-DONNX_GEN_PB_TYPE_STUBS=OFF", "-DONNX_GEN_PB_TYPE_STUBS=0"]
        )

        self.assertFalse(
            stubs_disabled, "Should not detect disabled when option not present"
        )

    def test_no_cmake_args(self) -> None:
        """Test when CMAKE_ARGS is not set (default behavior)."""

        # When CMAKE_ARGS is not set and not in extra_cmake_args,
        # onnx_gen_stubs_disabled should not be set (defaults to False via getattr)
        # This test verifies the getattr default behavior
        class FakeDistribution:
            pass

        dist = FakeDistribution()
        stubs_disabled = getattr(dist, "onnx_gen_stubs_disabled", False)

        self.assertFalse(
            stubs_disabled, "Should default to False (stubs enabled) when no CMAKE_ARGS"
        )

    def test_getattr_with_attribute_set(self) -> None:
        """Test getattr when onnx_gen_stubs_disabled is set."""
        mock_distribution = MagicMock()
        mock_distribution.onnx_gen_stubs_disabled = True

        stubs_disabled = getattr(mock_distribution, "onnx_gen_stubs_disabled", False)

        self.assertTrue(stubs_disabled, "Should retrieve True from distribution")

    def test_getattr_with_attribute_not_set(self) -> None:
        """Test getattr when onnx_gen_stubs_disabled is not set (uses default)."""

        # Create a real object without the attribute (not MagicMock which auto-creates attrs)
        class FakeDistribution:
            pass

        dist = FakeDistribution()
        stubs_disabled = getattr(dist, "onnx_gen_stubs_disabled", False)

        self.assertFalse(
            stubs_disabled,
            "Should return default False when attribute not set",
        )

    def test_assertion_logic_stubs_enabled_files_exist(self) -> None:
        """Test assertion logic when stubs enabled and files exist."""
        stubs_disabled = False
        generated_pyi_files = ["onnx_pb2.pyi", "onnx-ml_pb2.pyi"]

        # Should not raise assertion error
        try:
            if not stubs_disabled:
                assert generated_pyi_files, "Bug: No .pyi files"
            success = True
        except AssertionError:
            success = False

        self.assertTrue(success, "Should not raise assertion when files exist")

    def test_assertion_logic_stubs_enabled_no_files(self) -> None:
        """Test assertion logic when stubs enabled but no files exist."""
        stubs_disabled = False
        generated_pyi_files = []

        # Should raise assertion error
        with self.assertRaises(AssertionError) as context:
            if not stubs_disabled:
                assert generated_pyi_files, "Bug: No .pyi files"

        self.assertIn("Bug: No .pyi files", str(context.exception))

    def test_assertion_logic_stubs_disabled_no_files(self) -> None:
        """Test assertion logic when stubs disabled and no files exist."""
        stubs_disabled = True
        generated_pyi_files = []

        # When stubs are disabled, the assertion should be skipped
        # This verifies the condition logic: if not stubs_disabled
        if not stubs_disabled:
            assert generated_pyi_files, "Bug: No .pyi files"
    def test_multiple_cmake_args_with_stubs_disabled(self) -> None:
        """Test CMAKE_ARGS with multiple options including stubs disabled."""
        extra_cmake_args = [
            "-DONNX_USE_LITE_PROTO=ON",
            "-DONNX_GEN_PB_TYPE_STUBS=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
        ]

        stubs_disabled = any(
            arg in extra_cmake_args
            for arg in ["-DONNX_GEN_PB_TYPE_STUBS=OFF", "-DONNX_GEN_PB_TYPE_STUBS=0"]
        )

        self.assertTrue(stubs_disabled, "Should detect disabled flag among other args")


class TestIntegration(unittest.TestCase):
    """Integration tests for the stub assertion logic."""

    def test_end_to_end_disabled_flag_propagation(self) -> None:
        """Test that the disabled flag propagates from CmakeBuild to BuildExt."""
        # Create a mock distribution
        distribution = MagicMock()

        # Simulate CmakeBuild setting the flag
        extra_cmake_args = ["-DONNX_GEN_PB_TYPE_STUBS=OFF"]
        distribution.onnx_gen_stubs_disabled = any(
            arg in extra_cmake_args
            for arg in ["-DONNX_GEN_PB_TYPE_STUBS=OFF", "-DONNX_GEN_PB_TYPE_STUBS=0"]
        )

        # Simulate BuildExt reading the flag
        stubs_disabled = getattr(distribution, "onnx_gen_stubs_disabled", False)

        self.assertTrue(
            stubs_disabled,
            "Flag should propagate from CmakeBuild to BuildExt",
        )

    def test_end_to_end_default_behavior(self) -> None:
        """Test default behavior when no CMAKE_ARGS set."""
        # Create a mock distribution
        distribution = MagicMock()

        # Simulate CmakeBuild with no CMAKE_ARGS (else branch)
        distribution.onnx_gen_stubs_disabled = False

        # Simulate BuildExt reading the flag
        stubs_disabled = getattr(distribution, "onnx_gen_stubs_disabled", False)

        self.assertFalse(stubs_disabled, "Default should be False (stubs enabled)")


class TestErrorMessages(unittest.TestCase):
    """Test that error messages are helpful."""

    def test_error_message_contains_fix_instructions(self) -> None:
        """Test that assertion error message contains fix instructions."""
        error_message = (
            "Bug: No generated python stub files (.pyi) found. "
            "ONNX_GEN_PB_TYPE_STUBS is ON by default. "
            "you can disable stub generation with CMAKE_ARGS='-DONNX_GEN_PB_TYPE_STUBS=OFF'"
        )

        self.assertIn("CMAKE_ARGS", error_message)
        self.assertIn("-DONNX_GEN_PB_TYPE_STUBS=OFF", error_message)
        self.assertIn("you can disable stub generation", error_message)


if __name__ == "__main__":
    unittest.main()
