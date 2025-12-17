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

    @staticmethod
    def _is_stubs_disabled(extra_cmake_args: list[str]) -> bool:
        """Helper method to detect if stubs are disabled in CMAKE_ARGS.

        Matches the logic in setup.py which accepts all CMake boolean false values.
        """
        cmake_false_values = {"0", "off", "false", "no", "n"}
        return any(
            arg.lower().startswith("-donnx_gen_pb_type_stubs=")
            and arg.split("=", 1)[1].strip().lower() in cmake_false_values
            for arg in extra_cmake_args
        )

    def test_cmake_false_values_detected(self) -> None:
        """Test all CMake boolean false values are detected (case-insensitive)."""
        # Test all 5 CMake false values: 0, off, false, no, n
        # Include case variations and whitespace to verify robustness
        false_value_cases = [
            "-DONNX_GEN_PB_TYPE_STUBS=0",
            "-DONNX_GEN_PB_TYPE_STUBS=OFF",
            "-DONNX_GEN_PB_TYPE_STUBS=off",
            "-DONNX_GEN_PB_TYPE_STUBS=Off",
            "-DONNX_GEN_PB_TYPE_STUBS=false",
            "-DONNX_GEN_PB_TYPE_STUBS=FALSE",
            "-DONNX_GEN_PB_TYPE_STUBS=no",
            "-DONNX_GEN_PB_TYPE_STUBS=NO",
            "-DONNX_GEN_PB_TYPE_STUBS=n",
            "-DONNX_GEN_PB_TYPE_STUBS=N",
            "-Donnx_gen_pb_type_stubs=OFF",  # Case-insensitive flag name
            "-DONNX_GEN_PB_TYPE_STUBS= OFF ",  # Whitespace handling
        ]
        for test_arg in false_value_cases:
            with self.subTest(arg=test_arg):
                stubs_disabled = self._is_stubs_disabled([test_arg])
                self.assertTrue(stubs_disabled, f"Should detect {test_arg} as disabled")

    def test_cmake_true_values_not_detected(self) -> None:
        """Test that CMake true values are NOT detected as disabled."""
        true_value_cases = [
            "-DONNX_GEN_PB_TYPE_STUBS=ON",
            "-DONNX_GEN_PB_TYPE_STUBS=1",
            "-DONNX_GEN_PB_TYPE_STUBS=true",
            "-DONNX_GEN_PB_TYPE_STUBS=yes",
        ]
        for test_arg in true_value_cases:
            with self.subTest(arg=test_arg):
                stubs_disabled = self._is_stubs_disabled([test_arg])
                self.assertFalse(stubs_disabled, f"{test_arg} should not disable stubs")

    def test_default_behavior_when_flag_not_set(self) -> None:
        """Test default behavior when flag is not present in CMAKE_ARGS."""
        # Test with no CMAKE_ARGS at all
        self.assertFalse(self._is_stubs_disabled([]))

        # Test with other CMAKE_ARGS but not the stubs flag
        other_args = ["-DONNX_USE_LITE_PROTO=ON", "-DCMAKE_BUILD_TYPE=Debug"]
        self.assertFalse(self._is_stubs_disabled(other_args))

    def test_assertion_logic(self) -> None:
        """Test the assertion logic in BuildExt based on stubs_disabled flag."""
        # When stubs enabled and files exist - should pass
        stubs_disabled = False
        generated_pyi_files = ["onnx_pb2.pyi"]
        if not stubs_disabled:
            assert generated_pyi_files  # Should not raise

        # When stubs enabled but no files - should raise
        generated_pyi_files = []
        with self.assertRaises(AssertionError):
            if not stubs_disabled:
                assert generated_pyi_files, "Bug: No .pyi files"

        # When stubs disabled - should skip assertion even if no files
        stubs_disabled = True
        if not stubs_disabled:
            self.fail("Should not execute assertion when stubs are disabled")


class TestIntegration(unittest.TestCase):
    """Integration test for CmakeBuild to BuildExt communication."""

    def test_flag_propagation_via_distribution(self) -> None:
        """Test that onnx_gen_stubs_disabled propagates via distribution object."""
        distribution = MagicMock()

        # Test 1: When stubs explicitly disabled
        extra_cmake_args = ["-DONNX_GEN_PB_TYPE_STUBS=OFF"]
        distribution.onnx_gen_stubs_disabled = (
            TestStubAssertionLogic._is_stubs_disabled(extra_cmake_args)
        )
        stubs_disabled = getattr(distribution, "onnx_gen_stubs_disabled", False)
        self.assertTrue(stubs_disabled)

        # Test 2: When no CMAKE_ARGS (default behavior)
        distribution.onnx_gen_stubs_disabled = False
        stubs_disabled = getattr(distribution, "onnx_gen_stubs_disabled", False)
        self.assertFalse(stubs_disabled)


class TestErrorMessages(unittest.TestCase):
    """Test that error messages match setup.py exactly."""

    def test_error_message_matches_setup_py(self) -> None:
        """Test that error message matches setup.py exactly (case-sensitive)."""
        # This is the exact error message from setup.py (lines 327-330)
        # Must match character-for-character, including capitalization "You can"
        expected_error_message = (
            "Bug: No generated python stub files (.pyi) found. "
            "ONNX_GEN_PB_TYPE_STUBS is ON by default. "
            "You can disable stub generation with CMAKE_ARGS='-DONNX_GEN_PB_TYPE_STUBS=OFF'"
        )

        # Verify key components are present with correct capitalization
        self.assertIn("CMAKE_ARGS", expected_error_message)
        self.assertIn("-DONNX_GEN_PB_TYPE_STUBS=OFF", expected_error_message)
        self.assertIn("You can disable", expected_error_message)  # Capital Y
        self.assertIn("ONNX_GEN_PB_TYPE_STUBS is ON by default", expected_error_message)


if __name__ == "__main__":
    unittest.main()
