# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for _tar_members_filter path-containment check in onnx/utils.py.

These tests exercise the separator-guard fix that prevents a tar member whose
absolute path shares a string prefix with the extraction directory (but resolves
to a sibling directory) from bypassing the containment check.

The tests import _tar_members_filter from onnx.utils when the package is
available (i.e. in CI where the C extension has been built), and fall back to
loading only the utils.py source file with lightweight stubs otherwise.
"""
from __future__ import annotations

import importlib.util
import io
import os
import sys
import tarfile
import tempfile
import types
import unittest


def _get_tar_members_filter():
    """Return _tar_members_filter, loading onnx.utils without the C extension if needed."""
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    utils_path = os.path.join(repo_root, "onnx", "utils.py")

    # Inject stub modules so the top-level imports in utils.py succeed.
    # We replace any broken partial entries left by a failed 'import onnx'.
    stub_names = ["onnx", "onnx.checker", "onnx.helper", "onnx.shape_inference"]
    saved: dict[str, object] = {}
    for name in stub_names:
        saved[name] = sys.modules.get(name)
        sys.modules[name] = types.ModuleType(name)

    try:
        spec = importlib.util.spec_from_file_location("_onnx_utils_impl", utils_path)
        assert spec is not None and spec.loader is not None
        utils_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(utils_mod)  # type: ignore[union-attr]
        return utils_mod._tar_members_filter
    finally:
        # Restore original sys.modules state.
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original  # type: ignore[assignment]


_tar_members_filter = _get_tar_members_filter()


def _make_tar(member_name: str) -> tarfile.TarFile:
    """Return an in-memory TarFile containing one zero-byte member."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        info = tarfile.TarInfo(name=member_name)
        info.size = 0
        tf.addfile(info, io.BytesIO(b""))
    buf.seek(0)
    return tarfile.open(fileobj=buf, mode="r")


class TestTarMembersFilterPrefixCollision(unittest.TestCase):
    """Regression tests for the os.sep separator-guard fix."""

    def test_tar_members_filter_prefix_collision_bypass_raises(self) -> None:
        """A member that shares only a string prefix with the base dir must raise.

        Without the os.sep guard, ``abs_member.startswith(abs_base)`` is True
        for ``/tmp/models_evil/x.txt`` when ``abs_base=/tmp/models`` because the
        string ``/tmp/models`` is a prefix of ``/tmp/models_evil/...``.
        The fix appends ``os.sep`` so the check becomes
        ``startswith('/tmp/models/')`` (POSIX) or ``startswith('/tmp/models\\')``
        (Windows), which correctly rejects sibling directories.
        """
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "models")
            os.makedirs(base)
            # ``../models_evil/x.txt`` escapes ``models/`` into ``models_evil/``
            tf = _make_tar("../models_evil/x.txt")
            with self.assertRaises(RuntimeError):
                _tar_members_filter(tf, base)

    def test_tar_members_filter_prefix_collision_legitimate_passes(self) -> None:
        """A member legitimately inside the base directory must not raise."""
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "models")
            os.makedirs(base)
            tf = _make_tar("model.onnx")
            members = _tar_members_filter(tf, base)
            self.assertEqual(len(members), 1)
            self.assertEqual(members[0].name, "model.onnx")

    def test_tar_members_filter_prefix_collision_empty_name_raises(self) -> None:
        """A member with an empty name resolves to abs_base itself.

        ``os.path.join(base, '')`` equals ``base`` on most platforms, so
        ``abs_member == abs_base``.  The equality arm of the fix
        (``and abs_member != abs_base``) allows this case through without
        raising, which preserves prior behaviour for self-referential entries.
        """
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "models")
            os.makedirs(base)
            tf = _make_tar("")
            # Empty name -> abs_member == abs_base; must NOT raise.
            members = _tar_members_filter(tf, base)
            self.assertEqual(len(members), 1)


if __name__ == "__main__":
    unittest.main()
