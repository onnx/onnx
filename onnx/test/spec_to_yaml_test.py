# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import filecmp
import pathlib
import sys
import tempfile
import unittest

from ruamel.yaml import YAML

from onnx import defs

# Import the functions under test from tools/spec_to_yaml.py
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools"))
from spec_to_yaml import dump_value, main


class TestSpecToYaml(unittest.TestCase):
    """End-to-end tests for spec_to_yaml export."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.outdir = pathlib.Path(cls.tmpdir.name)
        saved_argv = sys.argv
        sys.argv = ["spec_to_yaml", "--output", str(cls.outdir)]
        main()
        sys.argv = saved_argv
        cls.yaml = YAML()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmpdir.cleanup()

    def test_directory_structure(self) -> None:
        self.assertTrue((self.outdir / "ai.onnx" / "latest").is_dir())
        self.assertTrue((self.outdir / "ai.onnx" / "old").is_dir())
        self.assertTrue((self.outdir / "ai.onnx.ml" / "latest").is_dir())

    def test_every_schema_has_file(self) -> None:
        schemas = defs.get_all_schemas_with_history()
        for schema in schemas:
            domain = schema.domain or "ai.onnx"
            filename = f"{schema.name}-{schema.since_version}.yaml"
            latest_path = self.outdir / domain / "latest" / filename
            old_path = self.outdir / domain / "old" / filename
            self.assertTrue(
                latest_path.exists() or old_path.exists(),
                f"Missing YAML for {domain}/{filename}",
            )

    def test_yaml_is_loadable_and_has_required_fields(self) -> None:
        for path in self.outdir.rglob("*.yaml"):
            with open(path) as f:
                data = self.yaml.load(f)
            self.assertIsInstance(data, dict, path)
            self.assertIn("name", data, path)
            self.assertIn("since_version", data, path)

    def test_falsy_values_preserved(self) -> None:
        """deprecated=False and min_input=0 must not be silently dropped."""
        add_schema = defs.get_schema("Add")
        add_path = (
            self.outdir / "ai.onnx" / "latest" / f"Add-{add_schema.since_version}.yaml"
        )
        with open(add_path) as f:
            data = self.yaml.load(f)
        self.assertIs(data["deprecated"], False)
        self.assertIn("min_input", data)
        self.assertIn("min_output", data)

    def test_fields_match_schema(self) -> None:
        add_schema = defs.get_schema("Add")
        add_path = (
            self.outdir / "ai.onnx" / "latest" / f"Add-{add_schema.since_version}.yaml"
        )
        with open(add_path) as f:
            data = self.yaml.load(f)
        self.assertEqual(data["name"], "Add")
        self.assertEqual(data["since_version"], add_schema.since_version)
        self.assertEqual(len(data["inputs"]), len(add_schema.inputs))
        self.assertEqual(len(data["outputs"]), len(add_schema.outputs))
        self.assertEqual(
            len(data["type_constraints"]), len(add_schema.type_constraints)
        )

    def test_no_methods_or_internal_attrs_exported(self) -> None:
        add_path = (
            self.outdir
            / "ai.onnx"
            / "latest"
            / f"Add-{defs.get_schema('Add').since_version}.yaml"
        )
        with open(add_path) as f:
            data = self.yaml.load(f)
        for key in (
            "consumed",
            "is_infinite",
            "function_body",
            "non_deterministic",
            "file",
            "line",
        ):
            self.assertNotIn(key, data)

    def test_deterministic_output(self) -> None:
        """Running twice produces identical files."""
        with tempfile.TemporaryDirectory() as tmpdir2:
            saved_argv = sys.argv
            sys.argv = ["spec_to_yaml", "--output", tmpdir2]
            main()
            sys.argv = saved_argv

            dir2 = pathlib.Path(tmpdir2)
            files1 = sorted(
                p.relative_to(self.outdir) for p in self.outdir.rglob("*.yaml")
            )
            files2 = sorted(p.relative_to(dir2) for p in dir2.rglob("*.yaml"))
            self.assertEqual(files1, files2)
            for rel_path in files1:
                self.assertTrue(
                    filecmp.cmp(self.outdir / rel_path, dir2 / rel_path, shallow=False),
                    f"Files differ: {rel_path}",
                )

    def test_bool_not_coerced_to_int(self) -> None:
        """dump_value(False) must return False, not 0."""
        self.assertIs(dump_value(False), False)
        self.assertIs(dump_value(True), True)
        self.assertEqual(dump_value(0), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
