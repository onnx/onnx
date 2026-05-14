# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests for tools/spec_to_yaml.py.

These tests cover a maintainer-only script; they are not part of the
installed ``onnx`` package. They are picked up via the ``tools`` entry in
``[tool.pytest.ini_options]`` in ``pyproject.toml``.
"""

from __future__ import annotations

import filecmp
import pathlib
import tempfile
import unittest

from ruamel.yaml import YAML
from spec_to_yaml import dump_schemas, dump_value

from onnx import defs


def _latest_yaml(
    outdir: pathlib.Path, op: str, domain: str = "ai.onnx"
) -> pathlib.Path:
    """Return the path to the latest-version YAML for ``op`` in ``domain``."""
    since = defs.get_schema(op).since_version
    return outdir / domain / "latest" / f"{op}-{since}.yaml"


def _load_yaml(yaml: YAML, path: pathlib.Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.load(f)  # type: ignore[no-any-return]


class TestSpecToYaml(unittest.TestCase):
    """End-to-end tests for spec_to_yaml export."""

    tmpdir: tempfile.TemporaryDirectory
    outdir: pathlib.Path
    yaml: YAML

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.outdir = pathlib.Path(cls.tmpdir.name)
        cls.yaml = YAML()
        # Generate once; tests below read from cls.outdir without re-dumping.
        dump_schemas(cls.outdir, verbose=False)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmpdir.cleanup()

    def test_every_schema_has_file(self) -> None:
        for schema in defs.get_all_schemas_with_history():
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
            data = _load_yaml(self.yaml, path)
            self.assertIsInstance(data, dict, path)
            self.assertIn("name", data, path)
            self.assertIn("since_version", data, path)

    def test_defaults_and_derived_fields_omitted(self) -> None:
        # The dump should contain only what the C++ author had to write. Add
        # is the canonical "everything at default" op: not deprecated, COMMON
        # support, deterministic, no attributes, no function body, all params
        # Single/homogeneous. None of those should appear.
        data = _load_yaml(self.yaml, _latest_yaml(self.outdir, "Add"))

        # Defaults — author wrote nothing to get these values.
        for key in ("deprecated", "support_level", "domain", "node_determinism"):
            self.assertNotIn(key, data)
        # Derived from inputs/outputs.
        for key in ("min_input", "max_input", "min_output", "max_output"):
            self.assertNotIn(key, data)
        # Framework state — only a "yes there's a C++ function pointer" flag.
        for key in (
            "has_function",
            "has_context_dependent_function",
            "has_type_and_shape_inference_function",
            "has_data_propagation_function",
        ):
            self.assertNotIn(key, data)
        # Empty containers.
        for key in (
            "attributes",
            "function_opset_versions",
            "context_dependent_function_opset_versions",
        ):
            self.assertNotIn(key, data)
        # Per-parameter defaults.
        for param in data["inputs"] + data["outputs"]:
            for key in ("option", "is_homogeneous", "min_arity"):
                self.assertNotIn(key, param, f"{param.get('name')}.{key}")

    def test_non_default_values_emitted(self) -> None:
        # Non-default values must still appear. Upsample-10 is deprecated.
        upsample = _load_yaml(self.yaml, _latest_yaml(self.outdir, "Upsample"))
        self.assertIs(upsample["deprecated"], True)

    def test_derivable_param_types_omitted(self) -> None:
        # For Add, every input/output references TC T whose allowed_type_strs
        # exactly enumerate the effective types — so the per-param `types`
        # field is derivable and must be omitted.
        data = _load_yaml(self.yaml, _latest_yaml(self.outdir, "Add"))
        for param in data["inputs"] + data["outputs"]:
            self.assertNotIn(
                "types", param, f"types should be derived for {param['name']}"
            )

    def test_variadic_param_keeps_non_default_option(self) -> None:
        # Concat has a Variadic input; the non-default option must be emitted.
        data = _load_yaml(self.yaml, _latest_yaml(self.outdir, "Concat"))
        options = [p.get("option") for p in data["inputs"]]
        self.assertIn("Variadic", options)

    def test_deterministic_output(self) -> None:
        """Running twice produces byte-identical files."""
        with tempfile.TemporaryDirectory() as raw_dir2:
            dir2 = pathlib.Path(raw_dir2)
            dump_schemas(dir2, verbose=False)

            files1 = sorted(
                p.relative_to(self.outdir) for p in self.outdir.rglob("*.yaml")
            )
            files2 = sorted(p.relative_to(dir2) for p in dir2.rglob("*.yaml"))
            self.assertEqual(files1, files2)

            _, mismatches, errors = filecmp.cmpfiles(
                self.outdir, dir2, [str(f) for f in files1], shallow=False
            )
            self.assertEqual(mismatches, [], f"Files differ between runs: {mismatches}")
            self.assertEqual(errors, [], f"Files unreadable: {errors}")

    def test_bool_not_coerced_to_int(self) -> None:
        """dump_value(False) must return False, not 0."""
        self.assertIs(dump_value(False), False)
        self.assertIs(dump_value(True), True)
        self.assertEqual(dump_value(0), 0)

    def test_attribute_name_dropped(self) -> None:
        for path in self.outdir.rglob("*.yaml"):
            data = _load_yaml(self.yaml, path)
            for key, attr in (data.get("attributes") or {}).items():
                self.assertNotIn("name", attr, f"{path.name}:{key}")

    def test_string_default_value_decoded(self) -> None:
        # Bytes defaults must render as strings, not `!!binary ...`.
        data = _load_yaml(self.yaml, _latest_yaml(self.outdir, "Conv"))
        self.assertEqual(data["attributes"]["auto_pad"]["default_value"], "NOTSET")

    def test_multiline_doc_uses_block_scalar(self) -> None:
        raw = _latest_yaml(self.outdir, "Add").read_text(encoding="utf-8")
        doc_line = next(line for line in raw.splitlines() if line.startswith("doc:"))
        indicator = doc_line.split(":", 1)[1].strip().split()[0]
        self.assertTrue(indicator.startswith("|"), doc_line)


if __name__ == "__main__":
    unittest.main(verbosity=2)
