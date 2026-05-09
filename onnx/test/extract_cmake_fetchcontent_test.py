# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

# ---------------------------------------------------------------------------
# Import tools/extract_cmake_fetchcontent.py directly (it is not a package).
# ---------------------------------------------------------------------------
_TOOLS_DIR = Path(__file__).parents[2] / "tools"
_SCRIPT = _TOOLS_DIR / "extract_cmake_fetchcontent.py"

spec = importlib.util.spec_from_file_location("extract_cmake_fetchcontent", _SCRIPT)
assert spec is not None and spec.loader is not None
_mod = importlib.util.module_from_spec(spec)
sys.modules["extract_cmake_fetchcontent"] = _mod
spec.loader.exec_module(_mod)  # type: ignore[union-attr]

_parse_cmake_variables = _mod._parse_cmake_variables
_parse_fetchcontent_declares = _mod._parse_fetchcontent_declares
_build_component = _mod._build_component
_make_bom = _mod._make_bom
_merge_into = _mod._merge_into

# ---------------------------------------------------------------------------
# Minimal CMake snippets that mirror the real CMakeLists.txt patterns.
# Version strings used here are synthetic fixture values — not tied to any
# real release. Update these only if a test needs a different format.
# ---------------------------------------------------------------------------

_FIXTURE_ABSL_VERSION = "20240722.0"
_FIXTURE_ABSL_SHA256 = (
    "f50e5ac311a81382da7fa75b97310e4b9006474f9560ac46f54a9967f07d4ae3"
)
_FIXTURE_NANOBIND_TAG = "v2.10.2"
_FIXTURE_NANOBIND_VERSION = "2.10.2"  # tag without leading "v"

# URL format mirrors the real abseil entry: .../releases/download/<ver>/name-<ver>.tar.gz
# so the version is extracted from the /download/<ver>/ path segment, not the filename.
_URL_CMAKE = f"""\
set(AbseilURL https://github.com/abseil/abseil-cpp/releases/download/{_FIXTURE_ABSL_VERSION}/abseil-cpp-{_FIXTURE_ABSL_VERSION}.tar.gz)
set(AbseilSHA256 {_FIXTURE_ABSL_SHA256})
FetchContent_Declare(
  absl
  URL ${{AbseilURL}}
  URL_HASH SHA256=${{AbseilSHA256}}
)
"""

_GIT_CMAKE = f"""\
FetchContent_Declare(
  nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind.git
  GIT_TAG {_FIXTURE_NANOBIND_TAG}
)
"""

_MULTI_CMAKE = _URL_CMAKE + "\n" + _GIT_CMAKE


class TestParseFetchContentDeclares(unittest.TestCase):
    def _parse(self, text: str) -> list[dict[str, str]]:
        variables = _parse_cmake_variables(text)
        return _parse_fetchcontent_declares(text, variables)

    def test_url_based_name(self) -> None:
        entries = self._parse(_URL_CMAKE)
        assert len(entries) == 1
        assert entries[0]["name"] == "absl"

    def test_url_resolved(self) -> None:
        entries = self._parse(_URL_CMAKE)
        assert "abseil-cpp" in entries[0]["url"]

    def test_url_hash_parsed(self) -> None:
        entries = self._parse(_URL_CMAKE)
        assert entries[0]["hash_alg"] == "SHA256"
        assert len(entries[0]["hash_val"]) == 64

    def test_git_based(self) -> None:
        entries = self._parse(_GIT_CMAKE)
        assert len(entries) == 1
        entry = entries[0]
        assert entry["name"] == "nanobind"
        assert "nanobind.git" in entry["git_url"]
        assert entry["git_tag"] == "v2.10.2"

    def test_multiple_entries(self) -> None:
        entries = self._parse(_MULTI_CMAKE)
        names = [e["name"] for e in entries]
        assert "absl" in names
        assert "nanobind" in names

    def test_no_fetchcontent(self) -> None:
        assert self._parse("set(FOO bar)") == []


class TestBuildComponent(unittest.TestCase):
    def _component_from(self, cmake: str, name: str) -> dict:
        variables = _parse_cmake_variables(cmake)
        entries = _parse_fetchcontent_declares(cmake, variables)
        entry = next(e for e in entries if e["name"].lower() == name)
        return _build_component(entry, cmake)

    def test_url_component_name_canonical(self) -> None:
        comp = self._component_from(_URL_CMAKE, "absl")
        assert comp["name"] == "abseil-cpp"

    def test_url_component_type(self) -> None:
        comp = self._component_from(_URL_CMAKE, "absl")
        assert comp["type"] == "library"

    def test_url_component_version_from_url(self) -> None:
        comp = self._component_from(_URL_CMAKE, "absl")
        assert "version" in comp
        assert _FIXTURE_ABSL_VERSION in comp["version"]

    def test_url_component_no_trailing_dot_in_version(self) -> None:
        # Regression: greedy [\d.]* in the URL regex once pulled in the '.' from
        # '.tar.gz', producing "20240722.0." instead of "20240722.0".
        comp = self._component_from(_URL_CMAKE, "absl")
        assert not comp["version"].endswith(".")

    def test_url_component_purl_github(self) -> None:
        comp = self._component_from(_URL_CMAKE, "absl")
        assert comp["purl"].startswith("pkg:github/abseil/")

    def test_url_component_purl_tag_consistent_with_version(self) -> None:
        # When an explicit version variable exists, the purl tag must match it
        # so that comp["version"] and the purl tag are consistent even when the
        # URL-embedded tag differs (e.g. URL has "v33.6" but VERSION says "6.33.6").
        ver = "1.2.3"
        url_tag = "v0.99"  # deliberately different from ver
        cmake = f"""\
set(MyDep_VERSION {ver})
set(MyURL https://github.com/example/mydep/releases/download/{url_tag}/mydep-{url_tag}.tar.gz)
FetchContent_Declare(
  MyDep
  URL ${{MyURL}}
  URL_HASH SHA256=abc123
)
"""
        variables = _parse_cmake_variables(cmake)
        entries = _parse_fetchcontent_declares(cmake, variables)
        comp = _build_component(entries[0], cmake)
        assert comp["version"] == ver
        assert comp["purl"].endswith(f"@v{ver}"), (
            f"purl tag must match version, got: {comp['purl']}"
        )

    def test_url_component_hash(self) -> None:
        comp = self._component_from(_URL_CMAKE, "absl")
        assert comp["hashes"][0]["alg"] == "SHA-256"

    def test_url_component_license_absl(self) -> None:
        comp = self._component_from(_URL_CMAKE, "absl")
        assert comp["licenses"][0]["license"]["id"] == "Apache-2.0"

    def test_git_component_version_strips_v(self) -> None:
        comp = self._component_from(_GIT_CMAKE, "nanobind")
        assert comp["version"] == _FIXTURE_NANOBIND_VERSION

    def test_git_component_purl_uses_tag(self) -> None:
        comp = self._component_from(_GIT_CMAKE, "nanobind")
        assert _FIXTURE_NANOBIND_TAG in comp["purl"]

    def test_git_component_external_ref_vcs(self) -> None:
        comp = self._component_from(_GIT_CMAKE, "nanobind")
        vcs_refs = [r for r in comp["externalReferences"] if r["type"] == "vcs"]
        assert len(vcs_refs) == 1
        assert "nanobind" in vcs_refs[0]["url"]

    def test_git_component_external_ref_distribution(self) -> None:
        comp = self._component_from(_GIT_CMAKE, "nanobind")
        dist_refs = [
            r for r in comp["externalReferences"] if r["type"] == "distribution"
        ]
        assert len(dist_refs) == 1
        assert _FIXTURE_NANOBIND_TAG in dist_refs[0]["url"]
        assert dist_refs[0]["url"].endswith(".tar.gz")

    def test_git_component_license_nanobind(self) -> None:
        comp = self._component_from(_GIT_CMAKE, "nanobind")
        assert comp["licenses"][0]["license"]["id"] == "BSD-3-Clause"

    def test_bom_ref_uses_canonical_name(self) -> None:
        comp = self._component_from(_URL_CMAKE, "absl")
        assert comp["bom-ref"] == f"abseil-cpp@{_FIXTURE_ABSL_VERSION}"

    def test_bom_ref_with_version(self) -> None:
        comp = self._component_from(_GIT_CMAKE, "nanobind")
        assert comp["bom-ref"] == f"nanobind@{_FIXTURE_NANOBIND_VERSION}"

    def test_unknown_dep_no_license(self) -> None:
        cmake = """\
FetchContent_Declare(
  unknown_dep
  GIT_REPOSITORY https://github.com/example/unknown.git
  GIT_TAG v1.0.0
)
"""
        comp = self._component_from(cmake, "unknown_dep")
        assert "licenses" not in comp

    def test_sha256_alg_normalized(self) -> None:
        cmake = """\
set(MyURL https://github.com/example/dep/archive/v1.0.tar.gz)
set(MySHA abc123)
FetchContent_Declare(
  mydep
  URL ${MyURL}
  URL_HASH SHA256=${MySHA}
)
"""
        comp = self._component_from(cmake, "mydep")
        assert comp["hashes"][0]["alg"] == "SHA-256"

    def test_sha1_alg_normalized(self) -> None:
        cmake = """\
set(MyURL https://github.com/example/dep/archive/v1.0.tar.gz)
set(MySHA abc123)
FetchContent_Declare(
  mydep
  URL ${MyURL}
  URL_HASH SHA1=${MySHA}
)
"""
        comp = self._component_from(cmake, "mydep")
        assert comp["hashes"][0]["alg"] == "SHA-1"


class TestMakeBom(unittest.TestCase):
    def setUp(self) -> None:
        variables = _parse_cmake_variables(_GIT_CMAKE)
        entries = _parse_fetchcontent_declares(_GIT_CMAKE, variables)
        self.components = [_build_component(e, _GIT_CMAKE) for e in entries]
        self.bom = _make_bom(self.components, "build")

    def test_format(self) -> None:
        assert self.bom["bomFormat"] == "CycloneDX"

    def test_spec_version(self) -> None:
        assert self.bom["specVersion"] == "1.7"

    def test_serial_number_is_urn_uuid(self) -> None:
        assert self.bom["serialNumber"].startswith("urn:uuid:")

    def test_lifecycle(self) -> None:
        assert self.bom["metadata"]["lifecycles"] == [{"phase": "build"}]

    def test_components_present(self) -> None:
        assert len(self.bom["components"]) == 1
        assert self.bom["components"][0]["name"] == "nanobind"

    def test_manufacturer(self) -> None:
        assert (
            self.bom["metadata"]["manufacturer"]["name"] == "ONNX Project Contributors"
        )

    def test_schema_field_present(self) -> None:
        assert self.bom["$schema"] == "https://cyclonedx.org/schema/bom-1.7.schema.json"


class TestMergeInto(unittest.TestCase):
    def _make_base_bom(self) -> dict:
        return {
            "bomFormat": "CycloneDX",
            "specVersion": "1.7",
            "serialNumber": "urn:uuid:00000000-0000-0000-0000-000000000001",
            "version": 1,
            "metadata": {},
            "components": [
                {"type": "library", "name": "numpy", "bom-ref": "numpy@2.0.0"}
            ],
            "dependencies": [{"ref": "numpy@2.0.0"}],
        }

    def _new_comps(self) -> list:
        variables = _parse_cmake_variables(_GIT_CMAKE)
        entries = _parse_fetchcontent_declares(_GIT_CMAKE, variables)
        return [_build_component(e, _GIT_CMAKE) for e in entries]

    def _merge(self, base: dict, new_comps: list) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp) / "base.json"
            tmp_path.write_text(json.dumps(base), encoding="utf-8")
            return _merge_into(tmp_path, new_comps, "build")

    def test_components_appended(self) -> None:
        result = self._merge(self._make_base_bom(), self._new_comps())
        names = [c["name"] for c in result["components"]]
        assert "numpy" in names
        assert "nanobind" in names

    def test_lifecycle_overwritten(self) -> None:
        result = self._merge(self._make_base_bom(), self._new_comps())
        assert result["metadata"]["lifecycles"] == [{"phase": "build"}]

    def test_dependencies_extended(self) -> None:
        result = self._merge(self._make_base_bom(), self._new_comps())
        refs = {d["ref"] for d in result["dependencies"]}
        nanobind_ref = f"nanobind@{_FIXTURE_NANOBIND_VERSION}"
        assert nanobind_ref in refs

    def test_no_duplicate_dependencies(self) -> None:
        nanobind_ref = f"nanobind@{_FIXTURE_NANOBIND_VERSION}"
        base = self._make_base_bom()
        base["dependencies"].append({"ref": nanobind_ref})
        result = self._merge(base, self._new_comps())
        refs = [d["ref"] for d in result["dependencies"]]
        assert refs.count(nanobind_ref) == 1

    def test_schema_field_added_when_missing(self) -> None:
        base = self._make_base_bom()
        result = self._merge(base, self._new_comps())
        assert result["$schema"] == "https://cyclonedx.org/schema/bom-1.7.schema.json"

    def test_schema_field_preserved_when_present(self) -> None:
        base = self._make_base_bom()
        base["$schema"] = "https://cyclonedx.org/schema/bom-1.7.schema.json"
        result = self._merge(base, self._new_comps())
        assert result["$schema"] == "https://cyclonedx.org/schema/bom-1.7.schema.json"


class TestAgainstRealCMakeLists(unittest.TestCase):
    """Smoke test: parse the actual CMakeLists.txt and verify expected deps."""

    _CMAKE_PATH = Path(__file__).parents[2] / "CMakeLists.txt"

    @classmethod
    def setUpClass(cls) -> None:
        if not cls._CMAKE_PATH.exists():
            raise unittest.SkipTest("CMakeLists.txt not found")
        text = cls._CMAKE_PATH.read_text(encoding="utf-8")
        variables = _parse_cmake_variables(text)
        entries = _parse_fetchcontent_declares(text, variables)
        cls.components = [_build_component(e, text) for e in entries]
        cls.by_name = {c["name"]: c for c in cls.components}

    def test_abseil_cpp_present(self) -> None:
        assert "abseil-cpp" in self.by_name

    def test_protobuf_present(self) -> None:
        assert "protobuf" in self.by_name

    def test_nanobind_present(self) -> None:
        assert "nanobind" in self.by_name

    def test_abseil_cpp_license(self) -> None:
        license_id = self.by_name["abseil-cpp"]["licenses"][0]["license"]["id"]
        assert license_id == "Apache-2.0"

    def test_protobuf_license(self) -> None:
        license_id = self.by_name["protobuf"]["licenses"][0]["license"]["id"]
        assert license_id == "BSD-3-Clause"

    def test_nanobind_license(self) -> None:
        license_id = self.by_name["nanobind"]["licenses"][0]["license"]["id"]
        assert license_id == "BSD-3-Clause"

    def test_all_have_versions(self) -> None:
        for name, comp in self.by_name.items():
            assert "version" in comp, f"{name} has no version"

    def test_all_have_purl(self) -> None:
        for name, comp in self.by_name.items():
            assert "purl" in comp, f"{name} has no purl"

    def test_no_version_has_trailing_dot(self) -> None:
        for name, comp in self.by_name.items():
            assert not comp.get("version", "").endswith("."), (
                f"{name} version ends with dot: {comp['version']!r}"
            )

    def test_purl_tag_matches_version(self) -> None:
        for name, comp in self.by_name.items():
            version = comp.get("version", "")
            purl = comp.get("purl", "")
            purl_tag = purl.split("@")[-1] if "@" in purl else ""
            assert purl_tag.lstrip("v") == version, (
                f"{name}: purl tag '{purl_tag}' does not match version '{version}'"
            )

    def test_abseil_cpp_purl_references_abseil(self) -> None:
        assert "abseil" in self.by_name["abseil-cpp"]["purl"]

    def test_protobuf_purl_references_protocolbuffers(self) -> None:
        assert "protocolbuffers" in self.by_name["protobuf"]["purl"]

    def test_nanobind_has_both_vcs_and_distribution_refs(self) -> None:
        refs = self.by_name["nanobind"]["externalReferences"]
        types = {r["type"] for r in refs}
        assert "vcs" in types
        assert "distribution" in types

    def test_protobuf_version_matches_version_variable(self) -> None:
        # CMakeLists.txt sets an explicit Protobuf_VERSION variable whose value
        # may differ from the URL-embedded tag (e.g. tag "v33.6" vs version "6.33.6").
        # The script must prefer the explicit variable over the URL tag.
        text = self._CMAKE_PATH.read_text(encoding="utf-8")
        variables = _parse_cmake_variables(text)
        expected = variables.get("Protobuf_VERSION") or variables.get(
            "PROTOBUF_VERSION"
        )
        if expected is None:
            self.skipTest("Protobuf_VERSION variable not found in CMakeLists.txt")
        protobuf = self.by_name["protobuf"]
        assert protobuf["version"] == expected
        assert f"@v{expected}" in protobuf["purl"]


class TestMainCLI(unittest.TestCase):
    """End-to-end tests for the main() entry point."""

    def _run_main(self, extra_args: list[str]) -> dict:
        cmake_snippet = _GIT_CMAKE
        with tempfile.NamedTemporaryFile(
            suffix=".txt", mode="w", delete=False, encoding="utf-8"
        ) as cmake_f:
            cmake_f.write(cmake_snippet)
            cmake_path = cmake_f.name
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False
        ) as out_f:
            out_path = out_f.name
        saved = sys.argv[:]
        try:
            sys.argv = [
                "extract_cmake_fetchcontent.py",
                "--cmake",
                cmake_path,
                "--output",
                out_path,
                *extra_args,
            ]
            _mod.main()
        finally:
            sys.argv = saved
            Path(cmake_path).unlink(missing_ok=True)
        result = json.loads(Path(out_path).read_text(encoding="utf-8"))
        Path(out_path).unlink(missing_ok=True)
        return result

    def test_schema_field_present(self) -> None:
        bom = self._run_main([])
        assert bom["$schema"] == "https://cyclonedx.org/schema/bom-1.7.schema.json"

    def test_leaf_dependency_entries_added_with_subject(self) -> None:
        bom = self._run_main(["--subject-name", "onnx", "--subject-version", "1.0.0"])
        refs = {d["ref"] for d in bom.get("dependencies", [])}
        nanobind_ref = f"nanobind@{_FIXTURE_NANOBIND_VERSION}"
        assert "onnx@1.0.0" in refs
        assert nanobind_ref in refs

    def test_root_depends_on_components(self) -> None:
        bom = self._run_main(["--subject-name", "onnx", "--subject-version", "1.0.0"])
        root = next(d for d in bom["dependencies"] if d["ref"] == "onnx@1.0.0")
        nanobind_ref = f"nanobind@{_FIXTURE_NANOBIND_VERSION}"
        assert nanobind_ref in root["dependsOn"]

    def test_leaf_entries_have_no_dependson(self) -> None:
        bom = self._run_main(["--subject-name", "onnx", "--subject-version", "1.0.0"])
        nanobind_ref = f"nanobind@{_FIXTURE_NANOBIND_VERSION}"
        leaf = next(
            (d for d in bom["dependencies"] if d["ref"] == nanobind_ref),
            None,
        )
        assert leaf is not None
        assert "dependsOn" not in leaf


if __name__ == "__main__":
    unittest.main()
