# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
r"""Extract CMake FetchContent dependencies and emit a CycloneDX 1.7 BOM.

Parses FetchContent_Declare blocks from a CMakeLists.txt and generates
CycloneDX 1.7 JSON components for each dependency. Supports both
URL-based (e.g. tarball with hash) and git-based (GIT_REPOSITORY/GIT_TAG)
declarations.

Can optionally merge the extracted components into an existing CycloneDX
JSON produced by another tool (e.g. cyclonedx-py environment).

This script captures build-time provenance for dependencies fetched during
compilation. It does not imply that these components are runtime dependencies
of the installed wheel.

Usage:
    # Standalone cmake-only BOM
    python tools/extract_cmake_fetchcontent.py --output cmake-deps.cdx.json

    # Merge cmake components into an existing Python-env BOM
    python tools/extract_cmake_fetchcontent.py \
        --merge-into build-python.cdx.json \
        --lifecycle build \
        --output onnx-build.cdx.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from cyclonedx.model import (
    ExternalReference,
    ExternalReferenceType,
    HashAlgorithm,
    HashType,
    XsUri,
)
from cyclonedx.model.bom import Bom
from cyclonedx.model.component import Component, ComponentType
from cyclonedx.model.contact import OrganizationalEntity
from cyclonedx.model.dependency import Dependency
from cyclonedx.model.license import DisjunctiveLicense
from cyclonedx.model.lifecycle import LifecyclePhase, PredefinedLifecycle
from cyclonedx.output.json import JsonV1Dot7
from packageurl import PackageURL

# ---------------------------------------------------------------------------
# CMake parsing helpers
# ---------------------------------------------------------------------------


def _parse_cmake_variables(text: str) -> dict[str, str]:
    """Return all set(VAR VALUE) assignments as a flat dict."""
    variables: dict[str, str] = {}
    # Two explicit alternatives avoid the optional-quote ambiguity that causes
    # O(n²) backtracking: quoted "([^"]*)" is always bounded; unquoted
    # ([^"\s)\n]+) excludes whitespace so it cannot overlap with the \s* that
    # follows, keeping backtracking O(n) even on malformed input.
    for m in re.finditer(
        r'set\s*\(\s*(\w+)\s+(?:"([^"]*)"|((?:[^"\s)\n])+))\s*\)',
        text,
        re.IGNORECASE,
    ):
        value = (m.group(2) if m.group(2) is not None else m.group(3) or "").strip()
        if value:
            variables[m.group(1)] = value
    return variables


def _resolve(value: str, variables: dict[str, str]) -> str:
    """Expand ${VAR} references using the variable dict."""

    def _replace(m: re.Match[str]) -> str:
        return variables.get(m.group(1), m.group(0))

    return re.sub(r"\$\{(\w+)\}", _replace, value)


def _find_version_variable(text: str, name: str) -> str | None:
    """Look for set(<Name>_VERSION ...) anywhere in the file."""
    m = re.search(
        rf'set\s*\(\s*{re.escape(name)}_version\s+(?:"([^"]*)"|((?:[^"\s)\n])+))\s*\)',
        text,
        re.IGNORECASE,
    )
    if not m:
        return None
    value = (m.group(1) if m.group(1) is not None else m.group(2) or "").strip()
    return value or None


def _parse_fetchcontent_declares(
    text: str, variables: dict[str, str]
) -> list[dict[str, str]]:
    """Return one dict per FetchContent_Declare block with resolved values."""
    results = []
    pattern = re.compile(
        r"FetchContent_Declare\s*\(\s*(\w+)\s+(.*?)\)",
        re.DOTALL | re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        name = match.group(1)
        body = match.group(2)
        entry: dict[str, str] = {"name": name}

        # URL-based
        m = re.search(r"\bURL\s+(\S+)", body)
        if m:
            entry["url"] = _resolve(m.group(1), variables)

        m = re.search(r"URL_HASH\s+(\w+)=(\S+)", body)
        if m:
            entry["hash_alg"] = m.group(1).upper()
            entry["hash_val"] = _resolve(m.group(2), variables)

        # Git-based
        m = re.search(r"GIT_REPOSITORY\s+(\S+)", body)
        if m:
            entry["git_url"] = _resolve(m.group(1), variables)

        m = re.search(r"GIT_TAG\s+(\S+)", body)
        if m:
            entry["git_tag"] = _resolve(m.group(1), variables)

        results.append(entry)
    return results


# ---------------------------------------------------------------------------
# CycloneDX component builder
# ---------------------------------------------------------------------------

# SPDX license IDs for known FetchContent dependencies.
# Keys are lowercased FetchContent names (as used in FetchContent_Declare).
# Update this table when a new dependency is added to CMakeLists.txt.
_KNOWN_LICENSES: dict[str, str] = {
    "absl": "Apache-2.0",
    "protobuf": "BSD-3-Clause",
    "nanobind": "BSD-3-Clause",
}

# Maps lowercased FetchContent names to their canonical package names.
# FetchContent names are often short aliases; canonical names match upstream.
_CANONICAL_NAMES: dict[str, str] = {
    "absl": "abseil-cpp",
}

_HASH_ALG_MAP: dict[str, HashAlgorithm] = {
    "MD5": HashAlgorithm.MD5,
    "SHA1": HashAlgorithm.SHA_1,
    "SHA256": HashAlgorithm.SHA_256,
    "SHA384": HashAlgorithm.SHA_384,
    "SHA512": HashAlgorithm.SHA_512,
}

_LIFECYCLE_PHASE_MAP: dict[str, LifecyclePhase] = {
    "design": LifecyclePhase.DESIGN,
    "pre-build": LifecyclePhase.PRE_BUILD,
    "build": LifecyclePhase.BUILD,
    "post-build": LifecyclePhase.POST_BUILD,
    "operations": LifecyclePhase.OPERATIONS,
    "discovery": LifecyclePhase.DISCOVERY,
    "decommission": LifecyclePhase.DECOMMISSION,
}


def _github_owner_repo(url: str) -> tuple[str, str] | None:
    """Extract (owner, repo) from a GitHub URL, or None."""
    m = re.match(r"https://github\.com/([^/]+)/([^/.]+)", url)
    return (m.group(1), m.group(2)) if m else None


def _build_component(entry: dict[str, str], text: str) -> Component:
    """Convert one parsed FetchContent entry to a CycloneDX Component."""
    fetch_name = entry["name"].lower()
    canonical_name = _CANONICAL_NAMES.get(fetch_name, fetch_name)

    version: str | None = None
    purl: PackageURL | None = None
    ext_refs: list[ExternalReference] = []
    hashes: list[HashType] = []

    if "url" in entry:
        url = entry["url"]
        version = _find_version_variable(text, entry["name"])
        if not version:
            v = re.search(r"[/-]v?(\d+\.\d+[\d.]*)", url)
            # Strip trailing dot that the greedy [\d.]* may pull in from ".tar.gz"
            version = v.group(1).rstrip(".") if v else None

        gh = _github_owner_repo(url)
        if gh:
            owner, repo = gh
            # Prefer the explicit version variable (with v-prefix) over the URL tag so
            # that comp["version"] and the purl tag are always consistent (e.g. protobuf
            # uses tag v33.6 in the download URL but reports its version as 6.33.6).
            tag: str | None
            if version:
                tag = f"v{version}"
            else:
                tag_m = re.search(r"/download/(v?[^/]+)/", url)
                tag = tag_m.group(1) if tag_m else None
            if tag:
                purl = PackageURL(
                    type="github", namespace=owner, name=repo, version=tag
                )

        ext_refs.append(
            ExternalReference(type=ExternalReferenceType.DISTRIBUTION, url=XsUri(url))
        )

        if "hash_alg" in entry:
            alg = _HASH_ALG_MAP.get(entry["hash_alg"].upper())
            if alg is not None:
                hashes.append(HashType(alg=alg, content=entry["hash_val"]))

    elif "git_url" in entry:
        git_url = entry["git_url"]
        git_tag = entry.get("git_tag", "")
        # Strip leading 'v' to get a clean semver version string
        version = git_tag.lstrip("v") if git_tag else None

        gh = _github_owner_repo(git_url)
        if gh:
            owner, repo = gh
            ref = git_tag or version
            purl = PackageURL(type="github", namespace=owner, name=repo, version=ref)

        ext_refs.append(
            ExternalReference(type=ExternalReferenceType.VCS, url=XsUri(git_url))
        )
        # Add a point-in-time distribution URL so consumers have a verifiable artifact
        # reference alongside the VCS pointer.
        if gh and git_tag:
            owner, repo = gh
            ext_refs.append(
                ExternalReference(
                    type=ExternalReferenceType.DISTRIBUTION,
                    url=XsUri(
                        f"https://github.com/{owner}/{repo}/archive/refs/tags/{git_tag}.tar.gz"
                    ),
                )
            )

    spdx_id = _KNOWN_LICENSES.get(fetch_name)
    licenses = {DisjunctiveLicense(id=spdx_id)} if spdx_id else set()

    bom_ref = f"{canonical_name}@{version}" if version else canonical_name

    return Component(
        type=ComponentType.LIBRARY,
        name=canonical_name,
        version=version,
        purl=purl,
        bom_ref=bom_ref,
        licenses=licenses,
        hashes=hashes,
        external_references=ext_refs,
    )


# ---------------------------------------------------------------------------
# BOM assembly
# ---------------------------------------------------------------------------

_TOOL_COMPONENT = Component(
    type=ComponentType.APPLICATION,
    name="extract_cmake_fetchcontent.py",
    description="Extracts CMake FetchContent dependencies into CycloneDX components",
    external_references=[
        ExternalReference(
            type=ExternalReferenceType.VCS,
            url=XsUri("https://github.com/onnx/onnx"),
        )
    ],
)

_ONNX_MANUFACTURER = OrganizationalEntity(name="ONNX Project Contributors")
_ONNX_SUPPLIER = OrganizationalEntity(
    name="Linux Foundation",
    urls=[XsUri("https://www.linuxfoundation.org")],
)


def _make_bom(components: list[Component], lifecycle: str) -> Bom:
    """Build a CycloneDX Bom from a list of components."""
    bom = Bom()
    phase = _LIFECYCLE_PHASE_MAP.get(lifecycle, LifecyclePhase.BUILD)
    bom.metadata.lifecycles.add(PredefinedLifecycle(phase=phase))
    bom.metadata.manufacturer = _ONNX_MANUFACTURER
    bom.metadata.supplier = _ONNX_SUPPLIER
    bom.metadata.tools.components.add(_TOOL_COMPONENT)
    bom.components.update(components)
    return bom


_SCHEMA_URL = "http://cyclonedx.org/schema/bom-1.7.schema.json"


def _merge_into(
    base_path: Path, new_components: list[Component], lifecycle: str
) -> dict[str, Any]:
    """Load an existing CycloneDX BOM JSON and append new_components to it."""
    # Serialize the new components via a temporary BOM to get schema-correct dicts.
    tmp_bom = Bom()
    tmp_bom.components.update(new_components)
    tmp_data = json.loads(JsonV1Dot7(tmp_bom).output_as_string())
    component_dicts: list[dict[str, Any]] = tmp_data.get("components", [])

    bom: dict[str, Any] = json.loads(base_path.read_text(encoding="utf-8"))
    bom.setdefault("$schema", _SCHEMA_URL)
    bom.setdefault("components", []).extend(component_dicts)
    meta = bom.setdefault("metadata", {})
    phase = _LIFECYCLE_PHASE_MAP.get(lifecycle, LifecyclePhase.BUILD)
    meta["lifecycles"] = [{"phase": phase.value}]
    meta["manufacturer"] = {"name": _ONNX_MANUFACTURER.name}
    meta["supplier"] = {
        "name": _ONNX_SUPPLIER.name,
        "url": ["https://www.linuxfoundation.org"],
    }
    meta.setdefault("tools", {}).setdefault("components", []).append(
        {
            "type": "application",
            "name": _TOOL_COMPONENT.name,
            "description": _TOOL_COMPONENT.description,
            "externalReferences": [
                {"type": "vcs", "url": "https://github.com/onnx/onnx"}
            ],
        }
    )
    # Add cmake components to the dependency graph alongside requirements-file components.
    deps = bom.setdefault("dependencies", [])
    existing_refs = {d["ref"] for d in deps}
    for comp in new_components:
        ref = str(comp.bom_ref)
        if ref and ref not in existing_refs:
            deps.append({"ref": ref})
    return bom


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_bundled_sbom(cmake_path: str, name: str, version: str) -> str:
    """Return a CycloneDX 1.7 BOM JSON string for the C++ libraries bundled in an ONNX wheel.

    Parses FetchContent_Declare blocks from *cmake_path* and sets *name*/*version*
    as the root component with all fetched libraries as dependencies, suitable for
    embedding in a wheel per PEP 770.
    """
    text = Path(cmake_path).read_text(encoding="utf-8")
    variables = _parse_cmake_variables(text)
    raw = _parse_fetchcontent_declares(text, variables)
    components = [_build_component(entry, text) for entry in raw]

    bom = _make_bom(components, "post-build")

    root = Component(
        type=ComponentType.LIBRARY,
        name=name,
        version=version,
        description="Open Neural Network Exchange (ONNX) — open format for AI/ML models",
        purl=PackageURL(type="pypi", name=name, version=version),
        bom_ref=f"{name}@{version}",
    )
    bom.metadata.component = root

    lib_deps = [Dependency(ref=c.bom_ref) for c in components]
    bom.dependencies.add(Dependency(ref=root.bom_ref, dependencies=lib_deps))
    bom.dependencies.update(lib_deps)

    return str(JsonV1Dot7(bom).output_as_string(indent=2))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cmake",
        default="CMakeLists.txt",
        help="Path to CMakeLists.txt (default: CMakeLists.txt)",
    )
    parser.add_argument(
        "--merge-into",
        metavar="PATH",
        help="Existing CycloneDX JSON to merge cmake components into",
    )
    parser.add_argument(
        "--lifecycle",
        default="build",
        choices=list(_LIFECYCLE_PHASE_MAP),
        help="CycloneDX 1.7 lifecycle phase to annotate on the output BOM (default: build)",
    )
    parser.add_argument("--output", required=True, help="Output CycloneDX JSON file")
    parser.add_argument(
        "--subject-name",
        metavar="NAME",
        help="Package name to set as metadata.component (e.g. onnx or onnx-weekly)",
    )
    parser.add_argument(
        "--subject-version",
        metavar="VERSION",
        help="Package version to set as metadata.component",
    )
    args = parser.parse_args()

    text = Path(args.cmake).read_text(encoding="utf-8")
    variables = _parse_cmake_variables(text)
    raw = _parse_fetchcontent_declares(text, variables)
    components = [_build_component(entry, text) for entry in raw]

    if args.merge_into:
        merged = _merge_into(Path(args.merge_into), components, args.lifecycle)
        out_text = json.dumps(merged, indent=2)
    else:
        bom = _make_bom(components, args.lifecycle)

        if args.subject_name and args.subject_version:
            name, version = args.subject_name, args.subject_version
            root = Component(
                type=ComponentType.LIBRARY,
                name=name,
                version=version,
                description="Open Neural Network Exchange (ONNX) — open format for AI/ML models",
                purl=PackageURL(type="pypi", name=name, version=version),
                bom_ref=f"{name}@{version}",
            )
            bom.metadata.component = root

            lib_deps = [Dependency(ref=c.bom_ref) for c in components]
            bom.dependencies.add(Dependency(ref=root.bom_ref, dependencies=lib_deps))
            bom.dependencies.update(lib_deps)

        out_text = JsonV1Dot7(bom).output_as_string(indent=2)

    Path(args.output).resolve().write_text(out_text, encoding="utf-8")


if __name__ == "__main__":
    main()
