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
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


def _github_owner_repo(url: str) -> tuple[str, str] | None:
    """Extract (owner, repo) from a GitHub URL, or None."""
    m = re.match(r"https://github\.com/([^/]+)/([^/.]+)", url)
    return (m.group(1), m.group(2)) if m else None


def _build_component(entry: dict[str, str], text: str) -> dict[str, Any]:
    """Convert one parsed FetchContent entry to a CycloneDX component."""
    fetch_name = entry["name"].lower()
    canonical_name = _CANONICAL_NAMES.get(fetch_name, fetch_name)
    comp: dict[str, Any] = {"type": "library", "name": canonical_name}

    if "url" in entry:
        url = entry["url"]
        version = _find_version_variable(text, entry["name"])
        if not version:
            v = re.search(r"[/-]v?(\d+\.\d+[\d.]*)", url)
            # Strip trailing dot that the greedy [\d.]* may pull in from ".tar.gz"
            version = v.group(1).rstrip(".") if v else None
        if version:
            comp["version"] = version

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
                comp["purl"] = f"pkg:github/{owner}/{repo}@{tag}"

        comp["externalReferences"] = [{"type": "distribution", "url": url}]

        if "hash_alg" in entry:
            # Normalize to CycloneDX hash algorithm names (e.g. SHA1 -> SHA-1)
            alg = re.sub(r"^SHA(\d+)$", r"SHA-\1", entry["hash_alg"].upper())
            comp["hashes"] = [{"alg": alg, "content": entry["hash_val"]}]

    elif "git_url" in entry:
        git_url = entry["git_url"]
        git_tag = entry.get("git_tag", "")
        # Strip leading 'v' to get a clean semver version string
        version = git_tag.lstrip("v") if git_tag else None
        if version:
            comp["version"] = version

        gh = _github_owner_repo(git_url)
        if gh:
            owner, repo = gh
            ref = git_tag or version
            comp["purl"] = (
                f"pkg:github/{owner}/{repo}@{ref}"
                if ref
                else f"pkg:github/{owner}/{repo}"
            )

        refs: list[dict[str, str]] = [{"type": "vcs", "url": git_url}]
        # Add a point-in-time distribution URL so consumers have a verifiable artifact
        # reference alongside the VCS pointer.
        if gh and git_tag:
            owner, repo = gh
            refs.append(
                {
                    "type": "distribution",
                    "url": f"https://github.com/{owner}/{repo}/archive/refs/tags/{git_tag}.tar.gz",
                }
            )
        comp["externalReferences"] = refs

    spdx_id = _KNOWN_LICENSES.get(fetch_name)
    if spdx_id:
        comp["licenses"] = [{"license": {"id": spdx_id}}]

    comp["bom-ref"] = (
        f"{canonical_name}@{comp['version']}" if "version" in comp else canonical_name
    )
    return comp


# ---------------------------------------------------------------------------
# BOM assembly
# ---------------------------------------------------------------------------

_TOOL_ENTRY: dict[str, Any] = {
    "type": "application",
    "name": "extract_cmake_fetchcontent.py",
    "description": "Extracts CMake FetchContent dependencies into CycloneDX components",
    "externalReferences": [{"type": "vcs", "url": "https://github.com/onnx/onnx"}],
}

_ONNX_MANUFACTURER: dict[str, Any] = {"name": "ONNX Project Contributors"}
_ONNX_SUPPLIER: dict[str, Any] = {
    "name": "Linux Foundation",
    "url": ["https://www.linuxfoundation.org"],
}


_SCHEMA_URL = "https://cyclonedx.org/schema/bom-1.7.schema.json"


def _make_bom(components: list[dict[str, Any]], lifecycle: str) -> dict[str, Any]:
    return {
        "$schema": _SCHEMA_URL,
        "bomFormat": "CycloneDX",
        "specVersion": "1.7",
        "serialNumber": f"urn:uuid:{uuid.uuid4()}",
        "version": 1,
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "lifecycles": [{"phase": lifecycle}],
            "manufacturer": _ONNX_MANUFACTURER,
            "supplier": _ONNX_SUPPLIER,
            "tools": {"components": [_TOOL_ENTRY]},
        },
        "components": components,
    }


def _merge_into(
    base_path: Path, new_components: list[dict[str, Any]], lifecycle: str
) -> dict[str, Any]:
    """Load an existing CycloneDX BOM and append new_components to it."""
    bom: dict[str, Any] = json.loads(base_path.read_text(encoding="utf-8"))
    bom.setdefault("$schema", _SCHEMA_URL)
    bom.setdefault("components", []).extend(new_components)
    meta = bom.setdefault("metadata", {})
    meta["lifecycles"] = [{"phase": lifecycle}]
    meta["manufacturer"] = _ONNX_MANUFACTURER
    meta["supplier"] = _ONNX_SUPPLIER
    meta.setdefault("tools", {}).setdefault("components", []).append(_TOOL_ENTRY)
    # Add cmake components to the dependency graph alongside requirements-file components.
    deps = bom.setdefault("dependencies", [])
    existing_refs = {d["ref"] for d in deps}
    for comp in new_components:
        ref = comp.get("bom-ref")
        if ref and ref not in existing_refs:
            deps.append({"ref": ref})
    return bom


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_bundled_sbom(cmake_path: str, name: str, version: str) -> dict[str, Any]:
    """Return a CycloneDX 1.7 BOM dict for the C++ libraries bundled in an ONNX wheel.

    Parses FetchContent_Declare blocks from *cmake_path* and sets *name*/*version*
    as the root component with all fetched libraries as dependencies, suitable for
    embedding in a wheel per PEP 770.
    """
    text = Path(cmake_path).read_text(encoding="utf-8")
    variables = _parse_cmake_variables(text)
    raw = _parse_fetchcontent_declares(text, variables)
    components = [_build_component(entry, text) for entry in raw]
    bom = _make_bom(components, "post-build")

    root_ref = f"{name}@{version}"
    bom.setdefault("metadata", {})["component"] = {
        "type": "library",
        "name": name,
        "version": version,
        "description": "Open Neural Network Exchange (ONNX) — open format for AI/ML models",
        "purl": f"pkg:pypi/{name}@{version}",
        "bom-ref": root_ref,
    }
    component_refs = [c["bom-ref"] for c in components if "bom-ref" in c]
    deps = bom.setdefault("dependencies", [])
    deps.insert(0, {"ref": root_ref, "dependsOn": component_refs})
    for ref in component_refs:
        deps.append({"ref": ref})
    return bom


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
        choices=[
            "design",
            "pre-build",
            "build",
            "post-build",
            "operations",
            "discovery",
            "decommission",
        ],
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
        bom = _merge_into(Path(args.merge_into), components, args.lifecycle)
    else:
        bom = _make_bom(components, args.lifecycle)

    if args.subject_name and args.subject_version:
        name, version = args.subject_name, args.subject_version
        root_ref = f"{name}@{version}"
        bom.setdefault("metadata", {})["component"] = {
            "type": "library",
            "name": name,
            "version": version,
            "description": "Open Neural Network Exchange (ONNX) — open format for AI/ML models",
            "purl": f"pkg:pypi/{name}@{version}",
            "bom-ref": root_ref,
        }
        # Emit a dependency graph so consumers can see which C++ libraries the root
        # component depends on. Leaf entries (no dependsOn) are included for each
        # component so the graph is complete per CycloneDX spec guidance.
        component_refs = [c["bom-ref"] for c in components if "bom-ref" in c]
        deps = bom.setdefault("dependencies", [])
        deps.insert(0, {"ref": root_ref, "dependsOn": component_refs})
        existing_refs = {d["ref"] for d in deps}
        for ref in component_refs:
            if ref not in existing_refs:
                deps.append({"ref": ref})

    out = Path(args.output).resolve()
    out.write_text(json.dumps(bom, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
