# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
r"""Extract CMake FetchContent dependencies and emit a CycloneDX 1.6 BOM.

Parses FetchContent_Declare blocks from a CMakeLists.txt and generates
CycloneDX 1.6 JSON components for each dependency. Supports both
URL-based (e.g. tarball with hash) and git-based (GIT_REPOSITORY/GIT_TAG)
declarations.

Can optionally merge the extracted components into an existing CycloneDX
JSON produced by another tool (e.g. cyclonedx-py environment).

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
from typing import Any, cast

# ---------------------------------------------------------------------------
# CMake parsing helpers
# ---------------------------------------------------------------------------


def _parse_cmake_variables(text: str) -> dict[str, str]:
    """Return all set(VAR VALUE) assignments as a flat dict."""
    variables: dict[str, str] = {}
    for m in re.finditer(
        r'set\s*\(\s*(\w+)\s+"?([^"\)\n]+)"?\s*\)',
        text,
        re.IGNORECASE,
    ):
        variables[m.group(1)] = m.group(2).strip()
    return variables


def _resolve(value: str, variables: dict[str, str]) -> str:
    """Expand ${VAR} references using the variable dict."""

    def _replace(m: re.Match[str]) -> str:
        return variables.get(m.group(1), m.group(0))

    return re.sub(r"\$\{(\w+)\}", _replace, value)


def _find_version_variable(text: str, name: str) -> str | None:
    """Look for set(<Name>_VERSION ...) anywhere in the file."""
    m = re.search(
        rf'set\s*\(\s*{re.escape(name)}_version\s+"?([^"\)\n]+)"?\s*\)',
        text,
        re.IGNORECASE,
    )
    return m.group(1).strip() if m else None


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
# Update this table when a new dependency is added to CMakeLists.txt.
_KNOWN_LICENSES: dict[str, str] = {
    "protobuf": "BSD-3-Clause",
    "nanobind": "BSD-3-Clause",
}


def _github_owner_repo(url: str) -> tuple[str, str] | None:
    """Extract (owner, repo) from a GitHub URL, or None."""
    m = re.match(r"https://github\.com/([^/]+)/([^/.]+)", url)
    return (m.group(1), m.group(2)) if m else None


def _apply_url_fields(comp: dict[str, Any], entry: dict[str, str], text: str) -> None:
    """Populate comp in-place for a URL-based FetchContent entry."""
    url = entry["url"]

    version = _find_version_variable(text, entry["name"])
    if not version:
        v = re.search(r"[/-]v?(\d+\.\d+[\d.]*)", url)
        version = v.group(1) if v else None
    if version:
        comp["version"] = version

    gh = _github_owner_repo(url)
    if gh:
        owner, repo = gh
        tag_m = re.search(r"/download/(v?[^/]+)/", url)
        tag = tag_m.group(1) if tag_m else version
        comp["purl"] = f"pkg:github/{owner}/{repo}@{tag}"

    comp["externalReferences"] = [{"type": "distribution", "url": url}]

    if "hash_alg" in entry:
        # Normalize to CycloneDX hash algorithm names (e.g. SHA1 -> SHA-1)
        alg = entry["hash_alg"].upper()
        alg = re.sub(r"^SHA(\d+)$", r"SHA-\1", alg)
        alg = re.sub(r"^MD(\d+)$", r"MD\1", alg)
        comp["hashes"] = [{"alg": alg, "content": entry["hash_val"]}]


def _apply_git_fields(comp: dict[str, Any], entry: dict[str, str]) -> None:
    """Populate comp in-place for a git-based FetchContent entry."""
    git_url = entry["git_url"]
    tag = entry.get("git_tag", "")
    # Strip leading 'v' to get a clean semver version string
    version = tag.lstrip("v") if tag else None
    if version:
        comp["version"] = version

    gh = _github_owner_repo(git_url)
    if gh:
        owner, repo = gh
        ref = tag or version
        comp["purl"] = (
            f"pkg:github/{owner}/{repo}@{ref}" if ref else f"pkg:github/{owner}/{repo}"
        )

    comp["externalReferences"] = [{"type": "vcs", "url": git_url}]


def _build_component(entry: dict[str, str], text: str) -> dict[str, Any]:
    """Convert one parsed FetchContent entry to a CycloneDX component."""
    name = entry["name"].lower()
    comp: dict[str, Any] = {"type": "library", "name": name}

    if "url" in entry:
        _apply_url_fields(comp, entry, text)
    elif "git_url" in entry:
        _apply_git_fields(comp, entry)

    spdx_id = _KNOWN_LICENSES.get(name)
    if spdx_id:
        comp["licenses"] = [{"license": {"id": spdx_id}}]

    comp["bom-ref"] = f"{name}@{comp['version']}" if "version" in comp else name
    return comp


# ---------------------------------------------------------------------------
# BOM assembly
# ---------------------------------------------------------------------------


def _make_bom(components: list[dict[str, Any]], lifecycle: str) -> dict[str, Any]:
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "serialNumber": f"urn:uuid:{uuid.uuid4()}",
        "version": 1,
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "lifecycles": [{"phase": lifecycle}],
            "tools": {
                "components": [
                    {
                        "type": "application",
                        "name": "extract_cmake_fetchcontent.py",
                        "description": "Extracts CMake FetchContent dependencies into CycloneDX components",
                        "externalReferences": [
                            {"type": "vcs", "url": "https://github.com/onnx/onnx"}
                        ],
                    }
                ]
            },
        },
        "components": components,
    }


def _merge_into(
    base_path: Path, new_components: list[dict[str, Any]], lifecycle: str
) -> dict[str, Any]:
    """Load an existing CycloneDX BOM and append new_components to it."""
    bom = cast("dict[str, Any]", json.loads(base_path.read_text(encoding="utf-8")))
    bom.setdefault("components", []).extend(new_components)
    meta = bom.setdefault("metadata", {})
    meta["lifecycles"] = [{"phase": lifecycle}]
    # Record this script as an additional tool in the BOM provenance chain.
    tools = meta.setdefault("tools", {})
    tools.setdefault("components", []).append(
        {
            "type": "file",
            "name": "extract_cmake_fetchcontent.py",
            "description": "Extracts CMake FetchContent dependencies into CycloneDX components",
            "externalReferences": [
                {"type": "vcs", "url": "https://github.com/onnx/onnx"}
            ],
        }
    )
    # Add cmake components to the dependencies array so they appear in the
    # dependency graph alongside the requirements-file components.
    deps = bom.setdefault("dependencies", [])
    existing_refs = {d["ref"] for d in deps}
    for comp in new_components:
        ref = comp.get("bom-ref")
        if ref and ref not in existing_refs:
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
        help="CycloneDX 1.6 lifecycle phase to annotate on the output BOM (default: build)",
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
        bom.setdefault("metadata", {})["component"] = {
            "type": "library",
            "name": name,
            "version": version,
            "purl": f"pkg:pypi/{name}@{version}",
        }

    out = Path(args.output).resolve()
    out.write_text(json.dumps(bom, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
