# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Generate a CycloneDX 1.6 build-lifecycle SBOM for cibuildwheel release builds.

Intentionally omitted (documented here for transparency):
- C++ compiler: only available inside the cibuildwheel container, not on the host runner
- scikit-build-core: installed in the PEP 517 isolated build environment; not visible here
- cmake: Linux builds run cmake inside the container, so it is not on the host PATH
- abseil-cpp: transitive FetchContent dependency of protobuf; not directly declared
- Python build tools (pip, setuptools, wheel): part of the isolated PEP 517 environment
"""
from __future__ import annotations

import json
import os
import subprocess
import uuid
from datetime import datetime, timezone


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main() -> None:
    platform_tag = os.environ.get("PLATFORM_TAG", "unknown")

    with open("VERSION_NUMBER") as f:
        onnx_version = f.read().strip()

    with open("sbom.cdx.json") as f:
        wheel_sbom = json.load(f)

    protobuf_component: dict | None = None
    for c in wheel_sbom.get("components", []):
        if c.get("name") == "protobuf":
            protobuf_component = c
            break

    cibw_version: str | None = None
    cibw_info = _run(["python", "-m", "pip", "show", "cibuildwheel"])
    if cibw_info:
        for line in cibw_info.splitlines():
            if line.startswith("Version:"):
                cibw_version = line.split(":", 1)[1].strip()
                break

    cmake_version: str | None = None
    cmake_out = _run(["cmake", "--version"])
    if cmake_out:
        first = cmake_out.splitlines()[0]  # "cmake version X.Y.Z"
        cmake_version = first.removeprefix("cmake version ").strip()

    components: list[dict] = []

    if protobuf_component:
        components.append(protobuf_component)

    if cibw_version:
        components.append(
            {
                "type": "library",
                "name": "cibuildwheel",
                "version": cibw_version,
                "purl": f"pkg:pypi/cibuildwheel@{cibw_version}",
            }
        )

    if cmake_version:
        components.append(
            {
                "type": "application",
                "name": "cmake",
                "version": cmake_version,
                "purl": f"pkg:generic/cmake@{cmake_version}",
            }
        )

    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "serialNumber": f"urn:uuid:{uuid.uuid4()}",
        "version": 1,
        "metadata": {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "lifecycles": [{"phase": "build"}],
            "component": {
                "type": "library",
                "name": "onnx",
                "version": onnx_version,
                "purl": f"pkg:pypi/onnx@{onnx_version}",
            },
            "tools": [
                {
                    "type": "application",
                    "name": "generate_build_sbom.py",
                    "vendor": "ONNX Project Contributors",
                }
            ],
        },
        "components": components,
    }

    output_file = f"build-sbom-{platform_tag}.cdx.json"
    with open(output_file, "w") as f:
        json.dump(sbom, f, indent=2)
        f.write("\n")

    print(f"Generated build SBOM: {output_file}")
    if not protobuf_component:
        print("WARNING: protobuf component not found in sbom.cdx.json")
    if not cibw_version:
        print("WARNING: cibuildwheel version not detected")
    if not cmake_version:
        print("INFO: cmake not on host PATH (expected for Linux container builds)")


if __name__ == "__main__":
    main()
