# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Check that NOTICE matches the dependency metadata in sbom.cdx.json.

NOTICE is the third-party attribution shipped inside the wheels, so its
versions and licenses have to follow the dependencies CMake actually builds.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main() -> int:
    sbom = json.loads((ROOT / "sbom.cdx.json").read_text(encoding="utf-8"))
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")

    # "## <name>" followed by "Version: <version>" and "License: <license>".
    sections = {
        match["name"]: match
        for match in re.finditer(
            r"^## (?P<name>\S+)$.*?^Version: (?P<version>\S+)$.*?^License: (?P<license>\S+)",
            notice,
            re.MULTILINE | re.DOTALL,
        )
    }

    problems = []
    for component in sbom["components"]:
        if component["type"] != "library":
            continue  # protoc binaries are tools, not linked into the wheel
        name = component["name"]
        section = sections.get(name)
        if section is None:
            problems.append(f"NOTICE has no '## {name}' section")
            continue
        expected_license = component["licenses"][0]["license"]["id"]
        if section["version"] != component["version"]:
            problems.append(
                f"{name}: NOTICE says version {section['version']}, "
                f"sbom.cdx.json says {component['version']}"
            )
        if section["license"] != expected_license:
            problems.append(
                f"{name}: NOTICE says license {section['license']}, "
                f"sbom.cdx.json says {expected_license}"
            )

    if problems:
        print("NOTICE is out of sync with sbom.cdx.json:")
        print("\n".join(f"  {problem}" for problem in problems))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
