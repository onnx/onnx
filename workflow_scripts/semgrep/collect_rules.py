# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ruamel.yaml import YAML


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect public Semgrep rules, excluding rule test fixtures."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--exclude", type=Path, required=True)
    args = parser.parse_args()
    excluded = {
        line.strip()
        for line in args.exclude.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    yaml = YAML(typ="safe")
    rules = []
    for path in sorted(args.source.rglob("*")):
        relative = path.relative_to(args.source)
        if (
            path.suffix not in {".yaml", ".yml"}
            or any(part.startswith(".") for part in relative.parts)
            or ".test." in path.name
            or path.name == "template.yaml"
        ):
            continue
        for document in yaml.load_all(path.read_text(encoding="utf-8")):
            if not isinstance(document, dict):
                continue
            for rule in document.get("rules", []):
                # Preserve a stable namespace for rules with the same short id.
                rule["id"] = ".".join((*relative.with_suffix("").parts, rule["id"]))
                if rule["id"] not in excluded:
                    rules.append(rule)
    if not rules:
        parser.error("No Semgrep rules were found")
    args.output.write_text(json.dumps({"rules": rules}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
