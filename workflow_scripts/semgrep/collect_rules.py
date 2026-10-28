# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from ruamel.yaml import YAML


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect public Semgrep rules, excluding rule test fixtures."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    yaml = YAML(typ="safe")
    rules = []
    for path in sorted(args.source.rglob("*")):
        if path.suffix not in {".yaml", ".yml"}:
            continue
        relative = path.relative_to(args.source)
        if (
            any(part.startswith(".") for part in relative.parts)
            or ".test." in path.name
            or path.name == "template.yaml"
        ):
            continue
        # The repository also contains YAML examples that are not rules.
        text = path.read_text(encoding="utf-8")
        if not text.startswith("rules:") and "\nrules:" not in text:
            continue
        for document in yaml.load_all(text):
            if not isinstance(document, dict) or "rules" not in document:
                continue
            for rule in document["rules"]:
                # Preserve a stable namespace for rules with the same short id.
                rule["id"] = ".".join((*relative.with_suffix("").parts, rule["id"]))
                rules.append(rule)
    if not rules:
        parser.error("No Semgrep rules were found")
    args.output.write_text(json.dumps({"rules": rules}, indent=2), encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger(__name__).info("Collected %d Semgrep rules", len(rules))


if __name__ == "__main__":
    main()
