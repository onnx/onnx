# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import json
import os
import sys

ver = os.environ["PROTOBUF_VERSION"]
with open("protobuf.tar.gz", "rb") as _f:
    sha = hashlib.sha256(_f.read()).hexdigest()

with open("sbom.cdx.json") as f:
    sbom = json.load(f)

patched = False
for c in sbom["components"]:
    if c["name"] == "protobuf":
        c["version"] = ver
        c["purl"] = f"pkg:github/protocolbuffers/protobuf@v{ver}"

        dist_ref = next(
            (r for r in c["externalReferences"] if r.get("type") == "distribution"),
            None,
        )
        if dist_ref is None:
            sys.exit(
                "ERROR: protobuf component has no 'distribution' externalReference in sbom.cdx.json"
            )
        dist_ref["url"] = (
            f"https://github.com/protocolbuffers/protobuf/releases/download/v{ver}/protobuf-{ver}.tar.gz"
        )

        sha256_entry = next((h for h in c["hashes"] if h.get("alg") == "SHA-256"), None)
        if sha256_entry is None:
            sys.exit(
                "ERROR: protobuf component has no SHA-256 hash entry in sbom.cdx.json"
            )
        sha256_entry["content"] = sha

        patched = True
        break

if not patched:
    sys.exit("ERROR: protobuf component not found in sbom.cdx.json")

with open("sbom.cdx.json", "w") as f:
    json.dump(sbom, f, indent=2)
    f.write("\n")
