<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# Verifying ONNX PyPI Releases with Sigstore Attestations

ONNX releases published through **Create Releases** have two complementary
verification paths: PyPI publisher attestations (PEP 740) and GitHub build
provenance attestations. This applies to release candidates as well as final
releases. See the [release checklist](OnnxReleases.md#build-and-publish-packages)
for the publication target and workflow inputs.

## Security Guarantees

PyPI attestation verification checks that:

- the artifact **has not been modified**,
- the signature is **publicly auditable** in Sigstore’s transparency log,
- the publisher identity matches **`onnx/onnx`**.

This does not replace installation tests, partner validation, or review of the
build's source commit. Build provenance is verified separately below.

## Verify the PyPI publisher

Use the exact wheel or sdist filename from the version being verified. The
example below uses PyPI. RCs on PyPI use the same verification procedure as
final releases.

```bash
python -m pip install pypi-attestations

pypi-attestations verify pypi \
  --repository https://github.com/onnx/onnx \
  pypi:onnx-1.20.1-cp313-cp313t-win_amd64.whl
```

## Verify GitHub build provenance

Download the wheel or sdist being checked, then verify its GitHub attestation
using the procedure in [Security Policy](../SECURITY.md#supply-chain-security).
Compare the attested repository, workflow, source commit, and artifact digest
with the intended release run and tag. A successful verification establishes
the recorded provenance, not that all release tests passed.

For a TestPyPI rehearsal, download the artifact from TestPyPI and use this
GitHub provenance check. The `pypi:` lookup in the example above queries PyPI,
not TestPyPI.

## References

- PEP 740 – Digital Attestations for Python Packages
  https://peps.python.org/pep-0740/

- Sigstore
  https://www.sigstore.dev/

- PyPI Attestations
  https://pypi.org/project/pypi-attestations/
