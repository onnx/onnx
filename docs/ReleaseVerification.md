<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# Verifying ONNX PyPI Releases with Sigstore Attestations

ONNX PyPI releases include **Sigstore attestations** compliant with **PEP 740**, enabling cryptographic verification of **integrity, provenance, and publisher identity**.

## Security Guarantees

Verification confirms that:

- the artifact **has not been modified**,
- it was **built and published by ONNX CI**,
- the signature is **publicly auditable** in Sigstore’s transparency log,
- the publisher identity matches **`onnx/onnx`**.

## Verify a Release

```bash
pip install pypi-attestations

pypi-attestations verify pypi \
  --repository https://github.com/onnx/onnx \
  pypi:onnx-1.20.1-cp313-cp313t-win_amd64.whl
```

## References

- PEP 740 – Digital Attestations for Python Packages
  https://peps.python.org/pep-0740/

- Sigstore
  https://www.sigstore.dev/

- PyPI Attestations
  https://pypi.org/project/pypi-attestations/
