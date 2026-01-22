<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# Verifying ONNX PyPI Releases with Sigstore Attestations

ONNX PyPI releases are published with **Sigstore-based digital attestations**
in accordance with **PEP 740 (Digital Attestations for Python Packages)**.
These attestations allow consumers to cryptographically verify the provenance,
integrity, and publishing identity of ONNX distribution artifacts.

This document describes:

- what Sigstore attestations are and why they matter,
- what properties are verified,
- how to verify ONNX releases step by step,
- how to use verification in automation and CI,
- common failure modes and troubleshooting guidance.

---

## Background and Motivation

Traditional Python package verification relies primarily on TLS transport
security and optional hash checking. While useful, these mechanisms do not
answer critical supply-chain questions such as:

- Who built this artifact?
- Was it built by the project’s official CI?
- Has it been replaced or modified after publication?

Sigstore attestations address these gaps by providing **cryptographically
verifiable provenance** tied to publicly auditable identities and logs.

ONNX uses Sigstore attestations to strengthen trust in its published releases.

---

## High-Level Architecture

For each ONNX artifact uploaded to PyPI:

1. The artifact is built in ONNX’s official GitHub Actions CI.
2. The build produces a signed attestation describing:
   - the artifact digest,
   - the build environment,
   - the publishing identity.
3. The attestation is signed using a short-lived certificate issued by
   Sigstore Fulcio.
4. The signature is recorded in Sigstore’s Rekor transparency log.
5. PyPI distributes the artifact and its associated attestation.

Consumers can independently verify all of the above.

---

## What Is Verified

Verification checks the following properties.

### Artifact Integrity

The local artifact’s SHA-256 digest must match the digest recorded in the
attestation. This ensures the file has not been altered.

---

### Signature Validity

The attestation signature must:

- be cryptographically valid,
- chain to Sigstore’s trusted root,
- have been created during the certificate’s validity window.

---

### Transparency Log Inclusion

The signature must appear in the Rekor transparency log with a valid inclusion
proof. This prevents undetectable signature replacement.

---

### Publisher Identity

Verification ensures the artifact was built and published by the expected
GitHub repository (`onnx/onnx`) using GitHub Actions OIDC-based trusted
publishing.

---

## Prerequisites

- Python **3.10 or newer**
- Linux, macOS, or **Windows Subsystem for Linux (WSL)** (recommended on Windows)
- Network access to PyPI and Sigstore services

---

## Installation

Install the verification tool:

```bash
pip install pypi-attestations
```

---

## Example: Verifying an ONNX Release from PyPI

This section demonstrates a complete verification workflow using a real ONNX
release artifact.

### Example Artifact

- Project: ONNX
- Version: `1.20.1`
- Artifact: Windows CPython 3.13 wheel  
  `onnx-1.20.1-cp313-cp313t-win_amd64.whl`

---

### Verification Command

```bash
pypi-attestations verify pypi \
  --repository https://github.com/onnx/onnx \
  pypi:onnx-1.20.1-cp313-cp313t-win_amd64.whl
```

The `--repository` option restricts verification to attestations whose signing
identity corresponds to the specified GitHub repository.

---

### Expected Output

```text
✓ Attestation verified
✓ Transparency log entry verified
✓ Identity policy satisfied
```

A non-zero exit status indicates verification failure.

---

## Verifying Locally Downloaded Artifacts

Artifacts downloaded manually can also be verified:

```bash
pypi-attestations verify pypi \
  --repository https://github.com/onnx/onnx \
  ./onnx-1.20.1-cp313-cp313t-win_amd64.whl
```

The attestation is still fetched from PyPI; only the artifact file is local.

---

## Verifying All Wheels for a Release

To verify all wheels for a given release, iterate over the artifacts:

```bash
for wheel in onnx-1.20.1-*.whl; do
  pypi-attestations verify pypi \
    --repository https://github.com/onnx/onnx \
    "$wheel"
  done
```

This is useful for auditing mirrors or internal artifact caches.

---

## Security Model Summary

Sigstore verification provides the following guarantees:

- **Integrity:** the artifact is unmodified
- **Authenticity:** the artifact was built by ONNX CI
- **Transparency:** the signature is publicly auditable
- **Identity:** the artifact originated from `onnx/onnx`

These guarantees significantly strengthen ONNX’s supply-chain security.

---

## References

- PEP 740 – Digital Attestations for Python Packages  
  https://peps.python.org/pep-0740/

- Sigstore  
  https://www.sigstore.dev/

- PyPI Attestations  
  https://pypi.org/project/pypi-attestations/
