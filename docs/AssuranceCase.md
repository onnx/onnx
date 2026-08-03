<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# ONNX Security Assurance Case

**Version:** 1.1
**Date:** June 2026
**Project:** ONNX (Open Neural Network Exchange)
**Scope:** ONNX Core (`onnx/onnx`) and the produced Python wheel

This document provides the security assurance case for ONNX Core, supporting the OpenSSF Best Practices Badge application.

## General scope and assurances

The onnx package aims to provide memory-safe parsing of untrusted protobuf bytes.
Using shape/type inference, version update utilities, and model validation is also considered memory-safe.
Resource exhaustion, however, may be triggered from within these utilities and users are advised to guard against this accordingly.

Validation utilities such as `onnx.checker.check_model` are provided on a best-effort basis (e.g. a validated `ModelProto` object may contain `NodeProto` objects that do not adhere to the ONNX specification).

The onnx reference implementation is not yet considered safe for production use on untrusted inputs.


## Threat Model

### Malicious model file

The attacker supplies a malicious ONNX/protobuf file to a user who parses, validates, or runs type/shape inference or version-conversion on it.

- **In scope:** memory safety while parsing, type/shape inference, version conversion, and validation of untrusted model bytes.
- **Out of scope:** resource exhaustion (DoS) from those utilities, and the reference runtime executing untrusted models.

### Supply chain

The attacker compromises a dependency, the build pipeline, or the published artifact — so that a user installing `onnx` (e.g. from PyPI) receives malicious code.

- **In scope:** integrity of the published wheels and statically compiled dependencies.
- **Out of scope:** compromise of a user's own machine or CI, and vulnerabilities in transitive dependencies' upstream code itself.

### External data references

A malicious model references external tensor data via attacker-controlled file paths, attempting to read files outside the model's directory.

- **In scope:** external-data paths are validated and normalized; no resolution outside the model directory.
- **Out of scope:** files the user has explicitly granted the model directory access to.


## Secure Design Principles (Saltzer & Schroeder)

| Principle | Application in ONNX Core |
|-----------|--------------------------|
| Economy of Mechanism | Protocol Buffers for serialization; validation centralized in checker.cc; minimal dependencies |
| Fail-Safe Defaults | Validation on by default; must opt out with `check_model=False`; unknown protobuf fields rejected |
| Complete Mediation | Every model load goes through the validation pipeline; all operator inputs are type- and shape-checked |
| Least Privilege | No elevated privileges required; no network access; file I/O restricted to explicitly specified paths |
| Separation of Privilege | External data loading requires both model reference and file system access; releases require SLSA attestation |
| Least Common Mechanism | No global mutable state; validation is stateless; each API call operates independently |
| Psychological Acceptability | Secure defaults need no configuration; clear validation error messages; type-annotated Python API |


## Common Weaknesses Mitigated

| CWE | Mitigation |
|-----|-----------|
| CWE-787/125 Out-of-bounds R/W | Modern C++ (std::vector, RAII); ASan in CI |
| CWE-20 Input Validation | Comprehensive model validation on load; protobuf schema enforcement; operator shape/type checking |
| CWE-416 Use After Free | RAII/smart pointers (unique_ptr, shared_ptr); ASan in CI; code review |
| CWE-190 Integer Overflow | Checked size arithmetic in tensor allocation; UBSan in CI |
| CWE-22 Path Traversal | External data paths validated and normalized; no auto-resolution outside model directory |
| CWE-78 Command Injection | No shell execution in ONNX Core; no system()/exec() usage; enforced by code review and static analysis |
| OWASP A06 Supply Chain | Dependabot; Sigstore signing; minimal dependency footprint; SBOM generation |
| CWE-79/89/352/434 | Not applicable — ONNX Core is not a web application or database |


## Security Testing

| Method | Details |
|--------|---------|
| Static analysis | CodeQL (GitHub Advanced Security), Clang Static Analyzer, sonarcloud |
| Dynamic analysis | ASan, MSan, UBSan, TSan in CI build matrix |
| Fuzzing | Early stage — OSS-Fuzz harnesses ([onnx/fuzz](https://github.com/onnx/onnx/tree/main/onnx/fuzz)) cover the checker, model loader, text parser, shape inference, version converter, and compose; reference evaluator and external-data parsing are not yet covered. A short smoke run of these harnesses is part of this repo's CI ([fuzz.yml](https://github.com/onnx/onnx/blob/main/.github/workflows/fuzz.yml)); the full OSS-Fuzz continuous campaigns run separately and are not surfaced here |
| Dependency scanning | Dependabot, OpenSSF Scorecard |


## Security Processes

**Vulnerability disclosure**: Reports via GitHub Security Advisories (preferred) or onnx-security@lists.lfaidata.foundation as a fallback; CVE assignment through Linux Foundation CNA. See [SECURITY.md](https://github.com/onnx/onnx/blob/main/SECURITY.md).

**Code review**: All changes require maintainer review; security-sensitive changes require Architecture SIG review; one approval for dependency updates; automated checks must pass before merge ([CODEOWNERS](https://github.com/onnx/onnx/blob/main/CODEOWNERS)).

**Build & distribution**: artifacts signed with Sigstore; PyPI Trusted Publishing with 2FA required for maintainers; SHA256 checksums published; actions pinned to SHA in CI.


## References

- ONNX Security Policy: https://github.com/onnx/onnx/blob/main/SECURITY.md
- ONNX IR Spec: https://github.com/onnx/onnx/blob/main/docs/IR.md
- Model Checker: https://github.com/onnx/onnx/blob/main/onnx/checker.cc
- OpenSSF Best Practices: https://bestpractices.coreinfrastructure.org/
- SLSA Framework: https://slsa.dev/
- CWE Top 25: https://cwe.mitre.org/top25/
- Saltzer & Schroeder Principles: https://web.mit.edu/Saltzer/www/publications/protection/

---

**Document Maintainer**: ONNX Architecture & Infrastructure SIG
**Last Updated**: June 2026
**Review Cycle**: Annual (or upon significant architectural changes)
