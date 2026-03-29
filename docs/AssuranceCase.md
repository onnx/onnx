<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# ONNX Security Assurance Case

**Version:** 1.0 *(DRAFT)*
**Date:** February 2026
**Project:** ONNX (Open Neural Network Exchange)
**Scope:** ONNX Core (`onnx/onnx`) and the produced Python wheel

This document provides the security assurance case for ONNX Core, supporting the OpenSSF Best Practices Badge application.

> **Out of scope**: Inference engines and execution providers that consume ONNX models (e.g. ONNX Runtime) are separate projects and are not covered by this document.

---

## 1. Threat Model

> **Note**: No specific threat modeling template (e.g. STRIDE, PASTA) was used. The threats below are identified based on the system's attack surface and known risks for serialization/parsing libraries.

ONNX Core is the reference implementation of the ONNX standard, consisting of the IR specification, model validator ([checker.cc](https://github.com/onnx/onnx/blob/main/onnx/checker.cc)), shape inference, protobuf serialization, and Python bindings.

### 1.1 Key Threats

| ID | Threat | Impact | Likelihood |
|----|--------|--------|------------|
| T1 | Malicious model file causing code execution | Critical | Medium |
| T2 | Model deserialization causing memory corruption | High | Medium |
| T3 | DoS via crafted model (resource exhaustion during parsing) | Medium | High |
| T4 | Supply chain compromise (malicious dependency or build) | Critical | Low |
| T5 | Information disclosure via model inspection | Medium | Medium |
| T6 | Compromised PyPI packages | High | Low |

### 1.2 Trust Boundaries

```
┌─────────────────────────────────────────────────────┐
│  Untrusted Zone                                     │
│  - User-provided ONNX models, third-party deps      │
└────────────────┬────────────────────────────────────┘
                 │ Boundary #1  (validation, schema checks, size limits)
                 ▼
┌─────────────────────────────────────────────────────┐
│  ONNX Core Processing Environment                   │
│  - Model parser/deserializer (protobuf)             │
│  - Model validator (checker.cc), shape inference    │
└────────────────┬────────────────────────────────────┘
                 │ Boundary #2  (RAII, bounds checking)
                 ▼
┌─────────────────────────────────────────────────────┐
│  System Resources (file system, memory)             │
└─────────────────────────────────────────────────────┘
```

**Boundary #3 (Build → Distribution):** SLSA provenance, code signing, checksum verification.

---

## 2. Secure Design Principles (Saltzer & Schroeder)

| Principle | Application in ONNX Core |
|-----------|--------------------------|
| Economy of Mechanism | Protocol Buffers for serialization; validation centralized in checker.cc; minimal dependencies |
| Fail-Safe Defaults | Validation on by default; must opt out with `check_model=False`; unknown protobuf fields rejected |
| Complete Mediation | Every model load goes through the validation pipeline; all operator inputs are type- and shape-checked |
| Least Privilege | No elevated privileges required; no network access; file I/O restricted to explicitly specified paths |
| Separation of Privilege | External data loading requires both model reference and file system access; releases require SLSA attestation |
| Least Common Mechanism | No global mutable state; validation is stateless; each API call operates independently |
| Psychological Acceptability | Secure defaults need no configuration; clear validation error messages; type-annotated Python API |

---

## 3. Common Weaknesses Mitigated

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

---

## 4. Security Testing

| Method | Details |
|--------|---------|
| Static analysis | CodeQL (GitHub Advanced Security), Clang Static Analyzer, sonarcloud |
| Dynamic analysis | ASan, MSan, UBSan, TSan in CI build matrix |
| Fuzzing | not yet |
| Dependency scanning | Dependabot, OpenSSF Scorecard |

---

## 5. Security Processes

**Vulnerability disclosure**: Reports via GitHub Security Advisories (preferred) or onnx-security@lists.lfaidata.foundation as a fallback; CVE assignment through Linux Foundation CNA. See [SECURITY.md](https://github.com/onnx/onnx/blob/main/SECURITY.md).

**Code review**: All changes require maintainer review; security-sensitive changes require Architecture SIG review; one approval for dependency updates; automated checks must pass before merge ([CODEOWNERS](https://github.com/onnx/onnx/blob/main/.github/CODEOWNERS)).

**Build & distribution**: artifacts signed with Sigstore; PyPI Trusted Publishing with 2FA required for maintainers; SHA256 checksums published; actions pinned to SHA in CI.

---

## 6. Known Limitations

1. **Model encryption**: Not provided; users must handle externally
2. **Formal verification**: Limited formal verification of checker and shape inference logic
3. **Sandboxing**: No built-in sandboxing of model parsing; relies on OS/container isolation

---

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
**Last Updated**: February 2026
**Review Cycle**: Annual (or upon significant architectural changes)
