<!--
SPDX-FileCopyrightText: Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# ONNX 12-Month Roadmap (Q3 2026 – Q2 2027)

> This is a living document and a starting point — not yet a complete picture of all ONNX SIGs and Working Groups.
> Every group is invited to add or update their milestones below.
> Milestones are **estimations**, not commitments, unless stated otherwise.

---

## How to contribute

1. Find your SIG/WG section below (or add a new one following the template).
2. Fill in your objectives per quarter.
3. Submit a PR or send your input to the roadmap coordinator.

---

## SIGs & Working Groups

---

### SIG Architecture & Infrastructure

**Lead(s):** Andreas Fehlner (TRUMPF Laser), Christian Bourjau (QuantCo)

**Last updated:** 2026-06-02

#### Q3 2026
- Remove test files from release packages (reduce package size and install surface)
- Automate and improve SBOM generation and publication
- Immutable releases (tamper-evident, verifiable artifacts)
- Convert deprecated branch protection rules to Rules within Github
- newer minimal protobuf

#### Q4 2026

- Begin C++ hardening (compiler flags, static analysis integration)
- Integrate fuzz testing into CI

#### Q1 2027
- Move to C++20
- OpenSSF Gold Badge
- Achieve SLSA Build Level 3 compliance

#### Q2 2027
- Improved conformance test suite (parameterized, filterable, usable from non-Python runtimes)
- Continued C++ hardening improvements

---

### Operator SIG

**Lead(s):** G. Ramalingam, Michal Karzynski

**Last updated:** 2026-06-08

Note: The items below may be reprioritized as needed. It may be better to view the following as
an unprioritized list of items for the next year.

#### Q3 2026:
- Attention op (fix causal-mask position-anchoring issue)
- Attention op (add support for local window)
- Attention op (add support for pre-softcap additive bias)
- Support symbolic shape inference

#### Q4 2026:
- Variable length scan support
- LinearAttention (support for Gated DeltaNet-2)

#### Q1 2027:
- Support Mixture-of-Exports (possibly via Grouped MatMul operator)
- RotaryEmbedding for visual models (2D)

#### Q2 2027:
- Support for ternary-value quantization and single-bit quantization
- FlexAttention

### SONNX Working Group

**Lead(s):** Eric JENN, Jean SOUYRIS
**Last updated:** 2026-06-01

#### Q3 2026
- First subset of ONNX operators (i) fully specified in compliance with the SONNX guidelines, (ii) fully formally specified and proved, (iii) implemented, (iv) tested and (v) integrated in the AIDGE framework.
- Final version of the informal specification guidelines.
- First version of the formal specification guidelines.
- First version of the verification guidelines (testing).
- First version of the graph execution semantics formally specified (excluding control flow ops).

#### Q4 2026
- Second subset of ONNX operators (i) fully specified in compliance with the SONNX guidelines, (ii) fully formally specified and proved, (iii) implemented, (iv) tested and (v) integrated in the AIDGE framework.
- Final version of the formal specification guidelines.
- First version of the numerical accuracy analysis guidelines.

#### Q1 2027
- Third subset of ONNX operators (i) fully specified in compliance with the SONNX guidelines, (ii) fully formally specified and proved, (iii) implemented, (iv) tested and (v) integrated in the AIDGE framework.
- Final version of the numerical accuracy analysis guidelines.
- Final version of the verification guidelines (all techniques).
- First subset of operators with numerical accuracy analysis.
- Final version of the graph execution formal specification (including control flow ops).

#### Q2 2027
- Fourth subset of ONNX operators (i) fully specified in compliance with the SONNX guidelines, (ii) fully formally specified and proved, (iii) implemented, (iv) tested and (v) integrated in the AIDGE framework.
- Second subset of operators with numerical accuracy analysis.

---


### [SIG/WG Name]

**Lead(s):** _TBD_
**Last updated:** _YYYY-MM-DD_

#### Q3 2026
- _Objective 1_
- _Objective 2_
- _Objective 3_

#### Q4 2026
- _Objective 1_
- _Objective 2_
- _Objective 3_

#### Q1 2027
- _Objective 1_
- _Objective 2_
- _Objective 3_

#### Q2 2027
- _Objective 1_
- _Objective 2_
- _Objective 3_

---

<!-- Add additional SIG/WG sections by copying the template block above -->
