<!--
SPDX-FileCopyrightText: Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# ONNX 12-Month Roadmap (Q4 2026 – Q3 2027)

> This is a living document and a starting point — not yet a complete picture of all ONNX SIGs and Working Groups.
> Every group is invited to add or update their milestones below.
> Milestones are **estimations**, not commitments, unless stated otherwise.

---

## How to contribute

1. Find your SIG/WG section below (or add a new one following the template).
2. Fill in your objectives per quarter.
3. Submit a PR or send your input to the roadmap coordinator.

---

## Special Interest Groups & Working Groups

---

### SIG Architecture & Infrastructure

**Lead(s):** Andreas Fehlner (TRUMPF Laser), Christian Bourjau (QuantCo)

**Last updated:** 2026-10-29

#### Q4 2026
- Automate and improve SBOM generation and publication
- Begin C++ hardening (compiler flags, static analysis integration)
- Integrate fuzz testing into CI
- Define a stable C-API
- Immutable releases (tamper-evident, verifiable artifacts)

#### Q1 2027
- Clarify need of test file corpus
- Move to C++20
- Achieve SLSA Build Level 3 compliance

#### Q2 2027
- OpenSSF Gold Badge
- Improved conformance test suite (parameterized, filterable, usable from non-Python runtimes)
- Continued C++ hardening improvements

#### Q3 2027
-

---

### Operator SIG

**Lead(s):** G. Ramalingam, Michal Karzynski

**Last updated:** 2026-06-08

Note: The items below may be reprioritized as needed. It may be better to view the following as
an unprioritized list of items for the next year.

#### Q3 2026:
- ~~Attention op (fix causal-mask position-anchoring issue)~~ (landing in [#8068](https://github.com/onnx/onnx/pull/8068))
- Attention op (add support for local window)
- Attention op (add support for pre-softcap additive bias)
- Support symbolic shape inference

#### Q4 2026:
- Variable length scan support
- LinearAttention (support for Gated DeltaNet-2)
- Support quantized formats defined by the GGUF format
- Better MoE model support

#### Q1 2027:
- Support Mixture-of-Experts (possibly via Grouped MatMul operator)
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


### Optimizations SIG

**Lead(s):** Freddy Chiu, Justin Chu

**Last updated:** 2026-11-13

Note: All milestones are tentative. SIG projects: [onnx/optimizer](https://github.com/onnx/optimizer), [onnx-shape-inference](https://github.com/onnx/onnx-shape-inference), [neural-compressor](https://github.com/onnx/neural-compressor).

#### Q4 2026
- Migration of the onnx-shape-inference project as a SIG-owned project
- onnx/optimizer: new release (last: v0.4.2, Jan 2026) with new fusion passes, a C++ IR entry point, ONNX function preservation, wheels for free-threaded Python and ARM, and many correctness fixes
- onnx/optimizer: resolve the reported output-mismatch bugs found by differential testing

#### Q1 2027
- Fusion passes for LLM patterns (e.g. GELU, attention, rotary embeddings) that target the native `Attention` and `RotaryEmbedding` operators; coordinate with the ONNX Script rewriter to avoid duplicate work
- Quantization: survey of ONNX-level quantization tools (ONNX Runtime quantization, AMD Quark, neural-compressor, vendor SDK forks) and a feature comparison for SDK users; decide the future role of neural-compressor
- Differential-testing regression suite for optimization passes, built from the reported cases and run on a schedule

#### Q2 2027
- Quantization: shared QDQ conventions and reference recipes, so that vendor tools produce comparable results; align with the Operator SIG's quantized-format work (GGUF formats, low-bit quantization)
- Lowered representation for kernel generators such as tinygrad: design note on an explicit form (e.g. no implicit broadcasting), using a separate domain only if needed (with the Operator SIG)
- Clarify the roles of onnx/optimizer, onnx-shape-inference, onnxsim, and the ONNX Script optimizer (which passes and analyses live where)
- Extend differential testing to the ONNX reference implementation, using the improved conformance test suite (Architecture & Infrastructure SIG)

#### Q3 2027
- onnx/optimizer: stable C++ API based on the IR entry point, so other tools can embed the passes
- onnx/optimizer: pass that makes broadcasting explicit, as a first step toward the lowered representation
- Quantized reference models in the shared test corpus
- Evaluate a remote model executor (onnx-remote) for measuring and tuning models on devices without a native converter (e.g. Snapdragon phones)

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
