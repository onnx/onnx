# CausalConvWithState Gap Analysis and Opportunities (GAO) Document

## 1. Critique of the Current Testing Plan
The existing test plan in `causal_conv_test_plan.md` is robust, clear, and cleanly aligns with established ONNX project practices. Its strength lies in using `Attention` as a proven template and organizing the work into logical phases (Backend Node Tests, Shape Inference, Version Converter, Parity, Documentation). 

**Strengths:**
- Strong coverage of fundamental parameters (`bias`, `past_state`, kernel sizes).
- Clear mapping of test scenarios to ONNX deliverables (protobuf models, shape validation, upgrade mechanisms).
- Direct focus on `decode_step` (the degenerate `L=1` case) which serves as the most critical production scenario in LLM generation.

**Weaknesses (Gaps):**
- **Empty Tensor / Zero-Length Cases:** No test captures sequence length `L=0` or batch size `B=0`. Empty tensors are notoriously problematic in ONNX model execution and shape inference.
- **Insufficient Negative Shape Inference Testing:** The negative tests only check rank errors and `k=0`. There are missing validations for dimension mismatches (e.g., `past_state` batch size or channels not matching those of `input`).
- **Missing BFloat16 Coverage:** While Float16 is intentionally included, BFloat16 (`bfloat16`) is the prevalent type in modern inference architectures and should be actively tested, especially since the operator handles float-based activations like SiLU.
- **Validating Invalid Attribute States:** Missing tests to restrict the `activation` attribute at the schema level if unexpected strings are provided.

---

## 2. Gaps Identification

| Area | Missing Test / Gap | Risk if Unaddressed |
|---|---|---|
| **Data Types** | BFloat16 (`bfloat16`) coverage. | BFloat16 is standard for modern models. If untested, type-casting omissions in the schema/evaluator could cause downstream execution failures. |
| **Edge Cases (Data)**| Batch size `B=0` or sequence length `L=0`. | Shape inference layers or backend executors might segfault/crash when expecting `k > 0` elements on the sequence axis. |
| **Negative Tests** | Dimension mismatches between `input`, `weight`, and `past_state`. | `past_state` shapes mapped incorrectly might pass initial shape inference but corrupt memory structurally at runtime. |
| **Activations** | Unrecognized string values for `activation`. | Although Python ref evaluator guards this, the C++ shape inference and node check negative cases should also ensure that an arbitrary string (e.g., `"relu"`) fails shape/type verification. |

---

## 3. Opportunities for Extension

**Opportunity 1: Expansive Shape Inference Failure Scenarios**
Extend Phase 2 locally to include a thorough set of `fail_shape_inference` models.
- Supply `past_state` with incompatible dims: `(B+1, C, k-1)`.
- Supply `weight` with shape `(C_out, C_in, k)` where `C_in > 1` (actively forcing a depthwise grouping violation).
- Pass an unrecognized string to the optional `activation` attribute (e.g., `relu` or `tanh`) to verify schema constraints reject it at load-time rather than run-time.

**Opportunity 2: Robust Type Verification for GenAI**
Generate backend models targeting modern AI data types heavily used in causal structures.
- `export_bfloat16` — Validate bf16 capabilities parity.
- (Future-proofing) Write exploratory checks in Python validating if the reference implementation cleanly handles or throws on integer types (`int8`/`uint8`) if quantization routing is introduced gracefully.

**Opportunity 3: Evaluator Defense on `L=0` Tolernace**
While `L=1` acts as the standard single-token autoregressive decoding, consider adding an `export_l_zero` model test. Assess if the ONNX graph spec bypasses convolution returning an empty output `(B, C, 0)` with the latest `k-1` window of `past_state`, or if the graph safely rejects `L=0`.

**Opportunity 4: Extreme Parameter Parity Verification**
Consider designing an `export_large_kernel_small_seq` case (e.g., `k=8, L=2`) coupled with `past_state` omitted. This strictly validates that `pad` allocation logic inside the reference execution layer avoids out-of-boundary access offsets.
