# Plan: Test the CausalConvWithState Operator

## TL;DR

`CausalConvWithState` is already fully wired into the codebase: the C++ schema is
registered for opset 25 in [onnx/defs/nn/defs.cc](onnx/defs/nn/defs.cc) (~line 3702–3900),
the schema is exported in [onnx/defs/operator_sets.h](onnx/defs/operator_sets.h) at lines
1402/1427, and a Python reference `_run` implementation lives at
[onnx/reference/ops/op_causal_conv_with_state.py](onnx/reference/ops/op_causal_conv_with_state.py)
(registered in [onnx/reference/ops/_op_list.py](onnx/reference/ops/_op_list.py)).
**What's missing is the test surface.** This plan adds the four standard ONNX test
artifacts — backend node tests (with generated `.pb` data), shape-inference unit tests,
version-converter (automatic upgrade) tests, and reference-evaluator parity tests — and
regenerates the auto-generated docs. We mirror the `Attention` op as the template since
it was added the same way (PR #6501) and is the closest structural match.

## Steps

### Phase 1 — Backend node test cases (the headline deliverable)

1. Create `onnx/backend/test/case/node/causal_conv_with_state.py` modeled directly on
   [onnx/backend/test/case/node/attention.py](onnx/backend/test/case/node/attention.py).
   Define a `CausalConvWithState(Base)` class with one `@staticmethod export_*()` per scenario:
   - `export_basic` — B=2, C=4, L=8, k=4, no bias, no past_state, activation="none".
   - `export_with_bias` — same shapes, includes bias input.
   - `export_with_past_state` — supplies a non-zero `past_state` of shape (B, C, k-1)
     to verify carry-over (this is the "decode step" scenario).
   - `export_silu` — `activation="silu"` to cover the fused activation branch.
   - `export_swish_alias` — `activation="swish"` (alias of silu) to lock in alias behavior.
   - `export_decode_step` — degenerate L=1 case (autoregressive single-token decode), most
     important production usage.
   - `export_kernel_size_one` — k=1 edge case where `past_state` shape is (B, C, 0).
   - `export_fp16` — float16 dtype to confirm dtype propagation.
   Each method: builds the node via `onnx.helper.make_node("CausalConvWithState", ...)`,
   constructs numpy inputs, calls `CausalConvWithState()._run(...)` from the reference
   module to get expected `(output, present_state)`, and invokes
   `expect(node, inputs=..., outputs=[output, present_state], name="test_causal_conv_with_state_*", opset_imports=[onnx.helper.make_opsetid("", 25)])`.
2. Regenerate the protobuf test data and stat coverage:
   - `python onnx/backend/test/cmd_tools.py generate-data` (this is what populates
     `onnx/backend/test/data/node/test_causal_conv_with_state_*/` with `model.onnx`,
     `input_*.pb`, `output_*.pb`).
   - `python onnx/backend/test/stat_coverage.py` to refresh coverage stats.

### Phase 2 — Shape inference unit tests *(parallel with Phase 1)*

3. Add `test_causal_conv_with_state*` methods in
   [onnx/test/shape_inference_test.py](onnx/test/shape_inference_test.py) following the
   `test_attention_4d` pattern (~line 2430). Cover:
   - Static shapes: assert `output` is `(B, C, L)` and `present_state` is `(B, C, k-1)`.
   - Dynamic `L` (symbolic dim) — output dim 2 must equal input dim 2; state dim 2 must
     remain `k-1` (still derivable from weight).
   - Dynamic `k` — state dim 2 should be unknown/empty.
   - Negative case: rank-2 input → `fail_shape_inference` raised.
   - Negative case: weight with k=0 → `fail_shape_inference` raised.

### Phase 3 — Version converter test *(parallel with Phase 1)*

4. Add `test_CausalConvWithState_1` (and a variant or two) in
   [onnx/test/version_converter/automatic_upgrade_test.py](onnx/test/version_converter/automatic_upgrade_test.py)
   following the `test_Attention_*` pattern (lines 117–203). Since opset 25 is the first
   version, this primarily exercises the no-op upgrade and confirms registration.

### Phase 4 — Reference evaluator parity tests *(depends on Phase 1)*

5. Add a small pytest module (or extend an existing one) under `onnx/test/` that loads
   each generated `model.onnx` from Phase 1 and runs it through
   `onnx.reference.ReferenceEvaluator`, asserting `np.allclose` against the stored
   `output_*.pb`. This is partially redundant with the backend test runner but guards
   against silent regressions in the Python ref impl. **Check first** whether
   `onnx/test/reference_evaluator_test.py` already auto-iterates node test data — if so,
   no new code is needed and this step collapses to "verify the new tests are picked up".

### Phase 5 — Doc regeneration & verification *(depends on Phases 1–3)*

6. Run `python onnx/defs/gen_doc.py` to refresh `docs/Operators.md`, `docs/Changelog.md`,
   and `docs/TestCoverage.md` (CI fails if these are stale — see `CLAUDE.md`).
7. Run lintrunner: `lintrunner -a --output oneline`.

## Relevant files

- [onnx/defs/nn/defs.cc](onnx/defs/nn/defs.cc) — existing C++ schema for opset 25
  (read-only reference: attributes, input/output specs, function body).
- [onnx/reference/ops/op_causal_conv_with_state.py](onnx/reference/ops/op_causal_conv_with_state.py)
  — existing Python reference impl. Import `CausalConvWithState` and call `._run(...)` to
  produce expected outputs in the test-case generator.
- [onnx/backend/test/case/node/attention.py](onnx/backend/test/case/node/attention.py)
  — structural template for the new `causal_conv_with_state.py` file (Base subclass,
  `export_*` static methods, `expect()` invocation, opset_imports).
- [onnx/test/shape_inference_test.py](onnx/test/shape_inference_test.py) — append new
  `test_causal_conv_with_state*` methods here, mirroring `test_attention_4d` (~line 2430).
- [onnx/test/version_converter/automatic_upgrade_test.py](onnx/test/version_converter/automatic_upgrade_test.py)
  — append `test_CausalConvWithState_*` mirroring `test_Attention_1` (line 117).
- `onnx/backend/test/data/node/test_causal_conv_with_state_*/` — generated, do not
  hand-edit. Will be created by step 2.
- [docs/Operators.md](docs/Operators.md), [docs/Changelog.md](docs/Changelog.md),
  [docs/TestCoverage.md](docs/TestCoverage.md) — auto-generated; refreshed by step 6.

## Verification

1. `pytest onnx/test/shape_inference_test.py -k causal_conv_with_state -v` — Phase 2 tests
   green.
2. `pytest onnx/test/version_converter/automatic_upgrade_test.py -k CausalConvWithState -v`
   — Phase 3 test green.
3. `pytest onnx/backend/test -k causal_conv_with_state -v` — backend runner picks up the
   newly generated data directories and the reference evaluator matches stored outputs.
4. `python onnx/defs/gen_doc.py && git diff --stat docs/` — confirms docs were regenerated
   and any diff is intentional (the CausalConvWithState entry should already be present
   from the prior schema landing; only TestCoverage.md should change meaningfully).
5. `lintrunner --output oneline` — clean.
6. Spot-check one generated artifact: load `onnx/backend/test/data/node/test_causal_conv_with_state_with_past_state/model.onnx` with `onnx.checker.check_model` and confirm
   it has opset 25 and the expected I/O signature.

## Decisions

- **Scope is testing only**: do not modify the C++ schema, Python ref impl, or function
  body. If a discrepancy surfaces during testing, raise it as a separate task rather than
  fixing it inline.
- **Use `expect()` + `_run()` pattern (mirrors Attention)** rather than computing expected
  outputs by hand — guarantees the backend tests stay aligned with the published reference
  implementation.
- **Opset 25 in `opset_imports`**: matches the schema registration in
  [onnx/defs/operator_sets.h](onnx/defs/operator_sets.h#L1402). If 25 is not yet the
  default opset on this branch, the tests still work because `make_opsetid` is explicit.
- **Test count target ~6–8 cases** — enough to cover every conditional branch in the
  reference impl (past_state present/absent, activation none/silu, bias present/absent,
  k=1 edge, fp16) without combinatorial explosion.

## Further Considerations

1. **Should we also add a C++ gtest** (`onnx/test/cpp/`)? The `Attention` op did not get
   one, and the shape-inference Python test exercises the C++ inference function via
   pybind. *Recommendation: skip — match the Attention precedent.*
2. **Negative/validation tests** for the C++ schema (rank checks, k≥1) — best added in
   the shape-inference test file (Phase 2). Already included in step 3.
3. **Random seed**: should `causal_conv_with_state.py` set `np.random.seed(...)` for
   reproducible `.pb` generation? *Recommendation: check what `attention.py` does and
   match it (it currently does not seed, relying on test data being checked in once).*
