# Plan: Test the LinearAttention Operator

## TL;DR

`LinearAttention` (opset 25) is fully wired in: schema in
[onnx/defs/nn/defs.cc](onnx/defs/nn/defs.cc#L3937) with a context-dependent function
body (~L4080+), Python reference at
[onnx/reference/ops/op_linear_attention.py](onnx/reference/ops/op_linear_attention.py),
and the design proposal at
[tommasos_personal_files/0009-linear-attention-and-causal-conv.md](tommasos_personal_files/0009-linear-attention-and-causal-conv.md).
**Tests are missing.** This plan adds the same four artifact families as the
CausalConvWithState plan ([tommasos_personal_files/causal_conv_test_plan.md](tommasos_personal_files/causal_conv_test_plan.md))
— backend node tests, shape-inference tests, version-converter tests, and a
function-body parity test — but the matrix is much larger because of three
mostly-orthogonal axes: **update_rule** (4) × **GQA** (MHA / GQA / MQA) × **decoding
phase** (prefill / decode-with-past). We mirror `Attention` (same opset 25 landing
pattern) and `CausalConvWithState` (same recurrent-with-past structure).

## Phases

### Phase 1 — Backend node test cases (headline deliverable)

Create [onnx/backend/test/case/node/linear_attention.py](onnx/backend/test/case/node/linear_attention.py)
modeled on [onnx/backend/test/case/node/attention.py](onnx/backend/test/case/node/attention.py)
and [onnx/backend/test/case/node/causal_conv_with_state.py](onnx/backend/test/case/node/causal_conv_with_state.py)
(once Phase 1 of that plan lands). Define `LinearAttention(Base)` with one
`@staticmethod export_*()` per scenario. **Use the reference impl `_run` to compute
expected outputs**, never hand-rolled math, so backend tests stay locked to the spec.

**Common shape baseline** (shared across most cases): B=2, T=4, H_q=4, H_kv=4,
d_k=8, d_v=8 — small enough for fast `.pb` files, large enough that GQA/MQA reshapes
are non-trivial.

**Update-rule coverage (the primary axis)** — one prefill case per rule, exercising
each branch of the function body builder:
1. `export_linear` — `update_rule="linear"`, no decay, no beta.
2. `export_gated` — `update_rule="gated"`, decay shape `(B,T,H_kv*d_k)`
   (per-key-dim, GLA/RWKV-6 layout).
3. `export_gated_per_head_decay` — `update_rule="gated"`, decay shape `(B,T,H_kv)`
   (per-head scalar, DeltaNet/RetNet layout). Locks in both decay broadcasting
   branches in the reference + function body.
4. `export_delta` — `update_rule="delta"`, beta shape `(B,T,H_kv)`.
5. `export_gated_delta` — default `update_rule="gated_delta"`, decay per-key-dim,
   beta per-head. The Qwen3-Next path; this is the hottest case.
6. `export_gated_delta_beta_scalar` — same, but beta shape `(B,T,1)` to cover the
   broadcastable-1 branch in the reference unpack.

**GQA / MQA coverage** (one variant per rule that supports it would explode the
matrix — pick the "real" production case):
7. `export_gated_delta_gqa` — H_q=8, H_kv=4, group_size=2. Forces the
   `group_size > 1` branch in the function body (Expand / Reshape state read path).
8. `export_gated_delta_mqa` — H_q=8, H_kv=1, group_size=8. Multi-query attention.

**Decoding / past_state coverage** — these are the production-critical autoregressive
cases:
9. `export_decode_step` — `gated_delta`, T=1, with non-zero `past_state`. The
   single-token decode loop iteration. Most-used production scenario.
10. `export_prefill_with_past` — `gated_delta`, T=4, non-zero `past_state` (continuation
    decode after a prefix). Verifies state carry-over across multiple steps.
11. `export_no_past_explicit_zeros` — pass an explicit zero `past_state` and confirm it
    matches omitting input 3 (regression guard for the `ConstantOfShape + CastLike`
    branch).

**Scale & dtype coverage:**
12. `export_explicit_scale` — `scale=0.25` (override), `gated_delta`. Hits the
    `scale != 0.0f` branch in the function body.
13. `export_fp16` — `gated_delta`, dtype float16, GQA on. Confirms `T` propagation
    and exercises the mixed-precision TODO in the reference.
14. `export_bfloat16` — bfloat16 dtype. (May need `np.float16` proxy + manual
    bfloat16 packing — check what `attention.py` does for bfloat16; if it skips,
    skip here too.)

**Edge cases (defer if `expect()` cannot easily express them):**
15. `export_linear_t1_no_past` — degenerate T=1, no past_state. First-token prefill.

**Generation:** for each case, call `LinearAttention()._run(...)` to produce
`(output, present_state)`, then
`expect(node, inputs=[...], outputs=[output, present_state],
name="test_linear_attention_*", opset_imports=[onnx.helper.make_opsetid("", 25)])`.

After authoring, regenerate `.pb` artifacts and stat coverage:
- `python onnx/backend/test/cmd_tools.py generate-data`
- `python onnx/backend/test/stat_coverage.py`

### Phase 2 — Shape inference unit tests *(parallel with Phase 1)*

Append `test_linear_attention_*` methods to
[onnx/test/shape_inference_test.py](onnx/test/shape_inference_test.py) following the
`test_attention_4d` pattern (~L2430). Cover at minimum:
1. `test_linear_attention_basic` — MHA, gated_delta. Output `(B, T, H_q*d_v)`,
   present_state `(B, H_kv, d_k, d_v)`, all dims static.
2. `test_linear_attention_gqa` — H_q != H_kv. Output last dim must derive from
   `H_q * d_v` not `H_kv * d_v`. This is the most likely place for a regression.
3. `test_linear_attention_with_past_state` — past_state provided; verify present_state
   shape equals past_state shape and that B/H_kv/d_k/d_v are propagated even when T or
   batch are symbolic.
4. `test_linear_attention_dynamic_T` — symbolic T dim must flow through to output.
5. `test_linear_attention_per_head_decay` and `test_linear_attention_per_keydim_decay`
   — both decay layouts must shape-infer cleanly (the schema currently relies on the
   function body for inference; verify whichever inference path the registered op
   actually uses).
6. Negative cases via `fail_shape_inference`:
   - `query.ndim != 3`
   - `q_num_heads % kv_num_heads != 0`
   - `gated_delta` without decay/beta inputs
   - `linear` *with* decay or beta inputs (forbidden per ref impl)

### Phase 3 — Version converter tests *(parallel with Phase 1)*

Add `test_LinearAttention_1` … `_N` in
[onnx/test/version_converter/automatic_upgrade_test.py](onnx/test/version_converter/automatic_upgrade_test.py)
mirroring `test_Attention_1`–`_8` (L117–203). Opset 25 is the first version, so this
exercises the no-op upgrade and confirms registration. Add at least one case per
update_rule because the optional-input mask differs per rule — the converter must
preserve that mask.

### Phase 4 — Function body / reference parity test *(depends on Phase 1)*

This is the "extra" complexity that CausalConvWithState did not need (it has no
function body). Add a focused pytest module
`onnx/test/linear_attention_function_body_test.py` (or extend an existing parity test
file — check `onnx/test/function_test.py` first to avoid duplication) that for each
case from Phase 1:
1. Builds the same node via `helper.make_node` and wraps it in a single-node model.
2. Runs it through `onnx.reference.ReferenceEvaluator` *without* function expansion
   (uses the registered `_run`).
3. Re-runs after `onnx.inliner.inline_local_functions` (or equivalent) to expand the
   function body via the registered `SetContextDependentFunctionBodyBuilder`, then
   evaluates again.
4. Asserts both outputs are `np.allclose` (tighter tol for fp32, looser for fp16/bf16).

This is the only way to catch divergence between the C++ function body builder and the
Python `_run` — neither the backend tests (Phase 1) nor shape inference (Phase 2)
exercise the function expansion path.

**Coverage selection for Phase 4:** at minimum one case per (update_rule × {MHA, GQA}
× {with_past, without_past}) — ~16 cases is acceptable; the parity check is fast.

### Phase 5 — Doc regeneration & lint *(depends on Phases 1–3)*

1. `python onnx/defs/gen_doc.py` — refresh `docs/Operators.md`, `docs/Changelog.md`,
   `docs/TestCoverage.md`.
2. `lintrunner -a --output oneline`.

## Relevant files

- [onnx/defs/nn/defs.cc](onnx/defs/nn/defs.cc#L3937) — schema + function body
  (read-only reference; do not modify).
- [onnx/reference/ops/op_linear_attention.py](onnx/reference/ops/op_linear_attention.py)
  — Python reference impl. Source of truth for expected outputs in Phase 1; the
  parity target in Phase 4.
- [onnx/backend/test/case/node/attention.py](onnx/backend/test/case/node/attention.py)
  — structural template for the new `linear_attention.py` (Base subclass, `export_*`
  static methods, GQA/dtype patterns, `expect()` invocation).
- [onnx/backend/test/case/node/causal_conv_with_state.py](onnx/backend/test/case/node/causal_conv_with_state.py)
  *(if landed)* — closest precedent for the recurrent-with-past pattern (`past_state`
  / `present_state` outputs, decode_step case).
- [onnx/test/shape_inference_test.py](onnx/test/shape_inference_test.py#L2430) —
  append after `test_attention_4d`.
- [onnx/test/version_converter/automatic_upgrade_test.py](onnx/test/version_converter/automatic_upgrade_test.py#L117)
  — append after the `test_Attention_*` block.
- [onnx/backend/test/case/node/__init__.py](onnx/backend/test/case/node/__init__.py#L98)
  — `function_expand_helper` is the reference pattern for Phase 4 function expansion
  if `inliner` does not work directly on context-dependent functions.
- `onnx/backend/test/data/node/test_linear_attention_*/` — generated by Phase 1
  step; do not hand-edit.
- [docs/Operators.md](docs/Operators.md), [docs/Changelog.md](docs/Changelog.md),
  [docs/TestCoverage.md](docs/TestCoverage.md) — auto-regenerated by Phase 5.
- [tommasos_personal_files/causal_conv_test_plan.md](tommasos_personal_files/causal_conv_test_plan.md)
  — sibling plan; same overall artifact families and verification structure.
- [tommasos_personal_files/linear_attn_function_body_plan.md](tommasos_personal_files/linear_attn_function_body_plan.md)
  — context on the function body construction this plan tests.

## Verification

1. `pytest onnx/test/shape_inference_test.py -k linear_attention -v` — Phase 2 green.
2. `pytest onnx/test/version_converter/automatic_upgrade_test.py -k LinearAttention -v`
   — Phase 3 green.
3. `pytest onnx/backend/test -k linear_attention -v` — Phase 1 backend runner picks up
   generated data and matches the reference.
4. `pytest onnx/test/linear_attention_function_body_test.py -v` — Phase 4 parity.
5. `python onnx/defs/gen_doc.py && git diff --stat docs/` — regenerated, diff sane.
6. `lintrunner --output oneline` — clean.
7. Spot-check generated artifacts: load
   `onnx/backend/test/data/node/test_linear_attention_decode_step/model.onnx` with
   `onnx.checker.check_model`, confirm opset 25, two outputs, six potential inputs.
8. Sanity: re-run Phase 4 parity suite under `ONNX_ML=0` build to catch any
   ML-only-flag drift in the function body.

## Decisions

- **Scope is testing only.** Do not modify the C++ schema, the function body builder,
  or the Python reference. If a parity divergence surfaces in Phase 4, file it as a
  separate task with the failing case attached.
- **Reference impl is the oracle**, not hand-derived expected values. This locks the
  backend tests to the published `_run` and avoids re-deriving the four recurrence
  formulas in test code.
- **Matrix bound: ~14–18 backend cases** (down from a theoretical 4 rules × 3 GQA × 2
  decay layouts × 2 past × 2 dtype = 96). Justification: each chosen case isolates one
  branch in either the function body or the reference unpack; combinations are covered
  by the parity test (Phase 4) at much lower per-case cost.
- **Phase 4 parity test is non-optional.** The function body is the most fragile new
  surface (4 build-time rule branches × GQA branches × past-state branch × Scan body
  graph construction). Without it, drift between C++ expansion and Python `_run` is
  invisible to CI.
- **Defer bfloat16** if the existing `attention.py` skips it. Backend `.pb` storage
  for bf16 has historically been awkward.
- **Opset 25 explicit** in every `make_opsetid` call, matching the schema registration
  even if 25 becomes the default opset on this branch.

## Further Considerations

1. **Random seed for deterministic `.pb` files** — `attention.py` does not seed and
   relies on artifacts being checked in once. Recommendation: match that — let the
   one-time `generate-data` run produce stable bytes; do not seed inside `export_*`.
2. **Should Phase 4 also assert symbolic execution under `Scan`?** The function body
   uses `Scan` over T. The reference evaluator must support `Scan` execution — verify
   on a quick smoke test before authoring all 16 cases. If `Scan` support is partial
   for opset 25 in the ref evaluator, Phase 4 may need to fall back to inlining the
   Scan body manually for T=1 cases only.
3. **GQA in shape inference** — the schema's inference function may or may not encode
   the GQA `H_q*d_v` vs `H_kv*d_v` distinction. Phase 2 will surface this; if missing,
   raise as a separate issue rather than fixing in this test PR.
4. **Negative-case enforcement** — the C++ schema's static checks may be looser than
   the Python reference's runtime checks. Phase 2 negative cases should target only
   what the registered shape-inference function actually validates. Audit before
   writing — do not over-spec the schema accidentally.
