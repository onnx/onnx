# Plan: Fix LinearAttention Function Body Expansion

## TL;DR

The `*_expanded_cpu` backend test failures (14 of them) have **two independent root
causes in the C++ function body**, plus one optional reference-evaluator improvement.
The function body emits a `Scan` whose body subgraph (a) consumes pre-transposed 4D
tensors with the time axis at position 2 and (b) references several outer-scope
constants by name. The reference evaluator's `Scan` op rejects both: it only supports
`scan_input_axes=0` and does not propagate outer scope into the body.

The cleanest fix lives entirely in [onnx/defs/nn/defs.cc](onnx/defs/nn/defs.cc) — make
the body self-contained and pre-transpose so T is the leading axis. As a secondary
follow-up, fix `Scan` in `onnx.reference` to (1) propagate outer-scope context like
`Loop` already does and (2) support non-zero `scan_input_axes` — both are
spec-compliant features the ref evaluator is missing.

## Discovered Facts

1. **Outer-scope capture not propagated to Scan body.**
   [onnx/reference/ops/op_scan.py](onnx/reference/ops/op_scan.py) (`_run`, ~L99–L154)
   builds `inputs` from `state_names_in` + `scan_names_in` only — no `context`
   parameter, no `need_context()` override. Compare to
   [onnx/reference/ops/op_loop.py](onnx/reference/ops/op_loop.py) (~L22–L51) which does
   `need_context() = True` and merges `context` into `inputs`. **Spec says subgraphs
   may capture outer-scope names**, so this is a ref-evaluator gap, not a function
   body bug per se.
2. **`scan_input_axes != 0` not implemented.**
   [onnx/reference/ops/op_scan.py](onnx/reference/ops/op_scan.py) (~L40–L42) raises if
   any axis is non-zero. Spec allows arbitrary axes. Same restriction applies to
   `scan_output_axes` further down.
3. **The function body relies on both.**
   [onnx/defs/nn/defs.cc](onnx/defs/nn/defs.cc) (~L4310–L4356) sets
   `scan_input_axes = [2,2,2,...]` and the body references `NegOne1D`, `NegTwo1D`,
   `ScaleFactor`, `StateExpandShape`, `StateReadShape` — all defined in the enclosing
   function body, not in the Scan body subgraph.
4. **Confirmation from the generated artifact**: in
   `onnx/backend/test/data/node/test_linear_attention_prefill_with_past_expanded/model.onnx`,
   the Q4D Transpose perm is `[0, 2, 1, 3]` (output `(B, H, T, D)`) and the Scan node
   has `scan_input_axes = [2, 2, 2, 2, 2]`, `scan_output_axes = [2]`, plus body inputs
   that reference outer-scope `..._function_NegOne1D` etc.

## Phases

### Phase A — Fix the function body to be self-contained (primary fix)

All edits in [onnx/defs/nn/defs.cc](onnx/defs/nn/defs.cc) inside the
`LinearAttention` `SetContextDependentFunctionBodyBuilder` (~L4083–L4360).
**Do not touch the registered Python `_run`** in
[onnx/reference/ops/op_linear_attention.py](onnx/reference/ops/op_linear_attention.py)
— it stays the oracle.

1. **Remove outer-scope name capture.** Move every constant the body references into
   the body itself as either:
   - `Constant` nodes inside the body subgraph (for `NegOne1D`, `NegTwo1D`, GQA
     `Two1D`), or
   - additional **carried state inputs** of the Scan (`ScaleFactor`,
     `StateExpandShape`, `StateReadShape`) that are passed in once and emerge
     unchanged at each step.
   - Recommended split:
     - Inline `Constant`s: `NegOne1D`, `NegTwo1D`, `Two1D` (tiny, no cost).
     - Carried state: `ScaleFactor` (value depends on attribute / dynamic d_k),
       and the GQA shape tensors when `group_size > 1`.

2. **Make the body operate with T as the leading axis.** Pre-transpose
   Q/K/V/Decay/Beta to `(T, B, H, D)` (perm `[1, 0, 2, 3]`) outside Scan, leave
   `scan_input_axes` unset (default 0), and post-transpose the accumulated output
   `(T, B, H_q, d_v)` → `(B, T, H_q, d_v)` → reshape to 3D. The current code already
   *comments* that this was the intent — the actual emitted perm is `[0, 2, 1, 3]`
   per the generated model — fix the emitted perm and drop the `scan_input_axes`
   attribute.

3. **Update the `read_block` GQA path** to consume `StateExpandShape` /
   `StateReadShape` from the body's new state inputs rather than outer scope.

4. **Re-emit Scan with default axes** (`scan_input_axes` and `scan_output_axes`
   omitted; default 0). Update the carried-state count in `num_scan_inputs`
   accounting (add 1 for `ScaleFactor`, +2 if GQA).

### Phase B — Regenerate and re-verify (depends on A)

1. Rebuild C++:
   `pip install -e . --no-build-isolation -v` (or copy the updated `.so` from
   `.setuptools-cmake-build/`).
2. `PYTHONPATH=. python onnx/backend/test/cmd_tools.py generate-data --clean`.
3. `PYTHONPATH=. pytest onnx/test/test_backend_reference.py -k linear_attention -v`
   — **expect all 28 (14 plain + 14 expanded) passing**.

### Phase C — Optional: tighten `onnx.reference.Scan` to spec (separate, decoupled task)

Even after Phase A makes the body self-contained, the ref evaluator's `Scan` is still
spec-incomplete. Two small fixes in
[onnx/reference/ops/op_scan.py](onnx/reference/ops/op_scan.py):

1. **Outer-scope propagation**: add `def need_context(self) -> bool: return True` and
   merge `context` into the body inputs dict in `_run`, mirroring
   [onnx/reference/ops/op_loop.py](onnx/reference/ops/op_loop.py) (~L48–L51). Update
   `_run`'s signature to accept `context=None`.
2. **Non-zero `scan_input_axes`**: replace the `RuntimeError` at
   [op_scan.py L42](onnx/reference/ops/op_scan.py#L42) with
   `np.moveaxis(value, axis, 0)` on each scan input before iterating. Same for
   `scan_output_axes` (analogous restriction further down).

Phase C should be its own PR (separate review surface; touches the reference evaluator
core).

## Decisions

- **Phase A is the canonical fix** even if Phase C also lands. Self-contained
  subgraphs are easier for *any* runtime to consume; relying on outer-scope capture
  is fragile across backends.
- **Carried state vs. inline Constant**: scalars/axes inline; shape tensors that
  depend on `Shape(query)` should be carried (recomputing `Shape` inside the body
  each iteration is wasteful and re-introduces dynamic-shape coupling).
- **`scale` lives as carried state**, not as a `Constant` inside the body, because
  the non-attribute branch derives it from `Shape(query)` at graph build time and
  we'd otherwise need to re-derive every iteration.
- **Test scope unchanged**: no new test files needed for Phase A — the existing 14
  expanded tests are exactly the regression gate. Phase C, if pursued, gets a
  dedicated unit test in
  [onnx/test/reference_evaluator_test.py](onnx/test/reference_evaluator_test.py) for
  `Scan` outer-scope capture and non-zero axes.

## Verification

1. After Phase A rebuild:
   `ls onnx/backend/test/data/node | grep linear_attention | wc -l` → 14, then
   inspect one model's Scan attrs (no `scan_input_axes`) and the body's `node.input`
   list (no outer-scope names like `..._function_NegOne1D`).
2. `PYTHONPATH=. pytest onnx/test/test_backend_reference.py -k linear_attention -v`
   → 28 passing (14 plain + 14 expanded), 28 CUDA-skipped.
3. `PYTHONPATH=. pytest onnx/test/test_backend_reference.py -v 2>&1 | tail` —
   confirm no other op (Attention, etc.) regressed from neighbour edits in
   `defs.cc`.
4. (If Phase C done) `PYTHONPATH=. pytest onnx/test/reference_evaluator_test.py -k scan -v`.

## Relevant Files

- [onnx/defs/nn/defs.cc](onnx/defs/nn/defs.cc) (~L4083–L4360) — the
  `LinearAttention` `SetContextDependentFunctionBodyBuilder`. **Phase A target.**
- [onnx/reference/ops/op_scan.py](onnx/reference/ops/op_scan.py) — `Scan` reference
  impl. **Phase C target.**
- [onnx/reference/ops/op_loop.py](onnx/reference/ops/op_loop.py) — pattern for
  `need_context()` + outer-scope propagation Phase C should mirror.
- [onnx/reference/ops/op_linear_attention.py](onnx/reference/ops/op_linear_attention.py)
  — Python reference; **do not modify** (oracle).
- [onnx/backend/test/case/node/linear_attention.py](onnx/backend/test/case/node/linear_attention.py)
  — backend test cases (already landed); regression gate for Phase A.
- [tommasos_personal_files/linear_attention_test_plan.md](tommasos_personal_files/linear_attention_test_plan.md)
  — sibling plan that scoped Phase 1 (these tests) and explicitly deferred function-body
  parity bugs to a separate task; this plan is that task.
- [tommasos_personal_files/linear_attn_function_body_plan.md](tommasos_personal_files/linear_attn_function_body_plan.md)
  — the original function-body construction plan; useful context for the Scan body
  layout.

## Further Considerations

1. **Could we just do Phase C and skip Phase A?** No — the function body would still
   embed `scan_input_axes=[2,...]` and outer-scope names, which are valid ONNX but
   harder for downstream runtimes to consume. Phase A produces a cleaner spec
   artifact; Phase C just makes the ref evaluator more complete.
2. **Performance of carried-state shape tensors**: Scan re-passes carried state every
   iteration in any optimizing backend; for small 1-D shape tensors this is free.
   Not a concern.
3. **Attribute parity**: this is *exactly* the kind of divergence Phase 4 of the test
   plan ([linear_attention_test_plan.md](tommasos_personal_files/linear_attention_test_plan.md))
   was designed to catch. Once Phase A lands, Phase 4's parity test becomes much
   smaller (the existing `_expanded` backend tests already provide most of the
   coverage).
4. **Why does Attention's function body work?** Attention does not use Scan — it's a
   straight-line decomposition. So neither outer-scope capture nor non-zero scan axes
   come up. LinearAttention is the first ONNX core op to use Scan inside a function
   body builder, which is why these latent ref-evaluator gaps are surfacing now.
