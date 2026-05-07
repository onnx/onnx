# LinearAttention Expanded Tests — Findings & Fix

## Summary

The 14 `test_linear_attention_*_expanded_cpu` failures had **two root causes**, both
addressed by Phase A of [linear_attention_function_body_fix_plan.md](linear_attention_function_body_fix_plan.md)
plus one minimal correctness fix in `onnx.reference.Scan` that the plan did not
anticipate.

| # | Cause | File | Status |
|---|-------|------|--------|
| 1 | Function body referenced outer-scope names + non-zero `scan_input_axes` | [onnx/defs/nn/defs.cc](../onnx/defs/nn/defs.cc) `LinearAttention` builder | Fixed |
| 2 | `Scan` reference impl mis-sliced loop-state names using `num_scan_inputs` | [onnx/reference/ops/op_scan.py](../onnx/reference/ops/op_scan.py) | Fixed |

## Cause 1 — Function body not self-contained (Phase A)

The C++ builder for `LinearAttention` emitted a `Scan` whose body subgraph (a) consumed
pre-transposed 4D tensors with the time axis at position 2 via
`scan_input_axes=[2,2,2,...]`, and (b) referenced several constants
(`NegOne1D`, `NegTwo1D`, `ScaleFactor`, `StateExpandShape`, `StateReadShape`) by name
from the enclosing function scope.

The ONNX reference `Scan` does not propagate outer scope into the body, and it only
supports `scan_input_axes=0`. Both produced "Unable to find input ..." errors.

### Fix

In `LinearAttention`'s `SetContextDependentFunctionBodyBuilder`:

- **Pre-transpose** Q/K/V/Decay/Beta to `(T, B, H, D)` outside the Scan and drop
  `scan_input_axes` (default 0).
- **Inline** axis constants (`NegOne`, `NegTwo`, `Two`) as `Constant` nodes inside
  the body subgraph.
- **Carry** the dynamic constants (`ScaleFactor`, plus `StateExpandShape` and
  `StateReadShape` when `group_size > 1`) as additional **loop-state** Scan inputs.
  They enter as body inputs and are echoed unchanged via `Identity` as body outputs
  each step.

The body is now fully self-contained — easier for any backend to consume, and no
longer depends on optional features of the host runtime's `Scan`.

## Cause 2 — `Scan` ref impl bug (off-by-attribute slicing)

`onnx/reference/ops/op_scan.py` `_common_run_shape` computed:

```python
state_names_in = self.input_names[: self.num_scan_inputs]
```

This uses `num_scan_inputs` (the count of *scan* inputs, declared by the attribute)
where it should use `num_loop_state_vars` (computed as
`len(args) - num_scan_inputs`). Every other slice in the function uses
`num_loop_state_vars` correctly — only this one was wrong.

When `num_loop_state_vars == num_scan_inputs` it works by coincidence. Pre-Phase-A,
LinearAttention's Scan had 1 loop-state and 3–5 scan inputs; the misalignment
silently mis-named the body input dict, but `dict.update` and `zip(strict=False)`
hid it — and downstream the body ran on the *wrong* tensors. After Phase A, with
the new carried state (`State0`, `ScaleFactor`, optionally
`StateExpandShape`+`StateReadShape`), the misalignment shifted further and produced
the user-visible failure:

```
RuntimeError: Inconsistent permutation [1, 0, 2, 3] with shape (2, 4, 8).
```

(`OutputAccum` ended up rank-3 instead of rank-4, breaking the post-Scan Transpose.)

### Fix

One-line change:

```diff
- state_names_in = self.input_names[: self.num_scan_inputs]
+ state_names_in = self.input_names[:num_loop_state_vars]
```

Pure-Python; no rebuild required.

## Why the fix isn't a "Phase C" change

The original plan deferred all `op_scan.py` work to Phase C (outer-scope propagation
+ non-zero `scan_input_axes`). Phase A made both of those Phase C items moot, but
exposed a *separate*, latent bug: incorrect loop-state slicing. The fix is strictly
a correctness patch — it does not add new `Scan` features.

## Verification

```bash
PYTHONPATH=. pytest onnx/test/test_backend_reference.py -k linear_attention -v
```

Expect 28 passes (14 plain + 14 expanded), 28 CUDA-skipped.

Also worth re-running the broader suite to confirm no regressions in other Scan
consumers:

```bash
PYTHONPATH=. pytest onnx/test/test_backend_reference.py -k scan -v
PYTHONPATH=. pytest onnx/test/reference_evaluator_test.py -k scan -v
```

## Follow-ups

1. Add a targeted unit test for `Scan` with `num_loop_state_vars != num_scan_inputs`
   in [onnx/test/reference_evaluator_test.py](../onnx/test/reference_evaluator_test.py)
   so this slicing bug can't silently regress.
2. Phase C of the original plan (outer-scope propagation + non-zero scan axes) is
   still nice-to-have for spec completeness but is no longer blocking
   `LinearAttention`. Track separately.

## Touched Files

- [onnx/defs/nn/defs.cc](../onnx/defs/nn/defs.cc) — `LinearAttention` function body
  rewritten to be self-contained (Phase A).
- [onnx/reference/ops/op_scan.py](../onnx/reference/ops/op_scan.py) — one-line
  loop-state slicing fix.
