# Plan: CausalConvWithState SiLU precision fix

Align the `CausalConvWithState` function body's SiLU/Swish path with the Python
reference implementation, which upcasts to float32 around `Sigmoid`+`Mul` for
fp16/bfloat16 inputs. Today the function body computes both in the native
dtype, while `onnx/reference/ops/op_causal_conv_with_state.py` does the
activation in float32 and casts the result back — so the expanded form does not
match the reference for low-precision dtypes.

## Steps

1. Patch the SiLU branch of the function body in `onnx/defs/nn/defs.cc`
   (inside `SetContextDependentFunctionBodyBuilder` for `CausalConvWithState`,
   the `if (activation == "silu" || activation == "swish")` block, ~L3905).
   Replace the existing two-line `Sigmoid`/`Mul` ONNX text with:
   - `ConvOutFloat = Cast <to = 1> (ConvOut)`   (1 = FLOAT)
   - `ConvSigmoid  = Sigmoid (ConvOutFloat)`
   - `MulOutFloat  = Mul (ConvOutFloat, ConvSigmoid)`
   - `output       = CastLike (MulOutFloat, ConvOut)`

   For float32 inputs this becomes a no-op cast pair that runtimes fold;
   for fp16/bf16 it mirrors the reference behavior.

2. Add an fp16 + silu node test in
   `onnx/backend/test/case/node/causal_conv_with_state.py` named
   `test_causal_conv_with_state_silu_fp16` (B=2, C=4, L=8, k=4 like
   `export_fp16`, with `activation="silu"`). Expected outputs come from the
   existing `_compute` helper.

3. Regenerate auto-generated artifacts:
   - `python onnx/defs/gen_doc.py` → updates `docs/Operators.md`,
     `docs/Changelog.md`, `docs/TestCoverage.md`.
   - Regenerate backend test data so
     `onnx/backend/test/data/node/test_causal_conv_with_state_silu_fp16/`
     is emitted.

4. Run lintrunner and the targeted tests (see Verification).

## Relevant files

- `onnx/defs/nn/defs.cc` (~L3905-L3911) — SiLU branch of the
  `CausalConvWithState` function-body builder. Only schema change.
- `onnx/reference/ops/op_causal_conv_with_state.py` (~L60-L66) — already
  upcasts; cited as the target behavior. No change.
- `onnx/backend/test/case/node/causal_conv_with_state.py` — add
  `export_silu_fp16` alongside `export_fp16` / `export_silu`.
- Regenerated: `docs/Operators.md`, `docs/Changelog.md`,
  `docs/TestCoverage.md`, plus the new test_data folder.

## Verification

1. `lintrunner -a --output oneline` — clean.
2. Rebuild (C++ change required): `ONNX_ML=1 python -m pip install -e . -v`.
3. Targeted tests:
   - `pytest onnx/test/schema_test.py -k CausalConvWithState`
   - `pytest onnx/test/reference_evaluator_test.py -k causal_conv_with_state`
   - `pytest onnx/test/test_backend_test.py -k causal_conv_with_state`
4. Manually inline the function via `onnx.inliner.inline_local_functions` for
   an fp16+silu model and compare the reference evaluator's output against
   running the original op. They should now agree within fp16 tolerance
   instead of diverging. Repeat with bfloat16.
5. `git status` shows regenerated docs and new test data; no unrelated
   generated file is dirty.

## Decisions

- Direction: align the function body to the reference (Cast → Sigmoid → Mul →
  CastLike), per the chosen option.
- Scope limited to `CausalConvWithState`. The standalone `Swish-24` op's body
  has the same pattern but changing it touches a different released schema —
  out of scope.
- Cast target is `to = 1` (FLOAT), matching the reference exactly and lossless
  for all constrained input types (float / float16 / bfloat16).

## Further considerations

1. **Cast precision.** Float32 (recommended, matches reference) vs. a
   dtype-aware conditional upcast. Recommendation: unconditional
   `Cast` + `CastLike` for simplicity; runtimes fold the no-op pair for fp32.
2. **Standalone `Swish-24` body.** Has the same gap. Recommendation: file a
   follow-up issue, do not bundle here — it widens the PR scope and touches a
   different released op schema.