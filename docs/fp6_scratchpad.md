<!-- SPDX-License-Identifier: Apache-2.0 -->

## FP6 (FLOAT6E2M3, FLOAT6E3M2) enablement plan

Motivation: add FP6 formats from the OCP Microscaling (MX) v1.0 spec to ONNX to enable AMD MIGraphX/ROCm workflows and wider HW enablement.

Clarifications requested
- Rounding: confirm RNE (round-to-nearest, ties-to-even) for downcasts, with explicit guard/round/sticky (GRS) behavior. We currently implement ties-to-even via integer rounding, and will refine to explicit GRS if spec requires (OK?).
- Encodings: confirm exponent biases (E2M3 bias=3, E3M2 bias=4), no NaN/Inf encodings, and handling of −0 (we map −0 to +0 per discussion; confirm).
- Packing: confirm little-endian bit order, 4 FP6 values packed in 24 bits → 3 bytes; last group padded with zeros if not multiple of 4.
- Initial operator set: Cast/CastLike, QuantizeLinear/DequantizeLinear. Any others desired for the initial IR version?

Scope of changes
- IR/protobuf: add enum values in `TensorProto.DataType` and bump IR version.
- Parser/checker/C++ IR bridge: accept new types in textual parser, model checker, and `ir_pb_converter` encode/decode lists.
- Python dtype mapping: map to `ml_dtypes.float6_*` when available; otherwise store as `uint8` and decode to `float32` in numpy_helper.
- Runtime reference ops: add float32<->fp6 conversion helpers used by Cast/QDQ.
- Packing/unpacking: pack/unpack 6-bit values for raw-data serialization.
- Docs: IR types table + technical note.
- Tests: pack/unpack round-trip; Cast / QDQ behavior incl. subnormals, ±0, NaN/Inf saturation.

Work items (tracked)
1) Protobuf/IR
   - Add `FLOAT6E2M3`, `FLOAT6E3M2`; bump IR to 13; note int32_data semantics for 6-bit.
2) C++
   - `data_type_utils.{h,cc}`: validity + string maps.
   - `parser.{h,cc}`: primitive type names + tensor parsing via int32_data path.
   - `checker.cc`: validate storage field selection for fp6.
   - `common/ir_pb_converter.cc`: add new cases for int32_data path.
3) Python
   - `_mapping.py`: dtype maps → `ml_dtypes.float6_*` or `uint8` fallback.
   - `numpy_helper.py`: `_pack_6bit`/`_unpack_6bit`; `to_array` returns float32 when `ml_dtypes` missing.
   - `helper.py`: `make_tensor` handles fp6: float32→fp6 via conversion, then pack 6-bit into `raw_data`.
   - Reference ops: `float32_to_float6*`, `float6*_to_float32` and wire into Cast/QDQ.
4) Docs
   - IR types table; `docsgen/technical/float6.md` with encoding/packing notes.
5) Tests
   - Round-trip pack/unpack; Cast; Quantize/Dequantize; subnormals; ±0; NaN/Inf saturation.

Rounding/subnormal policy (current)
- RNE ties-to-even using integer rounding; carry-propagation to exponent; subnormals generated when biased-exp underflows; −0 coerced to +0.
- Out-of-range/NaN/Inf saturate to max finite (sign-preserving for negatives).

CI/CD readiness
- Protobuf regenerated via `python -m onnx.gen_proto`.
- Lint/build: no new warnings in Python; C++ includes extended case lists only. Ensure no duplicate switch labels.
- Tests added for fp6 behavior; numpy_helper fallbacks validated.

Follow-ups
- If OCP spec mandates explicit GRS, refine converters to compute guard/round/sticky bits and implement exact RNE.
- Consider adding `ml_dtypes` minimum version pin once fp6 dtypes are released.


