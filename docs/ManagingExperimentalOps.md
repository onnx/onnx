<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# Managing Experimental and Preview Operators

## Deprecated Experimental Operators

The following experimental operators were deprecated and removed from ONNX. They should be removed from models, either substituted with newer superseding operators or decomposed into functionally equivalent operators:

Old operator        |New Operator
--------------------|--------------------------
`ATen`              |NA
`Affine`            |`Add(Mul(X, alpha), beta)`
`ConstantFill`      |`ConstantOfShape`
`Crop`              |`Slice-1`
`DynamicSlice`      |`Slice-10`
`GRUUnit`           |NA
`GivenTensorFill`   |`Const` or `ConstantOfShape`
`ImageScaler`       |`Add(Mul(X, scale), Unsqueeze(bias, axes=[0, 2, 3]))`
`ParametricSoftplus`|`Mul(alpha, Softplus(Mul(beta, X)))`
`Scale`             |`Mul(X, scale)`
`ScaledTanh`        |`Mul(Tanh(Mul(X, beta)), alpha)`
`Upsample-1`        |`Resize-10`

## Adding Experimental Operators [Deprecated - as of v1.5 experimental ops are no longer supported]

The old mechanism of marking standard operators as experimental using the experimental flag (via `SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)`) has been deprecated and is no longer supported for new operators. 

For historical operators (like `Upsample-1`), this flag has been kept or the operators have been documented as deprecated/removed. For all new experimental features, contributors should follow the **Preview Operator Domains** path described below.

## Preview Operator Domains (`ai.onnx.preview` & `ai.onnx.preview.training`)

To enable experimentation, research, and community feedback without cluttering the stable standard namespace, ONNX provides two preview domains:
- `ai.onnx.preview` (for standard operators under development)
- `ai.onnx.preview.training` (for training-specific operators under development)

These domains function as the successor to the deprecated experimental flag mechanism.

### Key Rules and Characteristics of Preview Domains

1. **No Versioning Guarantees**: Preview operators are registered with a fixed version (version 1) within their respective preview domains (e.g., `ai.onnx.preview` version 1). 
2. **In-place Modifications**: If changes are needed for a specific preview operator during the experimentation phase, its specification and schema should be modified directly in the repository without increasing its opset version.
3. **No Stability or Compatibility Guarantees**: Operators in the preview domains can have breaking changes introduced, or can even be completely removed at any time, based on feedback and review.
4. **Graduation to Standard Namespace**: Once a preview operator is mature and has consensus, it can graduate to the standard namespace (`ai.onnx` or `ai.onnx.training`). At that point, a new version of the operator is defined under the standard namespace, and the old version under the preview domain is eventually removed.

For the step-by-step process on proposing and implementing preview operators, please see [AddNewOp.md](AddNewOp.md#preview-domain-path).
