<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# Managing Experimental Operators

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

## Adding Experimental Operators [Deprecated - as of v1.5 experimental ops are no longer supported]

The experimental flag in ONNX operator definitions indicates that a customer of ONNX may not be able to take a long term dependency on that op. Ops in the ONNX namespace (ai.onnx) in the _main_ branch, whether experimental or not, go through the regular review process.

Experimental ops that are being worked on that do not have consensus yet can be managed in one of 2 ways:
1. Use a fork or branch – what you do in the fork or branch is entirely up to you. When you are ready, you can submit a PR using the normal process. This is the recommended way.
2. If a fork/branch is not workable (for example due to complexity of mapping different branches between multiple repos), put the experimental ops in a custom namespace in the main branch.
The specific process for this is:
 * Submit an Issue with a proposal explaining the motivation and plan. It does not need to include detailed technical design. Issues will be tagged as "experimental op".
 * Reviewers will generally approve by default unless the proposal directly conflicts with existing ops or somehow goes against general ONNX strategy. Approval is indicated by adding the "experiment approved" tag.
 * The approval is good for 3 months, but can be renewed if needed.
 * Experimental ops should be submitted in a PR in a custom namespace that is the name of the proposal, i.e. “proposal.controlflow”. The name should be descriptive rather than a company or entity name. These PRs will be approved by default as long as the parent proposal is approved and active.
 * Once experimentation is done, the ops can be submitted for addition to the ONNX namespace via the regular process. The owner can also choose to end the experiment without promoting the ops.
 * Either way, the custom namespace is deleted once experimentation is complete or when the approval expires.
