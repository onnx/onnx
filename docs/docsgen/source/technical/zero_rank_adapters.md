<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

(onnx-zero-rank-adapters)=

# Proposed zero-rank adapter convention

**Status: proposed exporter/modeling convention.** This pattern uses existing
ONNX operators without changing their semantics. It is not an adopted ONNX
requirement or a guarantee of support by every runtime.

## Motivation

Optional, swappable low-rank adapters, such as LoRA adapters, can share one graph
with the unadapted model. Represent an absent **bias-free** adapter with two
matrices whose shared low-rank dimension has size zero. The same two `MatMul`
nodes then compute the correction for both absent and present adapters. No `If`
node or full-sized zero weight matrix is needed.

Here, "adapter rank" means the chosen factorization dimension `r`, a nonnegative
integer. It is not the ONNX tensor rank (number of dimensions), nor a requirement
that the matrices have mathematical rank exactly `r`. Both adapter weights remain
rank-two tensors when `r = 0`: their shapes are `[input_dim, 0]` and
`[0, output_dim]`. A scalar tensor instead has shape `[]` and does not represent
an absent adapter.

## Shapes and numerical behavior

For an input with at least one dimension, use the following shapes. The leading
dimensions denoted by `...` may be absent, as in a single input vector.

| Value | Construction | Shape |
|-------|--------------|-------|
| `X` | Model input | `[..., input_dim]` |
| `down` | Adapter input | `[input_dim, r]` |
| `up` | Adapter input | `[r, output_dim]` |
| `hidden` | `MatMul(X, down)` | `[..., r]` |
| `correction` | `MatMul(hidden, up)` | `[..., output_dim]` |
| `base` | Base model output | `[..., output_dim]` |
| `Y` | `Add(base, correction)` | `[..., output_dim]` |

At `r = 0`, `hidden` has no elements. Each element of `correction` is a sum
over an empty reduction dimension, with value zero. The correction retains its
full output shape; it is not itself an empty tensor unless an output dimension
is also zero. This also avoids reading or multiplying elements of `X` in the
adapter path, including any NaNs in `X`.

The zero dimensions must be part of the **logical tensor shapes**. The weight
tensors have zero elements, not merely an empty serialized byte payload attached
to nonempty logical weights.

These are the required shape and numerical behaviors for an implementation
supporting this convention, including after optimization or fusion. Adding the
zero correction leaves finite base values numerically unchanged, but is not a
promise of bitwise identity (in particular for signed zero).

### Bias and scaling

This convention is bias-free. An up-projection bias of shape `[output_dim]`
would still be added at `r = 0`; it does not disappear with the weights.
Omit adapter biases, or separately supply zero biases when disabling an adapter.
A bias that belongs to the base model stays on the base path.

Ordinary LoRA scaling `alpha / r` is undefined at zero. Use a finite precomputed
coefficient, or the branch-free normalization `alpha / max(r, 1)` for finite
`alpha`. This agrees with `alpha / r` for every positive integer adapter rank.
If computing it inside the graph, obtain `r` from `Shape(down)` and `Gather`,
apply integer `Max` with one, then `Cast` to the scale's floating-point type
before `Div`. Multiply the correction by the resulting finite coefficient
before `Add`. The coefficient must also be representable and finite in the
chosen computation type: multiplying zero by infinity or NaN does not disable
an adapter.

## Switching adapters at runtime

Expose `down` and `up` as graph inputs and use the same symbolic dimension,
such as `"adapter_rank"`, for `down`'s second dimension and `up`'s first
dimension. Keep `input_dim`, `output_dim`, element types, and the graph topology
consistent. On each invocation, supply both weights with the same concrete `r`;
switching from `r = 2` to `r = 0` changes their shapes, not their tensor ranks.
Update any scaling coefficient and quantization parameters with the weights.

A symbolic dimension permits varying sizes; it does not select an adapter,
validate all cross-input relationships at runtime, or make initializers mutable.
An exporter must preserve the symbolic adapter dimension rather than specialize
it to the example rank. Static weights can use this pattern too, but replacing
initializers is model editing rather than ordinary runtime input switching.

## Executable example

This example uses opset 21 and the ONNX reference evaluator. It constructs one
graph with symbolic adapter rank, checks it, performs strict shape inference,
and runs ranks `0`, `2`, and `3` without changing the graph. The finite coefficient
is computed by the caller. `base` stands for the output of the unchanged base
path; a complete model would connect its base computation here.

The code is executed as part of the documentation build.

```{eval-rst}
.. exec_code::

    from __future__ import annotations

    import numpy as np

    from onnx import TensorProto, checker, helper, shape_inference
    from onnx.reference import ReferenceEvaluator

    def value(name, shape):
        return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)

    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("MatMul", ["X", "down"], ["hidden"]),
                helper.make_node("MatMul", ["hidden", "up"], ["correction"]),
                helper.make_node("Mul", ["correction", "coefficient"], ["scaled"]),
                helper.make_node("Add", ["base", "scaled"], ["Y"]),
            ],
            "zero_rank_adapter",
            [
                value("X", ["batch", 4]),
                value("down", [4, "adapter_rank"]),
                value("up", ["adapter_rank", 3]),
                value("base", ["batch", 3]),
                value("coefficient", []),
            ],
            [value("Y", ["batch", 3]), value("correction", ["batch", 3])],
        ),
        opset_imports=[helper.make_opsetid("", 21)],
        ir_version=10,
    )
    checker.check_model(model, full_check=True)
    model = shape_inference.infer_shapes(model, strict_mode=True)
    evaluator = ReferenceEvaluator(model)

    x = np.arange(8, dtype=np.float32).reshape(2, 4)
    base = np.arange(6, dtype=np.float32).reshape(2, 3)
    alpha = 6.0
    for r in (0, 2, 3):
        feeds = {
            "X": x,
            "down": np.ones((4, r), dtype=np.float32),
            "up": np.ones((r, 3), dtype=np.float32),
            "base": base,
            "coefficient": np.array(alpha / max(r, 1), dtype=np.float32),
        }
        y, correction = evaluator.run(None, feeds)
        expected = np.broadcast_to(x.sum(axis=-1, keepdims=True) * r, (2, 3))
        assert correction.shape == y.shape == (2, 3)
        np.testing.assert_array_equal(correction, expected)
        np.testing.assert_allclose(y, base + expected * feeds["coefficient"])
        if r == 0:
            np.testing.assert_array_equal(correction, np.zeros((2, 3), np.float32))
            feeds["X"] = np.full_like(x, np.nan)
            y, correction = evaluator.run(None, feeds)
            np.testing.assert_array_equal(correction, np.zeros((2, 3), np.float32))
            np.testing.assert_array_equal(y, base)
        print(f"adapter rank {r}: output shape {y.shape}")
```

The Python conditional above only checks the disabled case; the ONNX graph
contains no control flow.

## Blocked quantized weights

The same logical shapes can be produced using released `DequantizeLinear`
operators before the two `MatMul` nodes, for example with `UINT8` weights,
`FLOAT` scales, opset 21, `axis = 0`, and `block_size = 2`. For positive
`input_dim`, `output_dim`, and adapter rank `r`, use these shapes:

| Input to `DequantizeLinear` | Quantized weight | Scale and zero point |
|-----------------------------|------------------|----------------------|
| Down projection | `[input_dim, r]` | `[ceil(input_dim / 2), r]` |
| Up projection | `[r, output_dim]` | `[ceil(r / 2), output_dim]` |

At `r = 0`, with `input_dim = 4` and `output_dim = 3`, this gives:

| Projection | Quantized weight | Scale and zero point | Decoded weight |
|------------|------------------|----------------------|----------------|
| Down | `[4, 0]` | `[2, 0]` | `[4, 0]` |
| Up | `[0, 3]` | `[0, 3]` | `[0, 3]` |

Use the quantized weight's element type for the zero point. It may be omitted
when the operator's default zero point is appropriate. Nonempty scales must be
valid positive, finite quantization scales; empty scales have the shapes above,
not scalar placeholders. In a quantize/dequantize (QDQ) pair, give
`QuantizeLinear` and `DequantizeLinear` matching axes, block sizes, scales, and
zero points. Keep zero dimensions in the quantized weight shapes as well as the
decoded shapes. No packing proposal or new operator is needed.

For dynamic adapter inputs, preserve `"adapter_rank"` in the down scale shape
`[ceil(input_dim / 2), "adapter_rank"]`. Use a separate symbol, such as
`"adapter_blocks"`, for the up scale shape `["adapter_blocks", output_dim]`;
the caller must supply `adapter_blocks = ceil(r / 2)`. ONNX symbolic dimension
names alone cannot express this arithmetic relationship.

## Deployment checks

Validate the exporter and each intended backend or execution provider with both
zero and positive ranks. Check zero-dimension `MatMul`, the full-sized zero
correction, scaling, and shape propagation. For quantized adapters, also check
empty quantized weights and scales, and blocked QDQ support.

Exercise transitions between zero and positive ranks in the same runtime session.
Compilation caches, captured graphs, memory bindings, and adapter-specific
optimizations may specialize shapes or require recapture or recompilation.
Acceptance by the ONNX checker, shape inference, or reference evaluator does
not establish support by an external runtime.

A decoder whose output shape is fixed by a static attribute cannot directly
switch its decoded logical adapter rank between positive and zero on successive
invocations. This includes the proposed `DequantizePacked` with its static
`shape` attribute. This convention does not change that proposal or any other
operator contract.
