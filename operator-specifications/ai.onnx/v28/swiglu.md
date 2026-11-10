```toml
name = "SwiGLU"
domain = "ai.onnx"
node_determinism = "deterministic"
shape_inference = "defs::math::utils::SwiGLUShapeInference"

[attributes.alpha]
type = "FLOAT"
desc = "Coefficient that scales the gate input inside the sigmoid of the Swish activation. The default value is 1.0."
default = 1.0

[[inputs]]
name = "A"
type = "T"
desc = "Gate input tensor"
differentiation = "differentiable"

[[inputs]]
name = "B"
type = "T"
desc = "Linear (value) input tensor"
differentiation = "differentiable"

[[outputs]]
name = "Y"
type = "T"
desc = "Output tensor"
differentiation = "differentiable"

[type_constraints.T]
types = "all_float_types_ir4"
desc = "Constrain input and output types to float tensors."

[function_bodies]
28 = """
{
   SwishGate = Swish <alpha: float = @alpha> (A)
   Y = Mul (SwishGate, B)
}
"""
```

SwiGLU is a gated activation that takes two inputs, a gate `A` and a linear (value)
input `B`, and produces one output `Y`. It applies the Swish activation to the gate
and multiplies the result elementwise by the linear input:

```
Y = Swish_alpha(A) * B
```

The gate activation `Swish_alpha` is exactly the `Swish` operator with the same
`alpha`, i.e. `Swish_alpha(a) = a * Sigmoid(alpha * a)`. Inputs `A` and `B` must
have identical shapes; broadcasting is not applied and the output `Y` has the same
shape as the inputs.

Exporters typically produce `A` and `B` in one of two ways: for the common
two-projection form (e.g. Llama's `gate_proj`/`up_proj`) wire the two projection
outputs directly to `A` (gate) and `B` (value); for a fused/packed single
projection, split it upstream into `A` and `B` with `Split` (contiguous layout)
or `Slice`/`Gather` (interleaved layout).
