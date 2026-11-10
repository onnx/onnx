```toml
name = "Hardmax"
domain = "ai.onnx"
shape_inference = "defs::math::utils::HardmaxShapeInference_opset11"

[attributes.axis]
type = "INT"
desc = "Describes the axis of the inputs when coerced to 2D; defaults to one because the 0th axis most likely describes the batch_size. Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(input)."
default = 1
```

The operator computes the hardmax (1 for the first maximum value, and 0 for all others) values for each layer in the batch
 of the given input.

The input does not need to explicitly be a 2D vector; rather, it will be
coerced into one. For an arbitrary n-dimensional tensor
input \in [a_0, a_1, ..., a_{k-1}, a_k, ..., a_{n-1}] and k is
the axis provided, then input will be coerced into a 2-dimensional tensor with
dimensions [a_0 * ... * a_{k-1}, a_k * ... * a_{n-1}]. For the default
case where axis=1, this means the input tensor will be coerced into a 2D tensor
of dimensions [a_0, a_1 * ... * a_{n-1}], where a_0 is often the batch size.
In this situation, we must have a_0 = N and a_1 * ... * a_{n-1} = D.
Each of these dimensions must be matched correctly, or else the operator
will throw errors. The output tensor has the same shape
and contains the hardmax values of the corresponding input.
