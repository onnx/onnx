```toml
name = "ReduceMax"
domain = "ai.onnx"
shape_inference = "defs::reduction::utils::ReduceShapeInference_opset12"

[type_constraints.T]
types = [
    "tensor(uint32)",
    "tensor(uint64)",
    "tensor(int32)",
    "tensor(int64)",
    "tensor(float16)",
    "tensor(float)",
    "tensor(double)",
    "tensor(uint8)",
    "tensor(int8)",
]
desc = "Constrain input and output types to high-precision and 8 bit numeric tensors."
```

Computes the max of the input tensor's element along the provided axes. The resulting
tensor has the same rank as the input if keepdims equals 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.

The above behavior is similar to numpy, with the exception that numpy defaults keepdims to
False instead of True.
