```toml
name = "Split"
domain = "ai.onnx"

[attributes.axis]
type = "INT"
desc = "Which axis to split on. A negative value means counting dimensions from the back. Accepted range is [-rank, rank-1] where r = rank(input)."
default = 0

[attributes.num_outputs]
type = "INT"
desc = "Number of outputs to split parts of the tensor into. If the tensor is not evenly splittable the last chunk will be smaller."
```

Split a tensor into a list of tensors, along the specified 'axis'.
Either input 'split' or the attribute 'num_outputs' should be specified, but not both.
If the attribute 'num_outputs' is specified, then the tensor is split into equal sized parts.
If the tensor is not evenly splittable into `num_outputs`, the last chunk will be smaller.
If the input 'split' is specified, it indicates the sizes of each output in the split.
