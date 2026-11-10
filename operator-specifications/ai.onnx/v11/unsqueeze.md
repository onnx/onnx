```toml
name = "Unsqueeze"
domain = "ai.onnx"
shape_inference = "defs::tensor::utils::UnsqueezeShapeInference_opset11"

[attributes.axes]
type = "INTS"
desc = "List of integers indicating the dimensions to be inserted. Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(expanded)."
required = true
```

Insert single-dimensional entries to the shape of an input tensor (`data`).
Takes one required argument `axes` - which contains a list of dimension indices and this operator will insert a dimension of value `1` into the corresponding index of the output tensor (`expanded`).

For example:
  Given an input tensor (`data`) of shape [3, 4, 5], then
  Unsqueeze(data, axes=[0, 4]) outputs a tensor (`expanded`) containing same data as `data` but with shape [1, 3, 4, 5, 1].

The attribute `axes` should not contain any duplicate entries. It is an error if it contains duplicates.
The rank of the output tensor (`output_rank`) is the rank of the input tensor (`data`) plus the number of values in `axes`.
Each value in `axes` should be within the (inclusive) range [-output_rank , output_rank - 1].
The order of values in `axes` does not matter and can come in any order.
