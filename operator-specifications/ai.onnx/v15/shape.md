```toml
name = "Shape"
domain = "ai.onnx"
shape_inference = "defs::tensor::utils::ShapeShapeInference_opset15"
data_propagation = "defs::tensor::utils::ShapeDataPropagation_opset15"

[attributes.end]
type = "INT"
desc = "(Optional) Ending axis for slicing the shape. Negative value means counting dimensions from the back. If omitted, sizes of all axes upto (including) the last one will be included."

[attributes.start]
type = "INT"
desc = "(Optional) Starting axis for slicing the shape. Default value is 0.Negative value means counting dimensions from the back."
default = 0
```

Takes a tensor as input and outputs an 1D int64 tensor containing the shape of the input tensor.
Optional attributes start and end can be used to compute a slice of the input tensor's shape.
If start axis is omitted, the slice starts from axis 0.
The end axis, if specified, is exclusive (and the returned value will not include the size of that axis).
If the end axis is omitted, the axes upto the last one will be included.
Negative axes indicate counting back from the last axis.
Note that axes will be clamped to the range [0, r], where r is the
rank of the input tensor if they are out-of-range (after adding r in the case of
negative axis). Thus, specifying any end value > r is equivalent to specifying an end
value of r, and specifying any start value < -r is equivalent to specifying a start
value of 0. If start > end, the result will be an empty shape.

Examples:

```
Input tensor with shape: [2, 3, 4]
No attributes specified.
Output: [2, 3, 4]
```

```
Input tensor with shape: [2, 3, 4]
start: -1
Output: [4]
```

```
Input tensor with shape: [2, 3, 4]
end: -1
Output: [2, 3]
```

```
Input tensor with shape: [2, 3, 4]
start: 1
end: 2
Output: [3]
```
