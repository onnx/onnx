```toml
name = "CastMap"
domain = "ai.onnx.ml"
shape_inference = "defs::traditionalml::utils::CastMapShapeInference_opset1"

[attributes.cast_to]
type = "STRING"
desc = "A string indicating the desired element type of the output tensor, one of 'TO_FLOAT', 'TO_STRING', 'TO_INT64'."
default = "TO_FLOAT"

[attributes.map_form]
type = "STRING"
desc = "Indicates whether to only output as many values as are in the input (dense), or position the input based on using the key of the map as the index of the output (sparse).<br>One of 'DENSE', 'SPARSE'."
default = "DENSE"

[attributes.max_map]
type = "INT"
desc = "If the value of map_form is 'SPARSE,' this attribute indicates the total length of the output tensor."
default = 1

[[inputs]]
name = "X"
type = "T1"
desc = "The input map that is to be cast to a tensor"

[[outputs]]
name = "Y"
type = "T2"
desc = "A tensor representing the same data as the input map, ordered by their keys"

[type_constraints.T1]
types = ["map(int64, string)", "map(int64, float)"]
desc = "The input must be an integer map to either string or float."

[type_constraints.T2]
types = ["tensor(string)", "tensor(float)", "tensor(int64)"]
desc = "The output is a 1-D tensor of string, float, or integer."
```

Converts a map to a tensor.<br>The map key must be an int64 and the values will be ordered
in ascending order based on this key.<br>The operator supports dense packing or sparse packing.
If using sparse packing, the key cannot exceed the max_map-1 value.
