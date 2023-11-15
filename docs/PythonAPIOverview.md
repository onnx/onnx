<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# Python API Overview

The full API is described at [API Reference](https://onnx.ai/onnx/api).

## Loading an ONNX Model

```python
import onnx

# onnx_model is an in-memory ModelProto
onnx_model = onnx.load("path/to/the/model.onnx")
```
Runnable IPython notebooks:
- [load_model.ipynb](/onnx/examples/load_model.ipynb)

## Loading an ONNX Model with External Data


* [Default] If the external data is under the same directory of the model, simply use `onnx.load()`
```python
import onnx

onnx_model = onnx.load("path/to/the/model.onnx")
```

* If the external data is under another directory, use `load_external_data_for_model()` to specify the directory path and load after using `onnx.load()`

```python
import onnx
from onnx.external_data_helper import load_external_data_for_model

onnx_model = onnx.load("path/to/the/model.onnx", load_external_data=False)
load_external_data_for_model(onnx_model, "data/directory/path/")
# Then the onnx_model has loaded the external data from the specific directory
```

## Converting an ONNX Model to External Data
```python
from onnx.external_data_helper import convert_model_to_external_data

# onnx_model is an in-memory ModelProto
onnx_model = ...
convert_model_to_external_data(onnx_model, all_tensors_to_one_file=True, location="filename", size_threshold=1024, convert_attribute=False)
# Then the onnx_model has converted raw data as external data
# Must be followed by save
```

## Saving an ONNX Model
```python
import onnx

# onnx_model is an in-memory ModelProto
onnx_model = ...

# Save the ONNX model
onnx.save(onnx_model, "path/to/the/model.onnx")
```
Runnable IPython notebooks:
- [save_model.ipynb](/onnx/examples/save_model.ipynb)


## Converting and Saving an ONNX Model to External Data
```python
import onnx

# onnx_model is an in-memory ModelProto
onnx_model = ...
onnx.save_model(onnx_model, "path/to/save/the/model.onnx", save_as_external_data=True, all_tensors_to_one_file=True, location="filename", size_threshold=1024, convert_attribute=False)
# Then the onnx_model has converted raw data as external data and saved to specific directory
```


## Manipulating TensorProto and Numpy Array
```python
import numpy
import onnx
from onnx import numpy_helper

# Preprocessing: create a Numpy array
numpy_array = numpy.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float)
print(f"Original Numpy array:\n{numpy_array}\n")

# Convert the Numpy array to a TensorProto
tensor = numpy_helper.from_array(numpy_array)
print(f"TensorProto:\n{tensor}")

# Convert the TensorProto to a Numpy array
new_array = numpy_helper.to_array(tensor)
print(f"After round trip, Numpy array:\n{new_array}\n")

# Save the TensorProto
with open("tensor.pb", "wb") as f:
    f.write(tensor.SerializeToString())

# Load a TensorProto
new_tensor = onnx.TensorProto()
with open("tensor.pb", "rb") as f:
    new_tensor.ParseFromString(f.read())
print(f"After saving and loading, new TensorProto:\n{new_tensor}")

from onnx import TensorProto, helper

# Conversion utilities for mapping attributes in ONNX IR
# The functions below are available after ONNX 1.13
np_dtype = helper.tensor_dtype_to_np_dtype(TensorProto.FLOAT)
print(f"The converted numpy dtype for {helper.tensor_dtype_to_string(TensorProto.FLOAT)} is {np_dtype}.")
storage_dtype = helper.tensor_dtype_to_storage_tensor_dtype(TensorProto.FLOAT)
print(f"The storage dtype for {helper.tensor_dtype_to_string(TensorProto.FLOAT)} is {helper.tensor_dtype_to_string(storage_dtype)}.")
field_name = helper.tensor_dtype_to_field(TensorProto.FLOAT)
print(f"The field name for {helper.tensor_dtype_to_string(TensorProto.FLOAT)} is {field_name}.")
tensor_dtype = helper.np_dtype_to_tensor_dtype(np_dtype)
print(f"The tensor data type for numpy dtype: {np_dtype} is {helper.tensor_dtype_to_string(tensor_dtype)}.")

for tensor_dtype in helper.get_all_tensor_dtypes():
    print(helper.tensor_dtype_to_string(tensor_dtype))

```
Runnable IPython notebooks:
- [np_array_tensorproto.ipynb](/onnx/examples/np_array_tensorproto.ipynb)

## Creating an ONNX Model Using Helper Functions
```python
import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto


# The protobuf definition can be found here:
# https://github.com/onnx/onnx/blob/main/onnx/onnx.proto


# Create one input (ValueInfoProto)
X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 2])
pads = helper.make_tensor_value_info("pads", TensorProto.FLOAT, [1, 4])

value = helper.make_tensor_value_info("value", AttributeProto.FLOAT, [1])


# Create one output (ValueInfoProto)
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 4])

# Create a node (NodeProto) - This is based on Pad-11
node_def = helper.make_node(
    "Pad",                  # name
    ["X", "pads", "value"], # inputs
    ["Y"],                  # outputs
    mode="constant",        # attributes
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],        # nodes
    "test-model",      # name
    [X, pads, value],  # inputs
    [Y],               # outputs
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name="onnx-example")

print(f"The model is:\n{model_def}")
onnx.checker.check_model(model_def)
print("The model is checked!")
```
Runnable IPython notebooks:
- [make_model.ipynb](/onnx/examples/make_model.ipynb)
- [Protobufs.ipynb](/onnx/examples/Protobufs.ipynb)

## Conversion utilities for mapping attributes in ONNX IR
```python
from onnx import TensorProto, helper

np_dtype = helper.tensor_dtype_to_np_dtype(TensorProto.FLOAT)
print(f"The converted numpy dtype for {helper.tensor_dtype_to_string(TensorProto.FLOAT)} is {np_dtype}.")

field_name = helper.tensor_dtype_to_field(TensorProto.FLOAT)
print(f"The field name for {helper.tensor_dtype_to_string(TensorProto.FLOAT)} is {field_name}.")

# There are other useful conversion utilities. Please checker onnx.helper
```

## Checking an ONNX Model
```python
import onnx

# Preprocessing: load the ONNX model
model_path = "path/to/the/model.onnx"
onnx_model = onnx.load(model_path)

print(f"The model is:\n{onnx_model}")

# Check the model
try:
    onnx.checker.check_model(onnx_model)
except onnx.checker.ValidationError as e:
    print(f"The model is invalid: {e}")
else:
    print("The model is valid!")
```
Runnable IPython notebooks:
- [check_model.ipynb](/onnx/examples/check_model.ipynb)

### Checking a Large ONNX Model >2GB
Current checker supports checking models with external data, but for those models larger than 2GB, please use the model path for onnx.checker and the external data needs to be under the same directory.

```python
import onnx

onnx.checker.check_model("path/to/the/model.onnx")
# onnx.checker.check_model(loaded_onnx_model) will fail if given >2GB model
```

## Running Shape Inference on an ONNX Model
```python
import onnx
from onnx import helper, shape_inference
from onnx import TensorProto


# Preprocessing: create a model with two nodes, Y"s shape is unknown
node1 = helper.make_node("Transpose", ["X"], ["Y"], perm=[1, 0, 2])
node2 = helper.make_node("Transpose", ["Y"], ["Z"], perm=[1, 0, 2])

graph = helper.make_graph(
    [node1, node2],
    "two-transposes",
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3, 4))],
    [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (2, 3, 4))],
)

original_model = helper.make_model(graph, producer_name="onnx-examples")

# Check the model and print Y"s shape information
onnx.checker.check_model(original_model)
print(f"Before shape inference, the shape info of Y is:\n{original_model.graph.value_info}")

# Apply shape inference on the model
inferred_model = shape_inference.infer_shapes(original_model)

# Check the model and print Y"s shape information
onnx.checker.check_model(inferred_model)
print(f"After shape inference, the shape info of Y is:\n{inferred_model.graph.value_info}")
```
Runnable IPython notebooks:
- [shape_inference.ipynb](/onnx/examples/shape_inference.ipynb)

### Shape inference a Large ONNX Model >2GB
Current shape_inference supports models with external data, but for those models larger than 2GB, please use the model path for onnx.shape_inference.infer_shapes_path and the external data needs to be under the same directory. You can specify the output path for saving the inferred model; otherwise, the default output path is same as the original model path.

```python
import onnx

# output the inferred model to the original model path
onnx.shape_inference.infer_shapes_path("path/to/the/model.onnx")

# output the inferred model to the specified model path
onnx.shape_inference.infer_shapes_path("path/to/the/model.onnx", "output/inferred/model.onnx")

# inferred_model = onnx.shape_inference.infer_shapes(loaded_onnx_model) will fail if given >2GB model
```

## Running Type Inference on an ONNX Function

```python
import onnx
import onnx.helper
import onnx.parser
import onnx.shape_inference

function_text = """
    <opset_import: [ "" : 18 ], domain: "local">
    CastTo <dtype> (x) => (y) {
        y = Cast <to : int = @dtype> (x)
    }
"""
function = onnx.parser.parse_function(function_text)

# The function above has one input-parameter x, and one attribute-parameter dtype.
# To apply type-and-shape-inference to this function, we must supply the type of
# input-parameter and an attribute value for the attribute-parameter as below:

float_type_ = onnx.helper.make_tensor_type_proto(1, None)
dtype_6 = onnx.helper.make_attribute("dtype", 6)
result = onnx.shape_inference.infer_function_output_types(
    function, [float_type_], [dtype_6]
)
print(result) # a list containing the (single) output type
```

## Converting Version of an ONNX Model within Default Domain (""/"ai.onnx")
```python
import onnx
from onnx import version_converter, helper

# Preprocessing: load the model to be converted.
model_path = "path/to/the/model.onnx"
original_model = onnx.load(model_path)

print(f"The model before conversion:\n{original_model}")

# A full list of supported adapters can be found here:
# https://github.com/onnx/onnx/blob/main/onnx/version_converter.py#L21
# Apply the version conversion on the original model
converted_model = version_converter.convert_version(original_model, <int target_version>)

print(f"The model after conversion:\n{converted_model}")
```

## Utility Functions
### Extracting Sub-model with Inputs Outputs Tensor Names

Function `extract_model()` extracts sub-model from an ONNX model.
The sub-model is defined by the names of the input and output tensors *exactly*.

```python
import onnx

input_path = "path/to/the/original/model.onnx"
output_path = "path/to/save/the/extracted/model.onnx"
input_names = ["input_0", "input_1", "input_2"]
output_names = ["output_0", "output_1"]

onnx.utils.extract_model(input_path, output_path, input_names, output_names)
```

Note: For control-flow operators, e.g. If and Loop, the _boundary of sub-model_,
which is defined by the input and output tensors, should not _cut through_ the
subgraph that is connected to the _main graph_ as attributes of these operators.

### ONNX Compose

`onnx.compose` module provides tools to create combined models.

`onnx.compose.merge_models` can be used to merge two models, by connecting some of the outputs
from the first model with inputs from the second model. By default, inputs/outputs not present in the
`io_map` argument will remain as inputs/outputs of the combined model.

In this example we merge two models by connecting each output of the first model to an input in the second. The resulting model will have the same inputs as the first model and the same outputs as the second:
```python
import onnx

model1 = onnx.load("path/to/model1.onnx")
# agraph (float[N] A, float[N] B) => (float[N] C, float[N] D)
#   {
#      C = Add(A, B)
#      D = Sub(A, B)
#   }

model2 = onnx.load("path/to/model2.onnx")
#   agraph (float[N] X, float[N] Y) => (float[N] Z)
#   {
#      Z = Mul(X, Y)
#   }

combined_model = onnx.compose.merge_models(
    model1, model2,
    io_map=[("C", "X"), ("D", "Y")]
)
```

Additionally, a user can specify a list of `inputs`/`outputs` to be included in the combined model,
effectively dropping the part of the graph that does't contribute to the combined model outputs.
In the following example, we are connecting only one of the two outputs in the first model
to both inputs in the second. By specifying the outputs of the combined model explicitly, we are dropping the output not consumed from the first model, and the relevant part of the graph:
```python
import onnx

# Default case. Include all outputs in the combined model
combined_model = onnx.compose.merge_models(
    model1, model2,
    io_map=[("C", "X"), ("C", "Y")],
)  # outputs: "D", "Z"

# Explicit outputs. "Y" output and the Sub node are not present in the combined model
combined_model = onnx.compose.merge_models(
    model1, model2,
    io_map=[("C", "X"), ("C", "Y")],
    outputs=["Z"],
)  # outputs: "Z"
```

`onnx.compose.add_prefix` allows you to add a prefix to names in the model, to avoid a name collision
when merging them. By default, it renames all names in the graph: inputs, outputs, edges, nodes,
initializers, sparse initializers and value infos.

```python
import onnx

model = onnx.load("path/to/the/model.onnx")
# model - outputs: ["out0", "out1"], inputs: ["in0", "in1"]

new_model = onnx.compose.add_prefix(model, prefix="m1/")
# new_model - outputs: ["m1/out0", "m1/out1"], inputs: ["m1/in0", "m1/in1"]

# Can also be run in-place
onnx.compose.add_prefix(model, prefix="m1/", inplace=True)
```

`onnx.compose.expand_out_dim` can be used to connect models that expect a different number
 of dimensions by inserting dimensions with extent one. This can be useful, when combining a
 model producing samples with a model that works with batches of samples.

```python
import onnx

# outputs: "out0", shape=[200, 200, 3]
model1 = onnx.load("path/to/the/model1.onnx")

# outputs: "in0", shape=[N, 200, 200, 3]
model2 = onnx.load("path/to/the/model2.onnx")

# outputs: "out0", shape=[1, 200, 200, 3]
new_model1 = onnx.compose.expand_out_dims(model1, dim_idx=0)

# Models can now be merged
combined_model = onnx.compose.merge_models(
    new_model1, model2, io_map=[("out0", "in0")]
)

# Can also be run in-place
onnx.compose.expand_out_dims(model1, dim_idx=0, inplace=True)
```

## Tools
### Updating Model"s Inputs Outputs Dimension Sizes with Variable Length
Function `update_inputs_outputs_dims` updates the dimension of the inputs and outputs of the model,
to the provided values in the parameter. You could provide both static and dynamic dimension size,
by using dim_param. For more information on static and dynamic dimension size, checkout [Tensor Shapes](IR.md#tensor-shapes).

The function runs model checker after the input/output sizes are updated.
```python
import onnx
from onnx.tools import update_model_dims

model = onnx.load("path/to/the/model.onnx")
# Here both "seq", "batch" and -1 are dynamic using dim_param.
variable_length_model = update_model_dims.update_inputs_outputs_dims(model, {"input_name": ["seq", "batch", 3, -1]}, {"output_name": ["seq", "batch", 1, -1]})
```

## ONNX Parser

Functions `onnx.parser.parse_model` and `onnx.parser.parse_graph` can be used to create an ONNX model
or graph from a textual representation as shown below. See [Language Syntax](Syntax.md) for more details
about the language syntax.

```python
input = """
   agraph (float[N, 128] X, float[128, 10] W, float[10] B) => (float[N, 10] C)
   {
        T = MatMul(X, W)
        S = Add(T, B)
        C = Softmax(S)
   }
"""
graph = onnx.parser.parse_graph(input)

input = """
   <
     ir_version: 7,
     opset_import: ["" : 10]
   >
   agraph (float[N, 128] X, float[128, 10] W, float[10] B) => (float[N, 10] C)
   {
      T = MatMul(X, W)
      S = Add(T, B)
      C = Softmax(S)
   }
"""
model = onnx.parser.parse_model(input)

```

## ONNX Inliner

Functions `onnx.inliner.inline_local_functions` and `inline_selected_functions` can be used
to inline model-local functions in an ONNX model. In particular, `inline_local_functions` can
be used to produce a function-free model (suitable for backends that do not handle or support
functions). On the other hand, `inline_selected_functions` can be used to inline selected
functions. There is no support yet for inlining ONNX standard ops that are functions (also known
as schema-defined functions).

```python
import onnx
import onnx.inliner

model = onnx.load("path/to/the/model.onnx")
inlined = onnx.inliner.inline_local_functions(model)
onnx.save("path/to/the/inlinedmodel.onnx")
```