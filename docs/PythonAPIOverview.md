# Python API Overview

## Loading an ONNX Model
```python
import onnx

onnx_model = onnx.load('path/to/the/model.onnx')
# `onnx_model` is a ModelProto struct
```
Runnable IPython notebooks:
- [load_model.ipynb](https://github.com/onnx/onnx/tree/master/onnx/examples/load_model.ipynb)

## Loading an ONNX Model with External Data


* [Default] If the external data is under the same directory of the model, simply use `onnx.load()`
```python
import onnx

onnx_model = onnx.load('path/to/the/model.onnx')
```

* If the external data is under another directory, use `load_external_data_for_model()` to specify the directory path and load after using `onnx.load()`

```python
import onnx
from onnx.external_data_helper import load_external_data_for_model

onnx_model = onnx.load('path/to/the/model.onnx', load_external_data=False)
load_external_data_for_model(onnx_model, 'data/directory/path/')
# Then the onnx_model has loaded the external data from the specific directory
```

## Saving an ONNX Model
```python
import onnx

onnx_model = ... # Your model in memory as ModelProto

# Save the ONNX model
onnx.save(onnx_model, 'path/to/the/model.onnx')
```
Runnable IPython notebooks:
- [save_model.ipynb](https://github.com/onnx/onnx/tree/master/onnx/examples/save_model.ipynb)

## Manipulating TensorProto and Numpy Array
```python
import numpy
import onnx
from onnx import numpy_helper

# Preprocessing: create a Numpy array
numpy_array = numpy.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float)
print('Original Numpy array:\n{}\n'.format(numpy_array))

# Convert the Numpy array to a TensorProto
tensor = numpy_helper.from_array(numpy_array)
print('TensorProto:\n{}'.format(tensor))

# Convert the TensorProto to a Numpy array
new_array = numpy_helper.to_array(tensor)
print('After round trip, Numpy array:\n{}\n'.format(new_array))

# Save the TensorProto
with open('tensor.pb', 'wb') as f:
    f.write(tensor.SerializeToString())

# Load a TensorProto
new_tensor = onnx.TensorProto()
with open('tensor.pb', 'rb') as f:
    new_tensor.ParseFromString(f.read())
print('After saving and loading, new TensorProto:\n{}'.format(new_tensor))
```
Runnable IPython notebooks:
- [np_array_tensorproto.ipynb](https://github.com/onnx/onnx/tree/master/onnx/examples/np_array_tensorproto.ipynb)

## Creating an ONNX Model Using Helper Functions
```python
import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto


# The protobuf definition can be found here:
# https://github.com/onnx/onnx/blob/master/onnx/onnx.proto


# Create one input (ValueInfoProto)
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [3, 2])
pads = helper.make_tensor_value_info('pads', TensorProto.FLOAT, [1, 4])

value = helper.make_tensor_value_info('value', AttributeProto.FLOAT, [1])


# Create one output (ValueInfoProto)
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [3, 4])

# Create a node (NodeProto) - This is based on Pad-11
node_def = helper.make_node(
    'Pad', # node name
    ['X', 'pads', 'value'], # inputs
    ['Y'], # outputs
    mode='constant', # attributes
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [X, pads, value],
    [Y],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-example')

print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')
```
Runnable IPython notebooks:
- [make_model.ipynb](https://github.com/onnx/onnx/tree/master/onnx/examples/make_model.ipynb)
- [Protobufs.ipynb](https://github.com/onnx/onnx/tree/master/onnx/examples/Protobufs.ipynb)

## Checking an ONNX Model
```python
import onnx

# Preprocessing: load the ONNX model
model_path = 'path/to/the/model.onnx'
onnx_model = onnx.load(model_path)

print('The model is:\n{}'.format(onnx_model))

# Check the model
onnx.checker.check_model(onnx_model)
print('The model is checked!')
```
Runnable IPython notebooks:
- [check_model.ipynb](https://github.com/onnx/onnx/tree/master/onnx/examples/check_model.ipynb)

### Checking a Large ONNX Model >2GB
Current checker supports checking models with external data, but for those models larger than 2GB, please use the model path for onnx.checker and the external data needs to be under the same directory. 

```python
import onnx

onnx.checker.check_model('path/to/the/model.onnx')
# onnx.checker.check_model(loaded_onnx_model) will fail if given >2GB model
```


## Optimizing an ONNX Model
```python
import onnx
from onnx import optimizer

# Preprocessing: load the model to be optimized.
model_path = 'path/to/the/model.onnx'
original_model = onnx.load(model_path)

print('The model before optimization:\n{}'.format(original_model))

# A full list of supported optimization passes can be found using get_available_passes()
all_passes = optimizer.get_available_passes()
print("Available optimization passes:")
for p in all_passes:
    print(p)
print()

# Pick one pass as example
passes = ['fuse_consecutive_transposes']

# Apply the optimization on the original model
optimized_model = optimizer.optimize(original_model, passes)

print('The model after optimization:\n{}'.format(optimized_model))

# One can also apply the default passes on the (serialized) model
# Check the default passes here: https://github.com/onnx/onnx/blob/master/onnx/optimizer.py#L43
optimized_model = optimizer.optimize(original_model)
```
Runnable IPython notebooks:
- [optimize_onnx.ipynb](https://github.com/onnx/onnx/tree/master/onnx/examples/optimize_onnx.ipynb)

## Running Shape Inference on an ONNX Model
```python
import onnx
from onnx import helper, shape_inference
from onnx import TensorProto


# Preprocessing: create a model with two nodes, Y's shape is unknown
node1 = helper.make_node('Transpose', ['X'], ['Y'], perm=[1, 0, 2])
node2 = helper.make_node('Transpose', ['Y'], ['Z'], perm=[1, 0, 2])

graph = helper.make_graph(
    [node1, node2],
    'two-transposes',
    [helper.make_tensor_value_info('X', TensorProto.FLOAT, (2, 3, 4))],
    [helper.make_tensor_value_info('Z', TensorProto.FLOAT, (2, 3, 4))],
)

original_model = helper.make_model(graph, producer_name='onnx-examples')

# Check the model and print Y's shape information
onnx.checker.check_model(original_model)
print('Before shape inference, the shape info of Y is:\n{}'.format(original_model.graph.value_info))

# Apply shape inference on the model
inferred_model = shape_inference.infer_shapes(original_model)

# Check the model and print Y's shape information
onnx.checker.check_model(inferred_model)
print('After shape inference, the shape info of Y is:\n{}'.format(inferred_model.graph.value_info))
```
Runnable IPython notebooks:
- [shape_inference.ipynb](https://github.com/onnx/onnx/tree/master/onnx/examples/shape_inference.ipynb)

### Shape inference a Large ONNX Model >2GB
Current shape_inference supports models with external data, but for those models larger than 2GB, please use the model path for onnx.shape_inference.infer_shapes_path and the external data needs to be under the same directory. You can specify the output path for saving the inferred model; otherwise, the default output path is same as the original model path.

```python
import onnx

# output the inferred model to the original model path
onnx.shape_inference.infer_shapes_path('path/to/the/model.onnx')

# output the inferred model to the specified model path
onnx.shape_inference.infer_shapes_path('path/to/the/model.onnx', 'output/inferred/model.onnx')

# inferred_model = onnx.shape_inference.infer_shapes(loaded_onnx_model) will fail if given >2GB model
```

## Converting Version of an ONNX Model within Default Domain (""/"ai.onnx")
```python
import onnx
from onnx import version_converter, helper

# Preprocessing: load the model to be converted.
model_path = 'path/to/the/model.onnx'
original_model = onnx.load(model_path)

print('The model before conversion:\n{}'.format(original_model))

# A full list of supported adapters can be found here:
# https://github.com/onnx/onnx/blob/master/onnx/version_converter.py#L21
# Apply the version conversion on the original model
converted_model = version_converter.convert_version(original_model, <int target_version>)

print('The model after conversion:\n{}'.format(converted_model))
```

## Utility Functions
### Polishing the Model
Function `polish_model` runs model checker, optimizer, shape inference engine on the model,
and also strips the doc_string for you.
```python
import onnx


model = onnx.load('path/to/the/model.onnx')
polished_model = onnx.utils.polish_model(model)
```

### Extracting Sub-model with Inputs Outputs Tensor Names

Function `extract_model()` extracts sub-model from an ONNX model.
The sub-model is defined by the names of the input and output tensors *exactly*.

```python
import onnx

input_path = 'path/to/the/original/model.onnx'
output_path = 'path/to/save/the/extracted/model.onnx'
input_names = ['input_0', 'input_1', 'input_2']
output_names = ['output_0', 'output_1']

onnx.utils.extract_model(input_path, output_path, input_names, output_names)
```

Note: For control-flow operators, e.g. If and Loop, the _boundary of sub-model_,
which is defined by the input and output tensors, should not _cut through_ the
subgraph that is connected to the _main graph_ as attributes of these operators.

## Tools
### Updating Model's Inputs Outputs Dimension Sizes with Variable Length
Function `update_inputs_outputs_dims` updates the dimension of the inputs and outputs of the model,
to the provided values in the parameter. You could provide both static and dynamic dimension size,
by using dim_param. For more information on static and dynamic dimension size, checkout [Tensor Shapes](IR.md#tensor-shapes).

The function runs model checker after the input/output sizes are updated.
```python
import onnx
from onnx.tools import update_model_dims

model = onnx.load('path/to/the/model.onnx')
# Here both 'seq', 'batch' and -1 are dynamic using dim_param.
variable_length_model = update_model_dims.update_inputs_outputs_dims(model, {'input_name': ['seq', 'batch', 3, -1]}, {'output_name': ['seq', 'batch', 1, -1]})
```
