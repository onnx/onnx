# Important Functions in ONNX

## Loading an ONNX Model
```python
import onnx

onnx_model = onnx.load('path_to_model_file')
```

## Saving an ONNX Model
```python
import onnx

onnx_model = ... # Your model in memory

# Save the ONNX model
with open('path_to_model_file', 'wb') as f:
    f.write(onnx_model.SerializeToString())
```

## Manipulating TensorProto and Numpy Array
```python
import numpy
import onnx
import os
from onnx import numpy_helper

# Preprocessing: create a Numpy array
numpy_array = numpy.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float)
print('Original Numpy array:\n{}\n'.format(numpy_array))

# Convert the Numpy array to a TensorProto
tensor = numpy_helper.from_array(numpy_array)
print('TensorProto:\n{}'.format(tensor))

# Convert the TensorProto to a Numpy array
new_array = numpy_helper.to_array(tensor)
print('After round trip, Numpy array:\n{}\n'.format(numpy_array))

# Save the TensorProto
with open(os.path.join(os.path.dirname('tensor.pb'), 'wb') as f:
    f.write(tensor.SerializeToString())

# Load a TensorProto
new_tensor = onnx.TensorProto()
with open(os.path.join(os.path.dirname('tensor.pb'), 'rb') as f:
    new_tensor.ParseFromString(f.read())
print('After saving and loading, new TensorProto:\n{}'.format(new_tensor))
```

## Creating an ONNX Model Using Helper Functions
```python
# TODO
```

## Checking an ONNX Model
```python
import onnx
import os

# Preprocessing: Load the ONNX model
model_path = os.path.join(os.path.dirname('model.onnx')
onnx_model = onnx.load(model_path)

onnx.checker.check_model(onnx_model)
print(onnx_model)
```

## Optimizing an ONNX Model
```python
import onnx
import os

# Preprocessing: Load the ONNX model
model_path = os.path.join('model.onnx')
onnx_model = onnx.load(model_path)

onnx.checker.check_model(onnx_model)
print(onnx_model)

# TODO check properties in ONNX model
```

## Running Shape Inference on an ONNX Model
```python
# TODO
```
