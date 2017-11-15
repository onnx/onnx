### Implementing an ONNX backend

##### What is an ONNX backend

An ONNX backend is a library that can run ONNX models. As there are already many existing Deep Learning frameworks, you very likely don't need to create everything from scratch but rather create a converter that converts ONNX models to the corresponding framework specific representation and then deligate the execution to the framework. E.g. [onnx-caffe2](https://github.com/onnx/onnx-caffe2) , [onnx-coreml](https://github.com/onnx/onnx-coreml), [onnx-tensorflow](https://github.com/onnx/onnx-tensorflow) are all implemented as converters.

##### Unified Backend Interface

ONNX has defined an unified (Python) backend interface at https://github.com/onnx/onnx/blob/master/onnx/backend/base.py. There are three core concepts in this interface: `Device`, `Backend` and `BackendRep`.

- `Device` is a lightweight abstraction of varies hardwares, e.g. CPU, CUDA gpus.

- `Backend` is the entity that will take an ONNX model and the inputs, do the computation and then return the outputs to the users.

  For one-off execution, users could use `run_node` and `run_model` to practically get the results without too much hassle.

  For repeated execution, users should use `prepare` , in which the `Backend` should do all the preparation work for executing the model repeatly (e.g. loading initializers), and return a `BackendRep` as handle to the user

- `BackendRep` is the handle that a Backend returns after prepared itself for executing a model repeatedly. Users will then pass inputs to the `run` function of `BackendRep` for getting the corresponding results.

##### ONNX Backend Test

ONNX has provided a standard backend test suite for helping backend implementations to do verification. We strongly encorage each ONNX backend to run it.

There are two types of tests in this suite: Node Tests and Model Tests.

- Node Tests is intended to verify whehter a backend is doing the correct computation, having the expected bahavior of handling varies attributes for each individual operator. In each test case, backend will be given a node and some inputs, its returned outputs will then be compared with the expected outputs.
- Model Tests is intended to verify the backend at the model level, test cases are similar as in the Node tests, but instead of a node, the backend will be given an ONNX model.

Integrating the ONNX Backend Test suite into your CI should be pretty easy. Here are some examples of how a backend do the integration:

[onnx-caffe2 onnx backend test](https://github.com/onnx/onnx-caffe2/blob/master/tests/onnx_backend_test.py)

[onnx-tensorflow onnx backend test](https://github.com/onnx/onnx-tensorflow/blob/master/test/onnx_backend_test.py)

[onnx-coreml onnx backend test](https://github.com/onnx/onnx-coreml/blob/master/tests/onnx_backend_test.py)

If you have (pytest)[https://docs.pytest.org/en/latest/] installed, you can get a coverage report after running the ONNX backend test to see how well your backend is doing:

```
---------- onnx coverage: ----------
Operators (passed/loaded/total): 21/21/70
------------------------------------
╒════════════════════╤════════════════════╕
│ Operator           │ Attributes         │
│                    │ (name: #values)    │
╞════════════════════╪════════════════════╡
│ Slice              │ axes: 2            │
│                    │ ends: 3            │
│                    │ starts: 3          │
├────────────────────┼────────────────────┤
│ Constant           │ value: 1           │
├────────────────────┼────────────────────┤
│ Concat             │ axis: 0            │
├────────────────────┼────────────────────┤
│ Conv               │ group: 6           │
│                    │ kernel_shape: 5    │
│                    │ pads: 4            │
│                    │ strides: 3         │
│                    │ auto_pad: 0        │
│                    │ dilations: 0       │
├────────────────────┼────────────────────┤
│ Reshape            │ shape: 9           │
├────────────────────┼────────────────────┤
│ BatchNormalization │ consumed_inputs: 1 │
│                    │ epsilon: 2         │
│                    │ is_test: 1         │
│                    │ momentum: 0        │
│                    │ spatial: 0         │
├────────────────────┼────────────────────┤
│ Dropout            │ is_test: 1         │
│                    │ ratio: 2           │
├────────────────────┼────────────────────┤
│ MaxPool            │ kernel_shape: 2    │
│                    │ pads: 3            │
│                    │ strides: 2         │
│                    │ auto_pad: 0        │
│                    │ dilations: 0       │
├────────────────────┼────────────────────┤
│ Transpose          │ perm: 1            │
├────────────────────┼────────────────────┤
│ MatMul             │ No attributes      │
├────────────────────┼────────────────────┤
│ Relu               │ No attributes      │
├────────────────────┼────────────────────┤
│ LRN                │ alpha: 2           │
│                    │ beta: 1            │
│                    │ bias: 2            │
│                    │ size: 1            │
├────────────────────┼────────────────────┤
│ Add                │ axis: 1            │
│                    │ broadcast: 1       │
├────────────────────┼────────────────────┤
│ Abs                │ No attributes      │
├────────────────────┼────────────────────┤
│ Pad                │ mode: 3            │
│                    │ paddings: 2        │
│                    │ value: 1           │
├────────────────────┼────────────────────┤
│ Softmax            │ axis: 0            │
├────────────────────┼────────────────────┤
│ GlobalAveragePool  │ No attributes      │
├────────────────────┼────────────────────┤
│ Mul                │ axis: 1            │
│                    │ broadcast: 1       │
├────────────────────┼────────────────────┤
│ Sum                │ No attributes      │
├────────────────────┼────────────────────┤
│ Gemm               │ broadcast: 1       │
│                    │ transB: 1          │
│                    │ alpha: 0           │
│                    │ beta: 0            │
│                    │ transA: 0          │
├────────────────────┼────────────────────┤
│ AveragePool        │ kernel_shape: 3    │
│                    │ pads: 3            │
│                    │ strides: 2         │
│                    │ auto_pad: 0        │
╘════════════════════╧════════════════════╛
```

The numbers in the line `Operators (passed/loaded/total): 21/21/70` indicate there are 21 opererators have covered in all the test cases your backend have passed, 21 operators covered in all the test cases in the ONNX Backend test, and ONNX has in total 70 operators.