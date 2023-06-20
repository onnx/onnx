<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# Implementing an ONNX backend

## What is an ONNX backend

An ONNX backend is a library that can run ONNX models. Since many deep learning frameworks already exist, you likely won't need to create everything from scratch. Rather, you'll likely create a converter that converts ONNX models to the corresponding framework specific representation and then delegate the execution to the framework. For example, [onnx-caffe2 (as part of caffe2)](https://github.com/pytorch/pytorch/tree/master/caffe2/python/onnx) , [onnx-coreml](https://github.com/onnx/onnx-coreml), and [onnx-tensorflow](https://github.com/onnx/onnx-tensorflow) are all implemented as converters.

## Unified backend interface

ONNX has defined a unified (Python) backend interface at [onnx/backend/base.py](/onnx/backend/base.py).

There are three core concepts in this interface: `Device`, `Backend` and `BackendRep`.

- `Device` is a lightweight abstraction over various hardware, e.g., CPU, GPU, etc.

- `Backend` is the entity that will take an ONNX model with inputs, perform a computation, and then return the output.

  For one-off execution, users can use `run_node` and `run_model` to obtain results quickly.

  For repeated execution, users should use `prepare`, in which the `Backend` does all of the preparation work for executing the model repeatedly (e.g., loading initializers), and returns a `BackendRep` handle.

- `BackendRep` is the handle that a `Backend` returns after preparing to execute a model repeatedly. Users will then pass inputs to the `run` function of `BackendRep` to retrieve the corresponding results.

Note that even though the ONNX unified backend interface is defined in Python, your backend does not need to be implemented in Python. For example, yours can be created in C++, and tools such as [pybind11](https://github.com/pybind/pybind11) or [cython](http://cython.org/) can be used to fulfill the interface.

## ONNX backend test

ONNX provides a standard backend test suite to assist backend implementation verification. It's strongly encouraged that each ONNX backend runs this test.

Integrating the ONNX Backend Test suite into your CI is simple. The following are some examples demonstrating how a backend performs the integration:

- [onnx-caffe2 onnx backend test](https://github.com/pytorch/pytorch/blob/master/caffe2/python/onnx/tests/onnx_backend_test.py)

- [onnx-tensorflow onnx backend test](https://github.com/onnx/onnx-tensorflow/blob/main/test/backend/test_onnx_backend.py)

- [onnx-coreml onnx backend test](https://github.com/onnx/onnx-coreml/blob/master/tests/onnx_backend_models_test.py)

If you have [pytest](https://docs.pytest.org/en/latest/) installed, you can get a coverage report after running the ONNX backend test to see how well your backend is doing:

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

The numbers in the line `Operators (passed/loaded/total): 21/21/70` indicate 21 operators covered in all test cases of your backend have passed, 21 operators were covered in all test cases of the ONNX backend test, and ONNX has a total of 70 operators.
