<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# Broadcasting in ONNX

In ONNX, element-wise operators can take inputs with different shape,
as long as the input tensors are broadcastable to the same shape.
ONNX supports two types of broadcasting: multidirectional broadcasting and
unidirectional broadcasting. We will introduce these two types of broadcasting
respectively in the following sections.


## Multidirectional Broadcasting

In ONNX, a set of tensors are multidirectional broadcastable to the same shape
if one of the following is true:
- The tensors all have exactly the same shape.
- The tensors all have the same number of dimensions and the length of
each dimensions is either a common length or 1.
- The tensors that have too few dimensions can have their shapes prepended
with a dimension of length 1 to satisfy property 2.

For example, the following tensor shapes are supported by multidirectional broadcasting:

- shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar ==> shape(result) = (2, 3, 4, 5)
- shape(A) = (2, 3, 4, 5), shape(B) = (5,), ==> shape(result) = (2, 3, 4, 5)
- shape(A) = (4, 5), shape(B) = (2, 3, 4, 5), ==> shape(result) = (2, 3, 4, 5)
- shape(A) = (1, 4, 5), shape(B) = (2, 3, 1, 1), ==> shape(result) = (2, 3, 4, 5)
- shape(A) = (3, 4, 5), shape(B) = (2, 1, 1, 1), ==> shape(result) = (2, 3, 4, 5)

Multidirectional broadcasting is the same as [Numpy's broadcasting](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html#general-broadcasting-rules).

Multidirectional broadcasting is supported by the following operators in ONNX:
- [Add](Operators.md#Add)
- [And](Operators.md#And)
- [Div](Operators.md#Div)
- [Equal](Operators.md#Equal)
- [Greater](Operators.md#Greater)
- [Less](Operators.md#Less)
- [Max](Operators.md#Max)
- [Mean](Operators.md#Mean)
- [Min](Operators.md#Min)
- [Mul](Operators.md#Mul)
- [Or](Operators.md#Or)
- [Pow](Operators.md#Pow)
- [Sub](Operators.md#Sub)
- [Sum](Operators.md#Sum)
- [Where](Operators.md#Where)
- [Xor](Operators.md#Xor)

## Unidirectional Broadcasting

In ONNX, tensor B is unidirectional broadcastable to tensor A
if one of the following is true:
- Tensor A and B both have exactly the same shape.
- Tensor A and B all have the same number of dimensions and the length of
each dimensions is either a common length or B's length is 1.
- Tensor B has too few dimensions, and B can have its shapes prepended
with a dimension of length 1 to satisfy property 2.

When unidirectional broadcasting happens, the output's shape is the same as
the shape of A (i.e., the larger shape of two input tensors).

In the following examples, tensor B is unidirectional broadcastable to tensor A:

- shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar ==> shape(result) = (2, 3, 4, 5)
- shape(A) = (2, 3, 4, 5), shape(B) = (5,), ==> shape(result) = (2, 3, 4, 5)
- shape(A) = (2, 3, 4, 5), shape(B) = (2, 1, 1, 5), ==> shape(result) = (2, 3, 4, 5)
- shape(A) = (2, 3, 4, 5), shape(B) = (1, 3, 1, 5), ==> shape(result) = (2, 3, 4, 5)

Unidirectional broadcasting is supported by the following operators in ONNX:
- [Gemm](Operators.md#Gemm)
- [PRelu](Operators.md#PRelu)
