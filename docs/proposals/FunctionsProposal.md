<!--
Copyright (c) ONNX Project Contributors
-->

<!--- SPDX-License-Identifier: Apache-2.0 -->

## Proposal Adding Function into ONNX

Motivation:
1.  Reduce number of primitive operators in ONNX
To make it easier for hardware vendors to follow ONNX, we want to make it possible to define composite operators in terms of more primitive operators, reducing the number of kernels which must be directly implemented. For example, FC should be declared to be a composition MatMul and Add.

2. Expose customize function capability for graph optimization.
To provide a mechanism of doing graph optimization, say, kernel fusion (merge a subgraph into one node with generated efficient kernel codes). This will in turn help HW acceleration, since common-patterns of kernel fusion may be pre-defined as common functions in ONNX and no sub-graph (function) finding needed for kernel fusion anymore. For example, subgraph having "Add", "Sigmoid", "Tanh", "Mul" nodes could be merged into one fusion node with generated cuda kernel containing "+", "sigmoidf", "tanhf", "*".

3. Provide a flexible RNN implementation.
To define a library of RNN cells and allow the user to write a custom one.

MAJOR CHANGES:
1.	FunctionProto added to represent a function.
2.	FunctionSetProto added to represent a function set.
3.	AttributeProto updated to support function attribute type and allow attribute reference.
4.	ModelProto updated to contain customized function set.

Prototype details can be found [here](https://github.com/linkerzhang/onnx/blob/kezhan/add_function_private/onnx/onnx.in.proto)
