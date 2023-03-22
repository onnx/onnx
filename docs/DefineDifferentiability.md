<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# A Short Guide on the Differentiability Tag for ONNX Operators

## Differentiability Tag
The ONNX operator schema for each operator includes a differentiability tag for each input and output.
In this document, we explain the meaning of this tag and how to ensure the correctness of the tags.
Briefly, the tag identifies the set of differentiable inputs and differentiable outputs of an operator.
The meaning of the tag is that the partial derivative of each differentiable output is defined with respect to each differentiable output.

## Ways to Define Differentiability Tag
The differentiability definition of an operator consists of several aspects.

- Differentiable inputs, which can be referenced in Gradient's `xs` attribute.
- Differentiable outputs, which can be referenced in Gradient's `y` attribute.
- The math equation to compute the Jacobian matrix (or tensor). If a variable (input or output) is differentiable or not is judged by math. If the Jacobian matrix (or tensor) exists, then the considered operator has some differentiable inputs and outputs.

There are several strategies to implement auto-differentiation such as forward accumulation, backward accumulation, and dual variable.
Because most deep learning frameworks are backward-based, the reviewers should ensure the PR authors of tags provide enough details on that.
We present a couple of methods below to verify the differentiability for ONNX operator.

### Method 1: Reuse Existing Deep Learning Frameworks
The first way is to show that the considered operator's backward operation exists in an existing framework such as Pytorch or Tensorflow. In this case, the author should provide a runnable python script which computes the backward pass of the considered operator. The author should also point out how to map the Pytorch or Tensor code to ONNX format (for example, the author can call `torch.onnx.export` to save an ONNX model). The following script shows the differentiability of ONNX Reshape using  Pytorch.

```python
import torch
import torch.nn as nn

# A single-operator model. It's literally a Pytorch Reshape.
# Note that Pytorch Reshape can be directly mapped to ONNX Reshape.
class MyModel(nn.Module):
  def __init__(self):
    super(MyModel, self).__init__()

  def forward(self, x):
    y = torch.reshape(x, (x.numel(),))
    y.retain_grad()
    return y

model = MyModel()

x = torch.tensor([[1., -1.], [1., 1.]], requires_grad=True)
y = model(x)
dy = torch.tensor([1., 2., 3., 4.])

torch.autograd.backward([y],
  grad_tensors=[dy],
  retain_graph=True,
  create_graph=True,
  grad_variables=None)

# This example shows the input and the output in Pytorch are differentiable.
# From the exported ONNX model below, we also see that "x" is the first input
# of ONNX Reshape and "y" the output of ONNX Reshape. Therefore, we can say
# the first input and the output of ONNX Reshape are differentiable.
print(x.grad)
print(y.grad)

with open('model.onnx', 'wb') as f:
  torch.onnx.export(model, x, f)
```

### Method 2: Manually Do the Math
The second way is formally proving the existence of the Jacobian matrix (or tensor) from outputs to inputs with at least two numerical examples. In this case, the reviewer should go through the math and confirm if the numerical result is correct. The author should add enough details so that any STEM graduated student can easily review it.

For example, to show the differentiability of Add, the author may first write down its equation:
```
C = A + B
```
For the sake of simplicity, assume `A` and `B` are same-shape vector.
```
A = [a1, a2]^T
B = [b1, b2]^T
C = [c1, c2]^T
```
Here we use the symbol `^T` to denote transpose of the attached matrix or vector.
Let `X = [a1, a2, b1, b2]^T` and `Y = [c1, c2]^T` and consider Add as a function which maps `X` to `Y`.
Then, this function's Jacobian matrix is a 4-by-2 matrix,
```
J = [[dc1/da1, dc2/da1],
     [dc1/da2, dc2/da2],
     [dc1/db1, dc2/db1],
     [dc1/db2, dc2/db2]]
  = [[1, 0],
     [0, 1],
     [1, 0],
     [0, 1]]
```
If
```
dL/dC = [dL/dc1, dL/dc2]^T,
```
then `dL/dA = [dL/da1, dL/da2]^T` and `dL/dB = [dL/db1, dL/db2]^T` can be computed from elements in
```
  [[dL/da1], [dL/da2], [dL/db1], [dL/db2]]
= J * dL/dC
= [[dL/dc1], [dL/dc2], [dL/dc1], [dL/dc2]]
```
where `*` is standard matrix multiplication.
If `dL/dC = [0.2, 0.8]^T`, then `dL/dA = [0.2, 0.8]^T` and `dL/dB = [0.2, 0.8]^T`.
Notice that the procedure to compute `dL/dA` and `dL/dB` from `dL/dC` is usually called backward of an operator.
We can see backward operator of Add takes `dL/dC` as an input and produces two outputs `dL/dA` and `dL/dB`.
Consequently, all of `A`, `B`, and `C` are differentiable.
By flattening tensor into 1-D vector, this example can be extended to cover all tensors when shape broadcasting is not needed.
If broadcasting happens, the broadcasted element's gradient is the sum of all associated elements' gradient in its **non-broadcasting** case.
Let's consider the above example again.
If `B = [b]^T` becomes an 1-element vector, `B` may be broadcasted to `[b1, b2]^T` and `dL/dB = [dL/ db]^T = [dL/db1 + dL/db2]^T`.
For high-dimensional tensors, this is in fact a ReduceSum operation along all expanded axes.
