# A Short Guide on Defining Differentiability for ONNX operators

The differentiability definition of an operator consists of several aspects.

- Differentiable inputs, which can be referenced in Gradient's `xs` attribute.
- Differentiable outputs, which can be referenced in Gradient's `y` attribute.
- The math equation to compute the Jacobian matrix (or tensor). If a variable (input or output) is differentiable or not is judged by math. If the Jacobian matrix (or tensor) exists, then the considered operator has some differentiable inputs and outputs.

There are several strategies to implement auto-differentiation such as forward accumulation, backward accumulation, and dual variable.
Because most deep learning frameworks are backward-based, the reviewers should ensure the PR authors provide enough details on that.
Subsequently, we discuss several methods to verify the differentiability for ONNX operator.

## Method 1: Reuse Existing Deep Learning Frameworks
The first way is to show that the considered operator's backward operation exists in an existing framework such as Pytorch or Tensorflow. In this case, the author should provide a runnable python script which computes the backward pass of the considered operator. The author should also points out how to map the Pytorch or Tensor code to ONNX format (for example, the author can call `torch.onnx.export` to save an ONNX model). The following script shows the differentiability of ONNX Reshape using  Pytorch.

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

## Method 2: Manually Do the Math
The second way is formally proving the existence of the Jacobian matrix (or tensor) from outputs to inputs with at least two numerical examples. In this case, the reviewer should go through the math and confirm if the numerical result is correct. The author should add enough details so that any STEM graduated student can easily review it. 

For example, to show the differentiability of Add, the author may first write down its equation:
```
C = A + B
```
For the sake of simplicity, assume `A` and `B` are same-shape vector.
```
A = [a1, a2]
B = [b1, b2]
C = [c1, c2]
```
Let `X = [a1, a2, b1, b2]` and `Y = [c1, c2]` and consider Add as a function which maps `X` to `Y`.
Then, this function's Jacobian matrix is a 4-by-2 matrix,
```
J = [[dc1/da1, dc2/da1],
     [dc1/da2, dc2/da2],
     [dc1/db1, dc2/db1],
     [dc1/db2, dc2/db2]]
  = [[1, 0],
     [1, 0],
     [0, 1],
     [0, 1]]
```
If
```
dL/dC = [dL/dc1, dL/dc2],
```
then `dL/dA = [dL/da1, dL/da2]` and `dL/dB = [dL/db1, dL/db2]` can be computed from elements in
```
  [[dL/da1], [dL/da2], [dL/db1], [dL/db2]]
= J * dL/dC
= [[dL/dc1], [dL/dc1], [dL/dc2], [dL/dc2]]
```
where `*` is standard matrix multiplication.
If `dL/dC = [0.2, 0.8]`, then `dL/dA = [0.2, 0.2]` and `dL/dB = [0.8, 0.8]`.
Notice that the procedure to compute `dL/dA` and `dL/dB` from `dL/dC` is usually called backward of an operator.
We can see backward operator of Add takes `dL/dC` as an input and produces two outputs `dL/dA` and `dL/dB`.
This example can be extended to cover all tensors when shape broadcasting is not needed.
If broadcasting happens, the broadcasted element's gradient is the sum of all associated elements' gradient in its **non-broadcasting** case.
Let's consider the above example again.
If `B = [b]` becomes an 1-element vector, `B` may be broadcasted to `[b1, b1]` and `dL/dB = [dL/ db] = [dL/db1 + dL/db2]`.
For high-dimensional tensors, this is in fact a ReduceSum operation along all expanded axes.

## Method 3: Do the Math Using Auto-differentiation Libraries
If the considered operator is not directly available in any existing framework, the author can demonstrate that the gradient function of the considered operator exists using any auto-differentiation library. Of course, the author needs to provide runnable Python scripts and enough details for a smooth review process. It’s similar to the first method, but the author replaces Pytorch with their own auto-differentiation library.
