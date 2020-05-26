# A Short Guide on Defining Differentiability for ONNX operators

The differentiability definition of an operator consists of several aspects.

- Differentiable inputs, which can be referenced in Gradient's `xs` attribute.
- Differentiable outputs, which can be referenced in Gradient's `y` attribute.
- The math equation to compute the Jacobian matrix (or tensor). If a variable (input or output) is differentiable or not is judged by math. If the Jacobian matrix (or tensor) exists, then the considered operator has some differentiable inputs and outputs.

There are several strategies to implement auto-differentiation such as forward accumulation, backward accumulation, and dual variable.
Because most deep learning frameworks are backward-based, the reviewers should ensure the PR authors provide enough details on that.
Subsequently, we discuss several methods to define the differentiability for ONNX operator.

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

## Method 3: Do the Math Using Auto-differentiation Libraries
If the considered operator is not directly available in any existing framework, the author can demonstrate that the gradient function of the considered operator exists using any auto-differentiation library. Of course, the author needs to provide runnable Python scripts and enough details for a smooth review process. It’s similar to the first method, but the author replaces Pytorch with their own auto-differentiation library.
