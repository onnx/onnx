// /*
//  * SPDX-License-Identifier: Apache-2.0
//  */

#include <algorithm>
#include <cmath>

// #include "onnx/defs/function.h"
#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {

static const char* Gradient_ver1_doc = R"DOC(
Gradient operator computes the partial derivatives of a specific tensor w.r.t.
some other tensors. This operator is widely used in gradient-based training
algorithms. To illustrate its use, let's consider a computation graph,

```
X -----.
       |
       v
W --> Conv --> H --> Gemm --> Y
                      ^
                      |
                      Z
```

, where W and Z are trainable tensors. Note that operators' attributes are
omitted for the sake of simplicity. Let dY/dW (dY/dZ) be the gradient of
Y with respect to W (Z). The user can compute gradient by inserting Gradient
operator to form another graph shown below.

```
W --> Conv --> H --> Gemm --> Y
|      ^              ^
|      |              |
|      X              Z
|      |              |
|      |   .----------'
|      |   |  (W/Z/X is the 1st/2nd/3rd input of Gradient as shown in
|      |   |   "xs" followed by "zs")
|      v   v
'---> Gradient(xs=["W", "Z"], zs=["X"], y="Y")
       |   |
       |   '-----------------------------------> dY/dW (1st output of Gradient)
       |
       '---------------------------------------> dY/dZ (2nd output of Gradient)
```

By definition, the tensor "y" is a function of independent variables in "xs"
and "zs". Since we only compute the gradient of "y" w.r.t. the differentiable
variables in "xs", this Gradient only outputs dY/dW and dY/dZ. Note that "H"
cannot appear in "xs" and "zs". The reason is that "H" can be determined by
tensors "W" and "X" and therefore "H" is not an independent variable.

All outputs are optional. If needed, for example, user can assign an empty
string to the 1st output name of that Gradient to skip the generation of dY/dW.
Note that the concept of optional outputs can also be found in ONNX's RNN, GRU,
and LSTM.

Gradient operator can compute derivative against intermediate tensors. For
example, the gradient of Y with respect to H can be done via

```
W --> Conv --> H --> Gemm --> Y
       ^       |      ^
       |       |      |
       X       |      Z
       .-------'      |
       |   .----------'
       |   | (H/Z is the 1st/2nd input of Gradient as shown in "xs")
       v   v
      Gradient(xs=["H", "Z"], y="Y")
       |   |
       |   '-----------------------------------> dY/dH (1st output of Gradient)
       |
       '---------------------------------------> dY/dZ (2nd output of Gradient)
```

It is possible to represent high-order differentiation using Gradient operators.
For example, given the following linear model:

```
W --> Gemm --> Y --> Loss --> O
       ^              ^
       |              |
       X              L
```

To compute the 2nd order derivative of O with respect to W (denoted by
d^2O/dW^2), one can do

```
W --> Gemm --> Y --> Loss --> O
|      ^              ^
|      |              |
|      X .------------L
|      | |            |
|      | |            v
+------+-+> Gradient(xs=["X", "W"], zs=["L"], y="O") ---> dO/dX (1st output of Gradient)
|      | |    |
|      | |    '---> dO/dW (2nd output of Gradient)
|      v v
'---> Gradient(xs=["X", "W"], zs=["L"], y="dO/dW") ---> d(dO/dW)dX (1st output of
       |                                                  Gradient)
       |
       |
       '---> d^2O/dW^2 (2nd output of Gradient)
```

The tensors named in attributes "xs", "zs", and "y" define the differentiated
computation graph, and the inputs to Gradient node define the values at
which the gradient is computed. We can feed different tensors to the identified
graph. For example, one can compute the gradient of Y with respect to H at
a specific value of H, H_1, by providing that value as an input to the Gradient
node.

```
W --> Conv --> H --> Gemm --> Y
       ^              ^
       |              |
       X              Z

          Z_1 (2nd input of Gradient)
           |
           v
H_1 --> Gradient(xs=["H", "Z"], y="Y") ---> dY/dH when H = H_1 and Y = Y_1.
           |
           '------------------------------> dY/dZ (2nd output of Gradient)
```

When the inputs of Gradient are the tensors named in "xs" and "zs", the
computation can be optimized. More specifically, intermediate variables in
forward pass can be reused if the gradient is computed via reverse-mode
auto-differentiation.

)DOC";

ONNX_PREVIEW_TRAINING_OPERATOR_SET_SCHEMA(
    Gradient,
    1,
    OpSchema()
        .SetDoc(Gradient_ver1_doc)
        .Input(
            0,
            "Inputs",
            "The values fed into graph identified by the attributes. "
            "The i-th input is the value of the i-th tensor specified in the "
            "concatenated list of the attribute \"xs\" and the attribute "
            " \"zs\". For example, if xs=[\"A\", \"B\"] and zs=[\"C\"], the "
            "first input is used as the value of symbol \"A\" and the 3rd "
            "input is substituted for all the occurrences of \"C\".",
            "T1",
            OpSchema::Variadic,
            false)
        .Output(
            0,
            "Outputs",
            "The gradient of the tensor specified by the attribute \"y\" "
            "with respect to each of tensors specified in the "
            "attribute \"xs\". The i-th output is the gradient of \"y\" with "
            "respect to the i-th tensor specified in the attribute \"xs\".",
            "T2",
            OpSchema::Variadic,
            false)
        .Attr(
            "xs",
            "Input tensor names of the differentiated sub-graph. It "
            "contains only the necessary differentiated "
            "inputs of a (sub-)graph. Variables (usually called "
            "intermediate variables) that can be generated from inputs "
            "cannot be included in this attribute.",
            AttributeProto::STRINGS)
        .Attr(
            "zs",
            "Input tensor names of the differentiated sub-graph. It "
            "contains only the necessary non-differentiated "
            "inputs of a (sub-)graph. Variables (usually called "
            "intermediate variables) that can be generated from inputs "
            "cannot be included in this attribute.",
            AttributeProto::STRINGS,
            OPTIONAL_VALUE)
        .Attr(
            "y",
            "The targeted tensor. It can be viewed as the output of the "
            "differentiated function. The attribute \"xs\" and attribute "
            "\"zs\" are the minimal independent variable set that determines "
            "the value of \"y\".",
            AttributeProto::STRING)
        .TypeConstraint("T1", OpSchema::all_tensor_types(), "Allow outputs to be any kind of tensor.")
        .TypeConstraint(
            "T2",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Allow inputs to be any kind of floating-point tensor."));

} // namespace ONNX_NAMESPACE
