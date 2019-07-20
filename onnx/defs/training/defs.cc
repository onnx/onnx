// Copyright (c) ONNX Project Contributors.
// Licensed under the MIT license.

#include <algorithm>
#include <cmath>
#include "onnx/defs/function.h"
#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {
static const char* Gradient_ver12_doc = R"DOC(
Gradient operator computes the partial derivatives of a specific tensor to
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
operator to form

```
W --> Conv --> H --> Gemm --> Y
|      ^              ^
|      |              |
|      X              Z
|      |              |
|      |   .----------'
|      |   |  (X/W/Z is the 1st/2nd/3rd input of Gradient as shown in "xs")
|      v   v
'---> Gradient(xs=["X", "W", "Z"], y="Y") ---> dY/dX (1st output of Gradient)
       |   |
       |   '-----------------------------------> dY/dW (2nd output of Gradient)
       |
       '---------------------------------------> dY/dZ (3rd output of Gradient)
```

, where the content inside the braces of Gradient are attributes. The attribute
"xs" specifies the necessary inputs to compute the variable specified by
attribute "y". To compute gradient, the runtime should identify a sub-graph with
inputs specified by "xs" and outputs containing "y"'s content and do a backward
pass (or other auto-differentiation techniques whenever needed).

Since X is not trainable tensor, the user can avoid the creation of dY/dX by
assigning an empty string to the 1st output name of that Gradient. In other
words, all Gradient's outputs are optional. Note that the concept of optional
outputs can also be found in ONNX's RNN, GRU, and LSTM.

Gradient operator can compute derivative against intermediate tensors. For
example, the gradient of Y with respect to H can be done in

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

It is possible to represent high-order differentiation using Gradient operator.
An example for linear model is

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
|      X              L
|      |
|      |
+------+--> Gradient(xs=["X", "W"], y="Y") ---> dO/dX (1st output of Gradient)
|      |      |
|      |      '---> dO/dW (2nd output of Gradient)
|      v
'---> Gradient(xs=["X", "W"], y="dO/dW") ---> d(dO/dW)dX (1st output of
       |                                          Gradient)
       |
       |
       '---> d^2O/dW^2 (2nd output of Gradient)
```

When the inputs of Gradient are the tensors named in "xs", the computation
can be optimized. More specifically, a forward pass can be reused if the
gradient is computed via reverse-mode auto-differentiation.
We can feed different tensors to the identified graph. For example, one
can compute the gradient of Y with respect to H by substituting Y_1 into Y
and H_1 into H.

```
W --> Conv --> H --> Gemm --> Y
       ^              ^
       |              |
       X              Z

           Z_1 (the 2nd input of Gradient)
           |
           v
W_1 --> Gradient(xs=["H", "Z"], y="Y") ---> dY/dX when Y = Y_1
         |   |
         |   '-----------------------------------> dY/dW (2nd output of Gradient)
         |
         '---------------------------------------> dY/dZ (3rd output of Gradient)
```
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Gradient,
    12,
    OpSchema()
        .SetDoc(Gradient_ver12_doc)
        .Input(
            0,
            "Input",
            "The values fed into graph identified by the attributes. "
            "The i-th input is the value of the i-th tensor specified in the "
            "attribute \"xs\".",
            "V",
            OpSchema::Variadic,
            false)
        .Output(
            0,
            "Output",
            "The gradient of the attribute \"y\" with respect to \"xs\". "
            "The i-th output is the gradient of \"y\" with respect to the "
            "i-th tensor specified in the attribute \"xs\".",
            "V",
            OpSchema::Variadic,
            false)
        .Attr(
            "xs",
            "Input tensor names of the differentiated sub-graph. It "
            "contains and only contains the necessary inputs of a "
            "(sub-)graph. Variables (usually called intermediate "
            "variables) which can be generated by inputs cannot be "
            "included in this attribute.",
            AttributeProto::STRINGS)
        .Attr(
            "y",
            "The differentiated tensor. It is viewed as a function of "
            "the tensors named in \"xs\".",
            AttributeProto::STRING)
        );


} // namespace ONNX_NAMESPACE
