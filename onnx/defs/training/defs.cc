// Copyright (c) ONNX Project Contributors.
// Licensed under the MIT license.

#include <algorithm>
#include <cmath>
#include "onnx/defs/function.h"
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

ONNX_TRAINING_OPERATOR_SET_SCHEMA(
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
            OPTIONAL)
        .Attr(
            "y",
            "The targeted tensor. It can be viewed as the output of the "
            "differentiated function. The attribute \"xs\" and attribute "
            "\"zs\" are the minimal independent variable set that determines "
            "the value of \"y\".",
            AttributeProto::STRING)
        .TypeConstraint(
            "T1",
            OpSchema::all_tensor_types(),
            "Allow outputs to be any kind of tensor.")
        .TypeConstraint(
            "T2",
            {"tensor(float16)",
             "tensor(float)",
             "tensor(double)"},
            "Allow inputs to be any kind of floating-point tensor."));


static const char* GraphCall_ver1_doc = R"DOC(
The GraphCall operator invokes a graph inside TrainingInfoProto's
algorithm field. The GraphCall inputs and outputs are bound to those of
invoked graph by position. If a graph input has an initializer, that input
is considered optional. All graph outputs are optional.

Below Python syntax is used for describing dictionary and list.

Assume that ModelProto's graph field has
- name: "MyInferenceGraph"
- input: ["X", "W", "Z"]
- initializer: [W]
- output: ["Y"]

as visualized below for inference.

```
X -----.
       |
       v
W --> Conv --> H --> Gemm --> Y
                      ^
                      |
                      Z
```

Assume that the training algorithm contains

- inputs: ["X_1", "Z_1", "C"]
- initializer: [T]
- outputs: ["W_new"]

with a dictionary

- update_binding: {"W": "W_new", "T": "T_new"}

Inside the training algorithm graph, one can invoke the inference
graph via adding a GraphCall node with

- inputs: ["X_1", "W", Z_1"]
- outputs: ["Y_1"]
- an attribute graph_name="MyInferenceGraph",

The initializers, "W" and "T" in this case, in update_binding
are considered globally-visible and mutable variables, which
can be used as inputs of operators in the training graph.

An example training algorithm graph may look like

```
.-------- W (a global and mutable variable from
|         |  the inference graph)
|         |
|   .-----'-----------.
|   |                 |
|   |                 v
|   | .-- X_1 --> GraphCall(graph_name="MyInferenceGraph")
|   | |            |  |
|   | |            |  |
|   | |   Z_1 -----'  |
|   | |    |          V
|   | |    |         Y_1 ---> Loss ---> O
|   | |    |                    ^
|   | |    |                    |
|   | `--. |                    C
|   |    | |                    |
|   |    | |   .----------------'
|   |    | |   |
|   |    v v   v
|   `--> Gradient(xs=["W"], zs=["X_1", "Z_1", "C"], y="O")
|        |
|        v
|      dO_dW (gradient of W)      1 (a scalar one)
|        |                        |
|        V                        v
|       Div <--- T ------------> Add ---> T_new
|        |    (T is the number of training iterations.
|        |     T is also globally visible and mutable.)
|        v
`-----> Sub ----> W_new
```

where Loss is a dummy node which computes the minimized objective function.

The variable "W" is an optional input in the called graph.
If the user omits it, the input list of GraphCall becomes ["X_1", "", "Z_1"].
In this case, from the view of computation graph, the Conv operator invoked by
GraphCall's may be still connected the global "W" variable and therefore the
structure of the computation graph is unchanged.
)DOC";

ONNX_TRAINING_OPERATOR_SET_SCHEMA(
    GraphCall,
    1,
    OpSchema()
        .SetDoc(GraphCall_ver1_doc)
        .Input(
            0,
            "Inputs",
            "Inputs fed to the invoked graph. "
            "The i-th input here goes to the i-th input of the invoked graph. "
            "To omit an optional input in this field, "
            "the user can drop it or use an empty string.",
            "T",
            OpSchema::Variadic,
            false)
        .Output(
            0,
            "Outputs",
            "The outputs generated by the called graph. "
            "Its i-th value is bound to the i-th output of the called graph. "
            "Similar to the inputs, all outputs are optional.",
            "T",
            OpSchema::Variadic,
            false)
        .Attr(
            "graph_name",
            "The invoked graph's name. "
            "The only allowed value is the name of the inference graph, "
            "which is stored in \"ModelProto.graph.name\" "
            "in the ONNX model format.",
            AttributeProto::STRING)
        .TypeConstraint(
            "T",
            OpSchema::all_tensor_types(),
            "Allow inputs and outputs to be any kind of tensor."));

} // namespace ONNX_NAMESPACE
