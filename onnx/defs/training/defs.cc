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
algorithm field. The mapping from GraphCall inputs to the invoked graph's
input list is determined by the attributes "input_names" and "output_names".
Assume that ModelProto's graph field has

- name: MyInferenceGraph
- inputs: [X, W, Z]
- initializer: [W]
- outputs: Y

as visualized bellow.

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

- inputs: [X_1, Z_1, L]
- outputs: [W_new]

Inside the training algorithm graph, one can invoke the graph via adding
a GraphCall node with

- inputs: [X_1, Z_1]
- outputs: [Y_1]
- an attribute graph_name="MyInferenceGraph",
- an attribute input_names=["X", "Z"]. It means the first input "X_1"
  should be bound to "X" in the inference graph. Similarly, "Z_1" may
  be bound to "Z" in the inference graph. Note that the value of "W" is
  not needed because "W" is also an initializer in the inference graph.
- an attribute output_names=["Y"] which means the inference graph's
  output "Y" should be used as "Y_1" in the training algorithm.

A possible algorithm graph may contain something like

```
.---+---- W (declared in the inference graph's initializer list) 
|   |     '-----------.
|   |                 v
|   | .-- X_1 --> GraphCall(graph_name="MyInferenceGraph",
|   | |            |  |      input_names=["X", "W", "Z"],
|   | |            |  |      output_names=["Y"])
|   | |   Z_1 -----'  |
|   | |    |          V
|   | |    |         Y_1 ---> Loss ---> O
|   | |    |                    ^
|   | |    |                    |
|   | `--. |                    L
|   |    | |                    |
|   |    | |   .----------------'
|   |    | |   |
|   |    v v   v
|   `--> Gradient(xs=["W"], zs=["X_1", "Z_1"], y="O")
|        |
|        v
|      dO_dW (gradient of W)
|        |
|        v
`-----> Sub ----> W_new
```

where Loss is a dummy node which computes the minimized objective function.

Because "W" is an initializer in the called graph, the user can omit it
in the input list of the GraphCall and generate the following training
graph.

```
.---+---- W (declared in the inference graph's initializer list)
|   |
|   |
|   | .-- X_1 --> GraphCall(graph_name="MyInferenceGraph",
|   | |            |  |      input_names=["X", "Z"],
|   | |            |  |      output_names=["Y"])
|   | |   Z_1 -----'  |
|   | |    |          V
|   | |    |         Y_1 ---> Loss ---> O
|   | |    |                    ^
|   | |    |                    |
|   | `--. |                    L
|   |    | |                    |
|   |    | |   .----------------'
|   |    | |   |
|   |    v v   v
|   `--> Gradient(xs=["W"], zs=["X_1", "Z_1"], y="O")
|        |
|        v
|      dO_dW (gradient of W)
|        |
|        v
`-----> Sub ----> W_new
```

)DOC";

ONNX_TRAINING_OPERATOR_SET_SCHEMA(
    GraphCall,
    1,
    OpSchema()
        .SetDoc(GraphCall_ver1_doc)
        .Input(
            0,
            "Inputs",
            "The values fed into the graph specified by the \"graph_name\" attribute. "
            "The i-th input is bound to the input named \"input_names[i]\" in the "
            "called graph.",
            "T",
            OpSchema::Variadic,
            false)
        .Output(
            0,
            "Outputs",
            "The outputs produced by the called graph. Its i-th value is bound to the "
            "output named \"output_names[i]\" in the inference graph.",
            "T",
            OpSchema::Variadic,
            false)
        .Attr(
            "graph_name",
            "The invoked graph's name. "
            "Currently, the only allowed value is \"ModelProto.graph.name\".",
            AttributeProto::STRING)
        .Attr(
            "input_names",
            "Input names of the called graph. Optional inputs of the "
            "called graph can be omitted in this field.",
            AttributeProto::STRINGS)
        .Attr(
            "output_names",
            "Output names of the called graph. Only used outputs need "
            "to be specified in this field.",
            AttributeProto::STRINGS)
        .TypeConstraint(
            "T",
            OpSchema::all_tensor_types(),
            "Allow outputs to be any kind of tensor."));

} // namespace ONNX_NAMESPACE
