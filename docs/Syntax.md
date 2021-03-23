<!--- SPDX-License-Identifier: Apache-2.0 -->

Overview
========

This document describes a textual syntax for ONNX models, which is currently an experimental feature.
The syntax enables a compact and readable representation of ONNX models. It is motivated by a couple
of use-cases. One is to enable compact description of test-cases and its use in CI (both in the ONNX
repo as well as in other dependent repos such as ONNX-MLIR). The second is to help simplify the
definition of ONNX functions. Several of the existing function-definitions are verbose, and the
use of this syntax will lead to more compact, readable, and easier-to-maintain function definitions.

The API
-------

The key parser methods are the ```OnnxParser::Parse``` methods, used as below.

```cpp
  const char* code = R"ONNX(
<
  ir_version: 7,
  opset_import: [ "" : 10 ]
>
agraph (float[N, 128] X, float[128,10] W, float[10] B) => (float[N] C)
{
    T = MatMul(X, W)
    S = Add(T, B)
    C = Softmax(S)
}
)ONNX";

  ModelProto model;
  OnnxParser::Parse(model, code);

  checker::check_model(model);
```

See the [test-cases](../onnx/test/cpp/parser_test.cc) for more examples illustrating the API and syntax. 