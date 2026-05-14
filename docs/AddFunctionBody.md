<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# Adding a Function Body Definition for an Operator

A function body defines how an ONNX operator can be decomposed into simpler ONNX operators. This enables runtimes that don't natively support the operator to still execute it by expanding it into its constituent operations.

## Table of Contents

- [Adding a Function Body Definition for an Operator](#adding-a-function-body-definition-for-an-operator)
  - [Table of Contents](#table-of-contents)
  - [When to use a function body](#when-to-use-a-function-body)
  - [File locations](#file-locations)
  - [Simple function body (string-based)](#simple-function-body-string-based)
  - [Referencing attributes](#referencing-attributes)
  - [Context-dependent function body](#context-dependent-function-body)
  - [FunctionBuilder API](#functionbuilder-api)
  - [Multiple opset versions](#multiple-opset-versions)
  - [ONNX function body syntax](#onnx-function-body-syntax)
  - [Testing](#testing)

## When to use a function body

- The operator can be expressed in terms of other ONNX operators
- You want to provide a reference decomposition that any runtime can use
- The operator is being proposed as a "function" rather than a new primitive (see [Adding New Operator](AddNewOp.md) Step 1)

If an operator can be split into new primitives, prefer proposing those primitives and making the operator a function.

## File locations

| Component | File |
|-----------|------|
| Function body definition | `onnx/defs/<domain>/defs.cc` (inline with the schema) |
| FunctionBuilder utilities | `onnx/defs/function.h` |
| Function tests (C++) | `onnx/test/cpp/function_get_test.cc`, `onnx/test/cpp/function_verify_test.cc` |

## Simple function body (string-based)

For operators whose decomposition is the same regardless of attributes or optional inputs, use the `.FunctionBody()` method with an ONNX-format string:

```cpp
ONNX_OPERATOR_SET_SCHEMA(
    LessOrEqual,
    16,
    OpSchema()
        .SetDoc(LessOrEqual_ver16_doc)
        .Input(0, "A", "First input", "T", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .Input(1, "B", "Second input", "T", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .Output(0, "C", "Result", "T1", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .TypeConstraint("T", OpSchema::all_numeric_types_ir4(), "...")
        .TypeConstraint("T1", {"tensor(bool)"}, "...")
        .TypeAndShapeInferenceFunction(binaryLogicOpInference)
        .FunctionBody(R"ONNX(
        {
            O1 = Less (A, B)
            O2 = Equal (A, B)
            C = Or (O1, O2)
        }
        )ONNX"));
```

You can optionally specify the minimum opset version for which the function body is valid:

```cpp
        .FunctionBody(R"ONNX(
          {
            Zero = Constant <value = float {0.0}>()
            ZeroCast = CastLike (Zero, X)
            Y = Max (X, ZeroCast)
          }
        )ONNX", 18)  // This function body is valid from opset 18 onward
```

## Referencing attributes

Use `@attr_name` syntax to reference the operator's declared attributes inside the function body:

```cpp
ONNX_OPERATOR_SET_SCHEMA(
    LeakyRelu,
    16,
    OpSchema()
        .Attr("alpha", "Coefficient of leakage.", AttributeProto::FLOAT, 0.01f)
        .SetDoc(LeakyRelu_ver16_doc)
        .Input(0, "X", "Input tensor", "T", ...)
        .Output(0, "Y", "Output tensor", "T", ...)
        .TypeConstraint("T", {"tensor(bfloat16)", "tensor(float16)", "tensor(float)", "tensor(double)"}, "...")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput)
        .FunctionBody(R"ONNX(
          {
            Alpha = Constant <value_float: float = @alpha>()
            AlphaCast = CastLike (Alpha, X)
            Zero = Constant <value = float {0.0}>()
            ZeroCast = CastLike(Zero, X)
            XLessThanZero = Less(X, ZeroCast)
            AlphaMulX = Mul (AlphaCast, X)
            Y = Where (XLessThanZero, AlphaMulX, X)
          }
        )ONNX"));
```

The attribute must be declared in the schema's `.Attr()` call for `@attr_name` to work.

## Context-dependent function body

When the decomposition depends on which optional inputs are present, attribute values, or input types, use a context-dependent function body builder:

```cpp
static bool BuildContextDependentFunctionBodyClip(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  bool has_min = ctx.hasInput(1);
  bool has_max = ctx.hasInput(2);

  FunctionBuilder builder(functionProto);
  if (!has_min && !has_max) {
    builder.Add("output = Identity (input)");
  } else if (has_min && !has_max) {
    builder.Add("input_less_than_min = Less (input, min)");
    builder.Add("output = Where (input_less_than_min, min, input)");
  } else if (!has_min && has_max) {
    builder.Add("input_large_than_max = Less (max, input)");
    builder.Add("output = Where (input_large_than_max, max, input)");
  } else {
    builder.Add("input_less_than_min = Less (input, min)");
    builder.Add("tmp = Where (input_less_than_min, min, input)");
    builder.Add("output_large_than_max = Less (max, tmp)");
    builder.Add("output = Where (output_large_than_max, max, tmp)");
  }

  schema.BuildFunction(functionProto);
  return true;
}
```

Register it with the schema:

```cpp
ONNX_OPERATOR_SET_SCHEMA(
    Clip, 13,
    OpSchema()
        .Input(0, "input", "...", "T", OpSchema::Single, ...)
        .Input(1, "min", "...", "T", OpSchema::Optional, ...)
        .Input(2, "max", "...", "T", OpSchema::Optional, ...)
        .Output(0, "output", "...", "T", OpSchema::Single, ...)
        .TypeConstraint("T", OpSchema::all_numeric_types_ir4(), "...")
        .SetContextDependentFunctionBodyBuilder(BuildContextDependentFunctionBodyClip)
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));
```

### FunctionBodyBuildContext API

The context object provides information about the specific instantiation:

```cpp
struct FunctionBodyBuildContext {
  const AttributeProto* getAttribute(const std::string& name) const;  // nullptr if not set
  bool hasInput(int inputIndex) const;     // Is optional input present?
  bool hasOutput(int outputIndex) const;   // Is optional output present?
  const TypeProto* getInputType(int inputIndex) const;  // Input type info
};
```

## FunctionBuilder API

The `FunctionBuilder` class (from `onnx/defs/function.h`) provides a fluent API for constructing function bodies:

```cpp
FunctionBuilder builder(functionProto);

// Add nodes using ONNX text format
builder.Add("Y = Relu (X)");

// Add with inline attributes
builder.Add("X_ReduceMax = ReduceMax <keepdims = 1> (input, axes)");

// Add constants
builder.Const("alpha", std::vector<float>{0.01f});   // Tensor constant
builder.Const1D("axes", int64_t(1));                 // 1-D tensor constant

// Multi-line additions
builder.Add(R"(
    X_Sub = Sub (input, X_ReduceMax)
    X_Exp = Exp (X_Sub)
    X_ReduceSum = ReduceSum <keepdims = 1> (X_Exp, axes)
    output = Div (X_Exp, X_ReduceSum)
)");

// Add opset dependency
builder.AddOpset("", 18);  // default domain, version 18

// Always finalize with:
schema.BuildFunction(functionProto);
return true;
```

## Multiple opset versions

When the function body must change across opset versions (e.g., because a sub-op's signature changed), register multiple builders with explicit version numbers:

```cpp
ONNX_OPERATOR_SET_SCHEMA(
    Softmax, 13,
    OpSchema()
        // ...
        .SetContextDependentFunctionBodyBuilder(builderForOpset13)      // default (since_version)
        .SetContextDependentFunctionBodyBuilder(builderForOpset18, 18)  // opset 18+
);
```

The runtime selects the appropriate function body based on the opset version in the model.

## ONNX function body syntax

The text format for function bodies uses this grammar:

```
output_var = OpName <attr_name = value, ...> (input1, input2, ...)
```

Rules:
- **Variable names** are local intermediates within the function
- **Input/output names** must match the schema's declared `.Input()` and `.Output()` names exactly
- **Constants** are created with the `Constant` op (e.g., `Constant <value = float {0.0}>()`)
- **Type matching** — use `CastLike` instead of `Cast` when the target type depends on an input
- **Attributes** are referenced with `@attr_name` for the enclosing op's attributes

For the formal grammar, see [Syntax.md](Syntax.md). The parser implementation and its tests provide additional examples:

| Resource | File |
|----------|------|
| Formal syntax specification | [docs/Syntax.md](Syntax.md) |
| C++ parser implementation | `onnx/defs/parser.h`, `onnx/defs/parser.cc` |
| Python parser | `onnx/parser.py` |
| C++ parser tests | `onnx/test/cpp/parser_test.cc` |
| Python parser tests | `onnx/test/parser_test.py` |

## Testing

Function bodies are tested in the C++ test suite:

- **`onnx/test/cpp/function_get_test.cc`** — verifies `HasFunction()` and `GetFunction()` return correct results
- **`onnx/test/cpp/function_verify_test.cc`** — verifies function body type constraints and correctness

To run:

```bash
# Build with tests enabled
ONNX_BUILD_TESTS=1 pip install -e . -v

# Run C++ tests (Linux/macOS)
LD_LIBRARY_PATH=./.setuptools-cmake-build/ .setuptools-cmake-build/onnx_gtests --gtest_filter="*Function*"

# Run C++ tests (Windows)
.setuptools-cmake-build\Release\onnx_gtests.exe --gtest_filter="*Function*"
```

The node backend tests (in `onnx/backend/test/case/node/`) also implicitly test function body correctness when the reference implementation uses function expansion.
