---
name: add-function-body
description: Add a function body definition to an ONNX operator, defining how it decomposes into simpler ops. Use when asked to make an op decomposable, add a FunctionBody, implement SetContextDependentFunctionBodyBuilder, or express an op in terms of other ONNX operators.
---

Follow the full guide in [docs/AddFunctionBody.md](../../docs/AddFunctionBody.md).

## File Locations

| Component | File |
|-----------|------|
| Function body definition | `onnx/defs/<domain>/defs.cc` (inline with schema) |
| FunctionBuilder utilities | `onnx/defs/function.h` |
| Function tests | `onnx/test/cpp/function_get_test.cc`, `onnx/test/cpp/function_verify_test.cc` |

## Method 1: Simple String-Based Function Body

```cpp
ONNX_OPERATOR_SET_SCHEMA(
    LessOrEqual, 16,
    OpSchema()
        // ... inputs, outputs, type constraints ...
        .TypeAndShapeInferenceFunction(inferenceFunction)
        .FunctionBody(R"ONNX(
        {
            O1 = Less (A, B)
            O2 = Equal (A, B)
            C = Or (O1, O2)
        }
        )ONNX"));
```

With explicit opset version:
```cpp
        .FunctionBody(R"ONNX(...)ONNX", 18)  // Valid from opset 18
```

Referencing attributes with `@attr_name`:
```cpp
        .FunctionBody(R"ONNX(
          {
            Alpha = Constant <value_float: float = @alpha>()
            AlphaCast = CastLike (Alpha, X)
            ...
          }
        )ONNX")
```

## Method 2: Context-Dependent Function Body

For ops whose decomposition varies based on attributes or optional inputs:

```cpp
static bool BuildFunctionBodyMyOp(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  FunctionBuilder builder(functionProto);
  // Build graph based on ctx.hasInput(), ctx.getAttribute(), etc.
  builder.Add("output = SomeOp (input)");
  schema.BuildFunction(functionProto);
  return true;
}

// Register:
    .SetContextDependentFunctionBodyBuilder(BuildFunctionBodyMyOp)
```

## FunctionBuilder API

```cpp
FunctionBuilder builder(functionProto);
builder.Add("Y = Relu (X)");                          // Add node
builder.Const("alpha", std::vector<float>{0.01f});    // Constant tensor
builder.Const1D("axes", int64_t(1));                  // 1-D constant
builder.Add(R"(                                       // Multi-line
    X_Sub = Sub (X, X_Max)
    X_Exp = Exp (X_Sub)
)");
builder.AddOpset("", 18);                             // Opset dependency
schema.BuildFunction(functionProto);                  // Finalize
return true;
```

## Multiple Opset Versions

```cpp
    .SetContextDependentFunctionBodyBuilder(builderForOpset13)
    .SetContextDependentFunctionBodyBuilder(builderForOpset18, 18)
```

## ONNX Function Body Syntax

```
variable = OpName <attr = value> (input1, input2)
```

- Variable names are local intermediates
- Input/output names must match schema declarations
- Use `CastLike` (not `Cast`) when target type depends on input
- Reference attributes with `@attr_name`

For the formal grammar, see [docs/Syntax.md](../../docs/Syntax.md). Parser tests provide additional examples:

| Resource | File |
|----------|------|
| Formal syntax specification | `docs/Syntax.md` |
| C++ parser | `onnx/defs/parser.h`, `onnx/defs/parser.cc` |
| Python parser | `onnx/parser.py` |
| C++ parser tests | `onnx/test/cpp/parser_test.cc` |
| Python parser tests | `onnx/test/parser_test.py` |

## Code Style: Prefer Named Functions

Define context-dependent function body builders as **separate named functions** rather than inline lambdas within `ONNX_OPERATOR_SET_SCHEMA`. The macro expansion makes setting breakpoints on inline lambdas unreliable in debuggers.

Simple string-based `.FunctionBody(R"ONNX(...)ONNX")` definitions don't have this issue.

## After Making Changes

```bash
python onnx/defs/gen_doc.py
lintrunner -a --output oneline
```

## Common Mistakes

- Forgetting `schema.BuildFunction(functionProto)` at end of context-dependent builders
- Forgetting to `return true` from the builder function
- Variable names conflicting with input/output names
- Using `Cast` instead of `CastLike` for dynamic type matching
- Function body not producing all declared outputs
- Using `@attr_name` for an attribute not declared in `.Attr()` calls
