# Adding a Function Body Definition for an Operator

A function body defines how an operator decomposes into simpler ONNX operators, enabling runtimes to execute the op even without native support.

## File Locations

| Component | File |
|-----------|------|
| Function body definition | `onnx/defs/<domain>/defs.cc` (inline with schema) |
| FunctionBuilder utilities | `onnx/defs/function.h` |
| Function tests | `onnx/test/cpp/function_get_test.cc`, `onnx/test/cpp/function_verify_test.cc` |

## Method 1: Simple String-Based Function Body

For ops whose decomposition is independent of attributes/inputs:

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

### With Explicit Opset Version

```cpp
        .FunctionBody(R"ONNX(
          {
            Zero = Constant <value = float {0.0}>()
            ZeroCast = CastLike (Zero, X)
            Y = Max (X, ZeroCast)
          }
        )ONNX", 18)  // Valid from opset 18
```

### Referencing Attributes with `@attr_name`

```cpp
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
        )ONNX")
```

## Method 2: Context-Dependent Function Body

For ops whose decomposition depends on attributes or optional inputs:

```cpp
static bool BuildContextDependentFunctionBodyClip(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  bool has_min = ctx.hasInput(1);
  bool has_max = ctx.hasInput(2);

  FunctionBuilder builder(functionProto);
  if (has_min && has_max) {
    builder.Add("input_less_than_min = Less (input, min)");
    builder.Add("tmp = Where (input_less_than_min, min, input)");
    builder.Add("output_large_than_max = Less (max, tmp)");
    builder.Add("output = Where (output_large_than_max, max, tmp)");
  } else if (has_min) {
    builder.Add("input_less_than_min = Less (input, min)");
    builder.Add("output = Where (input_less_than_min, min, input)");
  } else {
    builder.Add("output = Identity (input)");
  }

  schema.BuildFunction(functionProto);
  return true;
}

// Register:
    .SetContextDependentFunctionBodyBuilder(BuildContextDependentFunctionBodyClip)
```

## FunctionBuilder API

```cpp
FunctionBuilder builder(functionProto);
builder.Add("Y = Relu (X)");                          // Add node
builder.Const("alpha", std::vector<float>{0.01f});    // Constant tensor
builder.Const1D("axes", int64_t(1));                  // 1-D constant
builder.Add("X_Max = ReduceMax <keepdims = 1> (X, axes)");  // Inline attr
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
    .SetContextDependentFunctionBodyBuilder(builderForOpset13, 13)
    .SetContextDependentFunctionBodyBuilder(builderForOpset18, 18)
```

## FunctionBodyBuildContext API

```cpp
ctx.getAttribute("name")    // Returns const AttributeProto* or nullptr
ctx.hasInput(index)         // Is optional input present?
ctx.hasOutput(index)        // Is optional output present?
ctx.getInputType(index)     // Returns const TypeProto*
```

## ONNX Function Body Syntax

```
variable = OpName <attr = value> (input1, input2)
```

- Variable names are local intermediates
- Input/output names must match schema declarations
- Use `CastLike` (not `Cast`) when target type depends on input
- Reference attributes with `@attr_name`

For the formal grammar, see `docs/Syntax.md`. The parser implementation and its tests provide additional examples:

| Resource | File |
|----------|------|
| Formal syntax specification | `docs/Syntax.md` |
| C++ parser implementation | `onnx/defs/parser.h`, `onnx/defs/parser.cc` |
| Python parser | `onnx/parser.py` |
| C++ parser tests | `onnx/test/cpp/parser_test.cc` |
| Python parser tests | `onnx/test/parser_test.py` |

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

## Code Style: Prefer Named Functions

Define context-dependent function body builders as **separate named functions** rather than inline lambdas within `ONNX_OPERATOR_SET_SCHEMA`. This makes it easier to set debugger breakpoints (the macro expansion makes breakpoints on inline lambdas unreliable).

```cpp
// PREFERRED: named function
static bool BuildFunctionBodyMyOp(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  FunctionBuilder builder(functionProto);
  // ...
  schema.BuildFunction(functionProto);
  return true;
}

ONNX_OPERATOR_SET_SCHEMA(
    MyOp, 21,
    OpSchema()
        .SetContextDependentFunctionBodyBuilder(BuildFunctionBodyMyOp));
```

Simple string-based `.FunctionBody(R"ONNX(...)ONNX")` definitions don't have this issue since there's no logic to step through.
