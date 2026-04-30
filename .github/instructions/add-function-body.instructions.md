---
applyTo: "onnx/defs/**"
---

# Adding a Function Body Definition for an Operator

A function body defines how an operator can be decomposed into simpler ONNX operators. This enables runtimes that don't natively support the op to still execute it.

## When to Add a Function Body

- The operator can be expressed in terms of other ONNX operators
- You want to provide a reference decomposition for runtimes
- The operator is being proposed as a "function" rather than a primitive

## File Locations

| Component | File |
|-----------|------|
| Function body definition | `onnx/defs/<domain>/defs.cc` (inline with schema) |
| FunctionBuilder utilities | `onnx/defs/function.h` |
| Function tests | `onnx/test/cpp/function_get_test.cc`, `onnx/test/cpp/function_verify_test.cc` |

## Method 1: Simple String-Based Function Body

For ops that decompose the same way regardless of attributes/inputs:

```cpp
ONNX_OPERATOR_SET_SCHEMA(
    LessOrEqual,
    16,
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

When the function body uses ops from a specific opset version:

```cpp
        .FunctionBody(
            R"ONNX(
          {
            Zero = Constant <value = float {0.0}>()
            ZeroCast = CastLike (Zero, X)
            Y = Max (X, ZeroCast)
          }
        )ONNX",
            18)  // Function body valid starting from opset 18
```

### Using Attributes in Function Body

Reference schema attributes with `@attr_name`:

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

For ops whose decomposition varies based on attributes or optional inputs:

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

// In schema registration:
ONNX_OPERATOR_SET_SCHEMA(
    Clip, 13,
    OpSchema()
        // ... inputs, outputs, constraints ...
        .SetContextDependentFunctionBodyBuilder(BuildContextDependentFunctionBodyClip)
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));
```

## FunctionBuilder API

```cpp
FunctionBuilder builder(functionProto);

// Add nodes (ONNX text format)
builder.Add("Y = Relu (X)");

// Add constants
builder.Const("alpha", std::vector<float>{0.01f});
builder.Const1D("axes", int64_t(1));  // 1-D tensor constant

// Add with inline attribute
builder.Add("X_ReduceMax = ReduceMax <keepdims = 1> (input, axes)");

// Add multi-line
builder.Add(R"(
    X_Sub = Sub (input, X_ReduceMax)
    X_Exp = Exp (X_Sub)
    output = Div (X_Exp, X_ReduceSum)
)");

// Add opset dependency
builder.AddOpset("", 18);  // default domain, version 18

// Finalize
schema.BuildFunction(functionProto);
return true;
```

## Multiple Opset Versions

An op can have different function bodies for different opset versions (e.g., when a sub-op's signature changes):

```cpp
    .SetContextDependentFunctionBodyBuilder(builderForOpset13, 13)
    .SetContextDependentFunctionBodyBuilder(builderForOpset18, 18)
```

## FunctionBodyBuildContext API

```cpp
struct FunctionBodyBuildContext {
  const AttributeProto* getAttribute(const std::string& name) const;
  bool hasInput(int inputIndex) const;
  bool hasOutput(int outputIndex) const;
  const TypeProto* getInputType(int inputIndex) const;
};
```

## ONNX Function Body Syntax

The string format uses this grammar:
```
variable = OpName <attr = value> (input1, input2)
```

- Variable names are local to the function (intermediate results)
- Input/output names must match the schema's declared input/output names
- Constants are created with the `Constant` op
- Use `CastLike` to match types dynamically
- Attributes are referenced with `@attr_name`

For the formal grammar, see [docs/Syntax.md](../../docs/Syntax.md). The parser implementation and its tests provide additional examples:

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

## Common Mistakes to Avoid

- Don't forget `schema.BuildFunction(functionProto)` at the end of context-dependent builders
- Don't forget to return `true` from the builder function
- Variable names in the function body must not conflict with input/output names
- Use `CastLike` instead of `Cast` when the target type depends on an input
- The function body must produce all declared outputs
- When using `@attr_name`, the attribute must be declared in the schema's `.Attr()` calls

## Code Style: Prefer Named Functions

Define context-dependent function body builders as **separate named functions** rather than inline lambdas within `ONNX_OPERATOR_SET_SCHEMA`. This makes it easier to set debugger breakpoints (the macro expansion makes breakpoints on inline lambdas unreliable).

```cpp
// PREFERRED: named function — easy to debug
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
        // ...
        .SetContextDependentFunctionBodyBuilder(BuildFunctionBodyMyOp));
```

Simple string-based `.FunctionBody(R"ONNX(...)ONNX")` definitions don't have this issue since there's no logic to step through.
