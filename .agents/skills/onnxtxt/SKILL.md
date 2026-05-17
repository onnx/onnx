---
name: onnxtxt
description: Read or write ONNX text format ("onnxtxt"). Use when authoring `.FunctionBody(R"ONNX(...)")` blocks, writing tests with `onnx.parser.parse_model` / `parse_graph`, using the C++ `OnnxParser`, debugging parser errors, or interpreting `Constant <value = ...>` and body-subgraph syntax.
---

ONNX has a compact text format implemented by `onnx/parser.py` (Python) and `onnx/defs/parser.{h,cc}` (C++). The formal grammar lives in [docs/Syntax.md](../../../docs/Syntax.md); this skill captures the practical conventions, idioms, and gotchas that matter when authoring or reviewing code that uses it.

## Where the format appears

| Surface | API |
|---|---|
| C++ function bodies | `.FunctionBody(R"ONNX( ... )ONNX")` and `FunctionBuilder::Add(...)` |
| Python test fixtures | `onnx.parser.parse_model("...")`, `onnx.parser.parse_graph("...")` |
| C++ tests | `OnnxParser` in `onnx/defs/parser.h` — parse a model, then call `shape_inference::InferShapes` |

## Core syntax

```
<var> = <OpName> <attr1 = value, attr2 = value> (<input1>, <input2>)
```

- Variables on the LHS are local to the surrounding graph (function body, subgraph, or model main graph).
- Inputs/outputs of a function body must match the names declared in the schema's `.Input(...)` / `.Output(...)`.
- Constants: `Const = Constant <value = float {0.0}>()` or `Alpha = Constant <value_float: float = @alpha>()`.
- Use `CastLike` (not `Cast`) when the target dtype depends on another input.
- Reference enclosing-op attributes with `@attr_name` (only inside function bodies, and only for attributes declared on the schema).

## Argument-order convention

**Simple scalar/tensor attributes** — keep the conventional `Op<attrs>(inputs)` form; reads well on one line:

```
Y = Transpose<perm = [2, 0, 1]>(X)
```

**Subgraph attributes** (Scan, Loop, If, ScanVarLen, …) — prefer `Op(inputs)<body = ...>`. The body spans multiple lines, so putting inputs first keeps the call site readable:

```
so, xo = Scan (s, x) <
    num_scan_inputs = 1,
    body = scan_body (float[1] s_in, float[1] x_in) => (float[1] s_out, float[1] x_out) {
        s_out = Add(s_in, x_in)
        x_out = Identity(x_in)
    }
>
```

## Testing idioms

| Test file | Recommendation |
|---|---|
| `onnx/test/shape_inference_test.py`, `onnx/test/reference_evaluator_test.py` | Use `onnx.parser.parse_model(...)` for one-off fixtures. |
| `onnx/backend/test/case/node/<op>.py` | Keep the outer `helper.make_node` + `expect(...)` (it drives data generation). For body-subgraph ops, build the body with `onnx.parser.parse_graph`. |
| `onnx/test/cpp/shape_inference_test.cc` | Use `OnnxParser` (`onnx/defs/parser.h`); pair with `shape_inference::InferShapes`. |
| `onnx/test/version_converter/automatic_upgrade_test.py` and similar harnesses | Keep the established `_test_op_upgrade` / `_test_op_downgrade` style — do not rewrite. |

Empirical: PR #7962 (ScanVarLen) cut ~58–70% of test LOC by switching to parser-based fixtures.

### Python example — shape inference fixture

```python
import onnx
import onnx.parser
import onnx.shape_inference

model = onnx.parser.parse_model("""
    <ir_version: 8, opset_import: ["" : 18]>
    g (float[2, 3, 4] X) => (float[4, 2, 3] Y) {
        Y = Transpose<perm = [2, 0, 1]>(X)
    }
""")
inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
```

### Python example — body subgraph for a node test

```python
body = onnx.parser.parse_graph("""
    b (float[1] s, float[1] xi) => (float[1] so, float[1] xo) {
        so = Identity(s)
        xo = Identity(xi)
    }
""")
node = onnx.helper.make_node("Scan", ["s", "x"], ["so", "xo"], body=body, num_scan_inputs=1)
```

### C++ example — shape inference test

```cpp
#include "onnx/defs/parser.h"
#include "onnx/shape_inference/implementation.h"

ModelProto model;
OnnxParser parser(R"ONNX(
    <ir_version: 8, opset_import: ["" : 18]>
    g (float[2, 3, 4] X) => (Y) {
        Y = Transpose<perm = [2, 0, 1]>(X)
    }
)ONNX");
ASSERT_TRUE(parser.Parse(model).IsOK());
shape_inference::InferShapes(model);
```

## Gotchas

- **`unk__*` materialization in C++ shape-inference tests.** Under `InferShapes`, unset output dims are materialized by `MaterializeSymbolicShape` into `dim_param` names like `unk__0`, `unk__1`, … Assertions on free dims must accept either an unset dim or an `unk__*` placeholder — use the `ExpectFreeDim` helper in `onnx/test/cpp/shape_inference_test.cc`.
- **Variable-name collisions.** Local variables in a function body must not reuse declared input/output names of the enclosing op.
- **`@attr_name` scope.** Only valid inside a function body, and only for attributes declared on the enclosing schema's `.Attr(...)` calls.
- **`CastLike` vs `Cast`.** Use `CastLike` when the desired target dtype is determined by another input; `Cast` requires a static `to` attribute.

## References

| Resource | Path |
|---|---|
| Formal grammar | [docs/Syntax.md](../../../docs/Syntax.md) |
| C++ parser | `onnx/defs/parser.h`, `onnx/defs/parser.cc` |
| Python parser | `onnx/parser.py` |
| C++ parser tests | `onnx/test/cpp/parser_test.cc` |
| Python parser tests | `onnx/test/parser_test.py` |
| Empirical LOC win | PR #7962 (ScanVarLen test rewrite) |
