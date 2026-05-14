---
applyTo: "onnx/defs/**"
---

# Adding or Updating an ONNX Operator

Follow the full procedure in [docs/AddNewOp.md](../../docs/AddNewOp.md).

## File Locations

| Component | File |
|-----------|------|
| Schema definition | `onnx/defs/<domain>/defs.cc` |
| Previous version (on update) | `onnx/defs/<domain>/old.cc` |
| Shape/type inference | Inline in schema via `.TypeAndShapeInferenceFunction(...)` |
| Reference implementation | `onnx/reference/ops/op_<lowercase_name>.py` |
| Node tests | `onnx/backend/test/case/node/<lowercase_name>.py` |
| Shape inference tests | `onnx/test/shape_inference_test.py` |
| Version converter adapter | `onnx/version_converter/adapters/<name>_<from>_<to>.h` |
| Upgrade/downgrade tests | `onnx/test/version_converter/automatic_upgrade_test.py` and `automatic_downgrade_test.py` |

## Domains

Operators are organized by domain subdirectory under `onnx/defs/`:
`math/`, `nn/`, `tensor/`, `logical/`, `reduction/`, `rnn/`, `sequence/`, `image/`, `text/`, `quantization/`, `controlflow/`, `optional/`, `traditionalml/`, `training/`

## Schema Registration Pattern

```cpp
ONNX_OPERATOR_SET_SCHEMA(
    OperatorName,
    OPSET_VERSION,
    OpSchema()
        .SetDoc(OperatorName_verN_doc)
        .Input(0, "X", "Description", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(0, "Y", "Description", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Attr("attr_name", "Description", AttributeProto::FLOAT, default_value)
        .TypeConstraint("T", {"tensor(float)", "tensor(double)", ...}, "Description")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput)
        .FunctionBody(R"ONNX(
          {
            ...
          }
        )ONNX", FUNCTION_OPSET_VERSION));
```

## Reference Implementation Pattern

```python
# onnx/reference/ops/op_<name>.py
from __future__ import annotations

import numpy as np

from onnx.reference.ops._op import OpRunUnaryNum  # or OpRunBinaryNumpy, OpRun


class OpName(OpRunUnaryNum):
    def _run(self, x):
        return (np.some_operation(x),)
```

## Node Test Pattern

```python
# onnx/backend/test/case/node/<name>.py
from __future__ import annotations

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class OpName(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "OpName",
            inputs=["x"],
            outputs=["y"],
        )
        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.some_operation(x)
        expect(node, inputs=[x], outputs=[y], name="test_opname")
```

### Body subgraphs: prefer `onnx.parser.parse_graph`

When the op takes a graph attribute (`Scan`, `Loop`, `If`, `ScanVarLen`, …), keep the outer `helper.make_node` + `expect(...)` call as shown above (it drives the test-data generation pipeline), but build the body subgraph with `onnx.parser.parse_graph` instead of a chain of `helper.make_node` / `helper.make_tensor_value_info` / `helper.make_graph` calls. The text format is dramatically more compact:

```python
body = onnx.parser.parse_graph("""
    b (float[1] s, float[1] xi) => (float[1] so, float[1] xo) {
        so = Identity(s)
        xo = Identity(xi)
    }
""")
node = onnx.helper.make_node("Scan", ["s", "x"], ["so", "xo"], body=body, num_scan_inputs=1)
```

## Test Authoring with `onnx.parser`

For Python tests outside the node-test harness — primarily `onnx/test/shape_inference_test.py` and `onnx/test/reference_evaluator_test.py` — prefer `onnx.parser.parse_model` over chains of `helper.make_*` calls. The ONNX text format is far more compact and the resulting test fixture reads as one self-contained block.

```python
# Before: helper.make_* chain
node = helper.make_node("Add", ["x", "y"], ["z"])
graph = helper.make_graph(
    [node], "g",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None]),
     helper.make_tensor_value_info("y", TensorProto.FLOAT, [None])],
    [helper.make_tensor_value_info("z", TensorProto.FLOAT, [None])])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

# After: text fixture
model = onnx.parser.parse_model("""
    <ir_version: 8, opset_import: ["" : 18]>
    g (float[N] x, float[N] y) => (float[N] z) { z = Add(x, y) }
""")
```

For C++ tests in `onnx/test/cpp/shape_inference_test.cc`, prefer the C++ `OnnxParser` (`onnx/defs/parser.h`): express the whole model as text, parse with `OnnxParser::Parse(model_proto, text)`, then run `shape_inference::InferShapes`. This replaces hand-building `NodeProto` / `GraphProto` and an `InferenceContextImpl` with a few lines.

Body-graph attributes (Scan / Loop / If) embed inline inside the `<...>` attribute list:

```
so, xo = Scan(s, x) <num_scan_inputs=1, body = b (float[1] si, float[1] xi) => (float[1] so, float[1] xo) { so = Identity(si) xo = Identity(xi) }>
```

**Argument order**: for ops with simple scalar/tensor attributes, the conventional `Op<attrs>(inputs)` form reads well — e.g. `Y = Transpose<perm=[2,0,1]>(X)`. For ops with subgraph attributes (`Scan`, `Loop`, `If`, `ScanVarLen`, …) the body spans multiple lines, and putting it after the inputs keeps them visible — prefer `Op(inputs)<body = ... { ... }>` as shown in the Scan example above. PR #7962 (ScanVarLen) settled on this convention.

**Exception**: helper-driven harnesses such as `onnx/test/version_converter/automatic_upgrade_test.py` and `automatic_downgrade_test.py` (the `_test_op_upgrade` / `_test_op_downgrade` convention) should keep their established style — do not rewrite them with the parser.

Concrete reference: PR #7962 (ScanVarLen) rewrote shape-inference, reference-evaluator, and node-test fixtures using this approach and reduced LOC by ~58–70% across the three files.

## After Making Changes

```bash
# Regenerate documentation and test data (required)
python onnx/defs/gen_doc.py
python onnx/backend/test/stat_coverage.py

# Only if proto was changed
python onnx/gen_proto.py

# Lint
lintrunner -a --output oneline
```

## Updating an Existing Operator (New Opset Version)

1. Move the current schema from `defs.cc` to `old.cc` in the same domain directory
2. Create the new version in `defs.cc` with the new opset version number
3. Add a version converter adapter if behavior changed
4. Add upgrade/downgrade tests using `_test_op_upgrade` / `_test_op_downgrade`
5. Update the operator set registration in `onnx/defs/operator_sets.h`

### Avoiding duplication between defs.cc and old.cc

When moving a schema to `old.cc`, avoid significant code/documentation duplication between old and new versions:

- **Shared utilities**: Extract common logic (doc strings, type constraint lists, shape inference helpers) into shared functions in the domain's header or utils file.
- **Parameterized functions**: When versions differ only slightly (e.g., expanded type list, additional optional input), use parameterized helpers that accept the differences as arguments.
- **Use judgment**: Some duplication is acceptable when sharing would create overly complicated logic. Prefer clarity over DRY when the alternative makes either version harder to understand independently.

## Common Mistakes to Avoid

- Don't edit generated files (`docs/Operators.md`, `docs/Changelog.md`, `*_pb2.py`) directly
- Don't forget to add the operator to `operator_sets.h` registration
- Don't forget DCO sign-off on commits (`git commit -s`)
- Don't forget `from __future__ import annotations` in new Python files
- Don't forget copyright headers on all new files
- When updating an op, always move old schema to `old.cc` first
- Always add upgrade/downgrade tests even for new ops (future-proofing)

## Code Style: Prefer Named Functions

Define shape inference functions and function body builders as **separate named functions** rather than inline lambdas within the `ONNX_OPERATOR_SET_SCHEMA` macro. This makes it easier to set debugger breakpoints (the macro expansion makes breakpoints on inline lambdas unreliable) and improves readability.

```cpp
// PREFERRED: named function
static void InferShapeForMyOp(InferenceContext& ctx) {
    propagateElemTypeFromInputToOutput(ctx, 0, 0);
    // ...
}

ONNX_OPERATOR_SET_SCHEMA(
    MyOp, 21,
    OpSchema()
        // ...
        .TypeAndShapeInferenceFunction(InferShapeForMyOp));
```

```cpp
// AVOID: inline lambda (harder to debug)
ONNX_OPERATOR_SET_SCHEMA(
    MyOp, 21,
    OpSchema()
        // ...
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
            // breakpoints here are unreliable due to macro expansion
        }));
```

The same applies to context-dependent function body builders — define them as named `static bool BuildFunctionBodyMyOp(...)` functions.
