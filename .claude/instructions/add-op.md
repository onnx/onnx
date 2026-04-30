# Adding or Updating an ONNX Operator

Follow the full procedure in `docs/AddNewOp.md`.

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
| Operator set registration | `onnx/defs/operator_sets.h` |

## Domains

Operators live under `onnx/defs/` in domain subdirectories:
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

## After Making Changes

```bash
python onnx/defs/gen_doc.py
python onnx/backend/test/stat_coverage.py
python onnx/gen_proto.py  # only if proto changed
lintrunner -a --output oneline
```

## Updating an Existing Operator

1. Move current schema from `defs.cc` to `old.cc` in the same domain directory
2. Create new version in `defs.cc` with the new opset version number
3. Update `onnx/defs/operator_sets.h`
4. Add version converter adapter if behavior changed
5. Add upgrade/downgrade tests

### Avoiding duplication between defs.cc and old.cc

When moving a schema to `old.cc`, avoid significant code/documentation duplication:

- Extract common logic (doc strings, type constraint lists, inference helpers) into shared functions in the domain's header or utils file.
- Use parameterized helpers when versions differ only slightly (e.g., expanded type list, additional optional input).
- Some duplication is acceptable when sharing would create overly complicated logic. Prefer clarity over DRY when the alternative makes either version harder to understand independently.

## Common Mistakes

- Don't edit generated files (`docs/Operators.md`, `docs/Changelog.md`, `*_pb2.py`)
- Don't forget to register the operator in `operator_sets.h`
- Don't forget DCO sign-off (`git commit -s`)
- Don't forget `from __future__ import annotations` in new Python files
- Don't forget copyright headers
- When updating, always move old schema to `old.cc` first

## Code Style: Prefer Named Functions

Define shape inference functions and function body builders as **separate named functions** rather than inline lambdas within `ONNX_OPERATOR_SET_SCHEMA`. This makes it easier to set debugger breakpoints (the macro expansion makes breakpoints on inline lambdas unreliable) and improves readability.

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

The same applies to context-dependent function body builders — define them as named `static bool BuildFunctionBodyMyOp(...)` functions.
