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
| Upgrade/downgrade tests | `onnx/test/version_converter/automatic_upgrade_test.py` |

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

## Common Mistakes to Avoid

- Don't edit generated files (`docs/Operators.md`, `docs/Changelog.md`, `*_pb2.py`) directly
- Don't forget to add the operator to `operator_sets.h` registration
- Don't forget DCO sign-off on commits (`git commit -s`)
- Don't forget `from __future__ import annotations` in new Python files
- Don't forget copyright headers on all new files
- When updating an op, always move old schema to `old.cc` first
- Always add upgrade/downgrade tests even for new ops (future-proofing)
