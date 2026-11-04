# Node Test Pattern

Example test file at `onnx/backend/test/case/node/<name>.py`:

```python
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

    @staticmethod
    def export_with_broadcasting() -> None:
        node = onnx.helper.make_node(
            "OpName",
            inputs=["x", "y"],
            outputs=["z"],
        )
        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(5).astype(np.float32)
        expect(node, inputs=[x, y], outputs=[x + y], name="test_opname_bcast")
```

Each `export*` static method becomes a separate test case. The `expect` helper validates the reference implementation output matches.
