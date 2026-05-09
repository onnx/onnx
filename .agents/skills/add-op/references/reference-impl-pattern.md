# Reference Implementation Pattern

Example at `onnx/reference/ops/op_<name>.py`:

```python
from __future__ import annotations

import numpy as np

from onnx.reference.ops._op import OpRunUnaryNum


class OpName(OpRunUnaryNum):
    def _run(self, x):
        return (np.some_operation(x),)
```

## Base Classes

| Base class | Use case |
|-----------|----------|
| `OpRun` | General-purpose (any signature) |
| `OpRunUnary` | Single input, single output |
| `OpRunUnaryNum` | Single numeric input, validates output dtype matches |
| `OpRunBinary` | Two inputs |
| `OpRunBinaryNumpy` | Two inputs using a numpy function directly |

## Binary Op Example

```python
from __future__ import annotations

import numpy as np

from onnx.reference.ops._op import OpRunBinaryNumpy


class Add(OpRunBinaryNumpy):
    def __init__(self, onnx_node, run_params):
        OpRunBinaryNumpy.__init__(self, np.add, onnx_node, run_params)
```

## General Op Example

```python
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun


class Concat(OpRun):
    def _run(self, *inputs, axis=None):
        return (np.concatenate(inputs, axis=axis),)
```
