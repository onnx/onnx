<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# ReferenceEvaluator

This is a first attempt to provide an implementation for all operators
defined by onnx. This is a pure python implementation.
Mismatches may remain between the official specification and the implementation here.
In the case of such a mismatch, the official spec overrides this implementation.
The class can use any implementation available in folder
[ops](https://github.com/onnx/onnx/tree/main/onnx/reference/ops).
It covers most of the tests defined in
[onnx/backend/test/case](https://github.com/onnx/onnx/tree/main/onnx/backend/test/case)
and reported on [ONNX Backend Scoreboard](http://onnx.ai/backend-scoreboard/).

The class `ReferenceEvaluator` is used as follows:

```python
import numpy as np
from onnx.reference import ReferenceEvaluator

X = np.array(...)
sess = ReferenceEvaluator("model.onnx")
results = sess.run(None, {"X": X})
print(results[0])
```

In addition to the implementation of every operator, it can be used
to display intermediate results and help debugging a model.

```python

import numpy as np
from onnx.reference import ReferenceEvaluator

X = np.array(...)
sess = ReferenceEvaluator("model.onnx", verbose=1)
results = sess.run(None, {"X": X})
print(results[0])
```
