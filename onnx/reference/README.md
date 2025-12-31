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

## Array API Support

The reference implementation uses the [Array API standard](https://data-apis.org/array-api/latest/API_specification/index.html)
to support multiple array backends including:

- NumPy
- CuPy (GPU arrays)
- JAX
- PyTorch
- MLX (Apple Silicon)

This is achieved through the [array-api-compat](https://github.com/data-apis/array-api-compat) package,
which provides a compatibility layer for different array libraries.

The class `ReferenceEvaluator` is used as follows:

```python
import numpy as np
from onnx.reference import ReferenceEvaluator

X = np.array(...)
sess = ReferenceEvaluator("model.onnx")
results = sess.run(None, {"X": X})
print(results[0])
```

You can also use arrays from other libraries:

```python
import torch
from onnx.reference import ReferenceEvaluator

# Using PyTorch tensors
X = torch.array(...)
sess = ReferenceEvaluator("model.onnx")
results = sess.run(None, {"X": X})
print(results[0])  # Output will be a PyTorch tensor
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
