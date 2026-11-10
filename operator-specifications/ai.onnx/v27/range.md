```toml
name = "Range"
domain = "ai.onnx"
function_bodies = {}

[attributes.stash_type]
type = "INT"
desc = "The data type used for intermediate computation when T is float16 or bfloat16. Defaults to 1 (float). Has no effect for other types."
default = 1

[type_constraints.T]
types = [
    "tensor(float)",
    "tensor(double)",
    "tensor(int16)",
    "tensor(int32)",
    "tensor(int64)",
    "tensor(float16)",
    "tensor(bfloat16)",
]
desc = "Constrain input types to common numeric type tensors."
```

Generate a tensor containing a sequence of numbers that begin at `start` and extends by increments of `delta`
up to `limit` (exclusive).

The number of elements in the output of range is computed as below:
```
number_of_elements = max( ceil( (limit - start) / delta ) , 0 )
```
The pseudocode determining the contents of the output is shown below:
```
for(int i=0; i<number_of_elements; ++i) {
  output[i] =  start + (i * delta);
}
```
Example 1:
```
Inputs: start = 3, limit = 9, delta = 3
Output: [3, 6]
```
Example 2:
```
Inputs: start = 10, limit = 4, delta = -2
Output: [10, 8, 6]
```

For `float16` and `bfloat16` inputs, the `stash_type` attribute controls the precision used for
intermediate accumulation. Setting `stash_type` to `1` (float) causes `start`, `limit`, and
`delta` to be cast to 32-bit float before the loop, with the output cast back to the original
type. This avoids precision loss for large ranges where successive additions in float16 or
bfloat16 would otherwise be inexact (e.g. `x + 1 == x` for large `x`).
