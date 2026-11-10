```toml
name = "Einsum"
domain = "ai.onnx"

[attributes.equation]
type = "STRING"
desc = "Einsum expression string."
required = true

[[inputs]]
name = "Inputs"
type = "T"
desc = "Operands"
option = "variadic"
differentiation = "differentiable"

[[outputs]]
name = "Output"
type = "T"
desc = "Output tensor"
differentiation = "differentiable"

[type_constraints.T]
types = "all_numeric_types"
desc = "Constrain input and output types to all numerical tensor types."
```

An einsum of the form `term1, term2 -> output-term` produces an output tensor using the following equation

```
output[output-term] = reduce-sum( input1[term1] * input2[term2] )
```

where the reduce-sum performs a summation over all the indices occurring in the input terms (term1, term2)
that do not occur in the output-term.

The Einsum operator evaluates algebraic tensor operations on a sequence of tensors, using the Einstein summation
convention. The equation string contains a comma-separated sequence of lower case letters and/or upper case letters.
Each term corresponds to an operand tensor, and the characters within the terms correspond to operands dimensions.
Lower case letters and upper case letters are treated as distinct symbols, that is, "a" and "A" refer to different
symbols.

This sequence may be followed by "->" to separate the left and right hand side of the equation.
If the equation contains "->" followed by the right-hand side, the explicit (not classical) form of the Einstein
summation is performed, and the right-hand side indices indicate output tensor dimensions. In other cases,
output indices are (implicitly) set to the sequence of indices appearing exactly once in the equation, sorted in
increasing order of their ASCII values (so that all upper case letters precede all lower case letters, e.g.,
"A" < "Z" < "a" < "z").

When a dimension character is repeated in the left-hand side, it represents summation along the dimension.

The equation may contain ellipsis ("...") to enable broadcasting. Ellipsis must indicate a fixed number of dimensions.
Specifically, every occurrence of ellipsis in the equation must represent the same number of dimensions.
The right-hand side may contain exactly one ellipsis. In implicit mode, the ellipsis dimensions are set to the
beginning of the output. The equation string may contain space (U+0020) character.
