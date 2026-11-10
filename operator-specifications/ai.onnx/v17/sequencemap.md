```toml
name = "SequenceMap"
domain = "ai.onnx"
# Setting `NonDeterministic` since it needs to be evaluated after function inlining
node_determinism = "non_deterministic"

[attributes.body]
type = "GRAPH"
desc = "The graph to be run for each sample in the sequence(s). It should have as many inputs and outputs as inputs and outputs to the SequenceMap function."
required = true

[[inputs]]
name = "input_sequence"
type = "S"
desc = "Input sequence."

[[inputs]]
name = "additional_inputs"
type = "V"
desc = "Additional inputs to the graph"
option = "variadic"
homogeneous = false
min_arity = 0

[[outputs]]
name = "out_sequence"
type = "S"
desc = "Output sequence(s)"
option = "variadic"
homogeneous = false

[type_constraints.S]
types = [
    "seq(tensor(uint8))",
    "seq(tensor(uint16))",
    "seq(tensor(uint32))",
    "seq(tensor(uint64))",
    "seq(tensor(int8))",
    "seq(tensor(int16))",
    "seq(tensor(int32))",
    "seq(tensor(int64))",
    "seq(tensor(float16))",
    "seq(tensor(float))",
    "seq(tensor(double))",
    "seq(tensor(string))",
    "seq(tensor(bool))",
    "seq(tensor(complex64))",
    "seq(tensor(complex128))",
]
desc = "Constrain input types to any sequence type."

[type_constraints.V]
types = [
    "tensor(uint8)",
    "tensor(uint16)",
    "tensor(uint32)",
    "tensor(uint64)",
    "tensor(int8)",
    "tensor(int16)",
    "tensor(int32)",
    "tensor(int64)",
    "tensor(float16)",
    "tensor(float)",
    "tensor(double)",
    "tensor(string)",
    "tensor(bool)",
    "tensor(complex64)",
    "tensor(complex128)",
    "seq(tensor(uint8))",
    "seq(tensor(uint16))",
    "seq(tensor(uint32))",
    "seq(tensor(uint64))",
    "seq(tensor(int8))",
    "seq(tensor(int16))",
    "seq(tensor(int32))",
    "seq(tensor(int64))",
    "seq(tensor(float16))",
    "seq(tensor(float))",
    "seq(tensor(double))",
    "seq(tensor(string))",
    "seq(tensor(bool))",
    "seq(tensor(complex64))",
    "seq(tensor(complex128))",
]
desc = "Constrain to any tensor or sequence type."
```

Applies a sub-graph to each sample in the input sequence(s).

Inputs can be either tensors or sequences, with the exception of the first input which must
be a sequence. The length of the first input sequence will determine the number of samples in the
outputs. Any other sequence inputs should have the same number of samples. The number of inputs
and outputs, should match the one of the subgraph.

For each i-th element in the output, a sample will be extracted from the input sequence(s) at
the i-th position and the sub-graph will be applied to it.
The outputs will contain the outputs of the sub-graph for each sample, in the same order as in
the input.

This operator assumes that processing each sample is independent and could executed in parallel
or in any order. Users cannot expect any specific ordering in which each subgraph is computed.
