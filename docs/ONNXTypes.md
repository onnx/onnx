<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# ONNX Types

## Opaque Type

An Opaque type (`TypeProto.Opaque`) enables the definition of user-defined
types, beyond the built-in kinds (tensors, sequences, maps, optionals, and
sparse tensors) that ONNX defines directly in its proto schema. It is
identified by a `(domain, name)` pair, analogous to how a custom op is
identified by a `(domain, op_type)` pair: the meaning of an Opaque type is
defined by, and only needs to be understood by, the producer/consumer of
the custom-domain ops that use it. As with all ONNX types (including
Tensor), the ONNX spec does not define how a value of an Opaque type is
represented internally by a backend -- that is entirely up to the
implementation. ONNX itself just treats an Opaque-typed value as an opaque
piece of data (identified solely by its `domain` and `name`) that gets
passed between nodes.

### Use-cases

Opaque types let a custom domain introduce new kinds of values -- along
with custom ops that produce/consume them -- that are only meaningful to
the ops of that domain, without requiring any change to the ONNX spec
itself. This is useful, for example, to represent a stateful *handle*
(e.g., a file handle, a database connection, or a random-number
generator) that is created by one custom op and consumed by others. More
generally, the Opaque type also gives the ONNX standard itself a way to
introduce new built-in types in the future without needing to change the
`TypeProto` schema.

### Example: a stateful random-number generator (RNG)

The example below illustrates using an Opaque type to represent a stateful
random-number generator (RNG). It uses two illustrative custom ops (in a
custom domain `test.rng`, not part of the ONNX spec):

* `CreateRNG(seed) -> rng` creates a new RNG (of Opaque type
  `test.rng.RNG`) from an integer seed.
* `RandomTensor(rng) -> Y, rng_out` uses the given RNG to generate a
  tensor `Y` of a requested shape (with values drawn, say, from a standard
  normal distribution), and also returns an updated RNG `rng_out`.

Note that since ONNX ops are (side-effect free) functions, `RandomTensor`
cannot simply mutate its input RNG in place to reflect the fact that
generating a random value conceptually advances the RNG's internal state.
Instead, that state update is made explicit: the op returns a new/updated
RNG as an additional output, alongside the generated tensor. A caller that
wants to draw a sequence of random tensors would thread the RNG through a
sequence of calls to `RandomTensor`, using the `rng_out` from one call as
the `rng` input to the next.

This example is deliberately simple: it does not implement an actual RNG
algorithm, nor does it pin down all the details (such as the precise
semantics of the state update) that a real-world stateful-RNG design would
need to address. Its purpose is just to illustrate how an Opaque type can
be declared, produced, consumed, and type/shape-inferred.

An Opaque type can be written explicitly in ONNX's text format (see
[Syntax.md](Syntax.md)) using the syntax `opaque(domain, name)` (or
`opaque(name)` when no domain is needed, or plain `opaque()` when neither
is specified). A model using the `CreateRNG` and `RandomTensor` ops above,
expressed using ONNX's text format, looks like this:

```
<
    ir_version: 10,
    opset_import: ["": 21, "test.rng": 1]
>
agraph (int64 seed) => (float[2,3] Y, opaque(test.rng, RNG) rng2)
{
    rng = test.rng.CreateRNG (seed)
    Y, rng2 = test.rng.RandomTensor <shape = [2, 3]> (rng)
}
```

Here, `rng2` (the second graph output, produced by `RandomTensor`) is
explicitly declared with the Opaque type `test.rng.RNG` using the
`opaque(test.rng, RNG)` syntax. The intermediate value `rng` (produced by
`CreateRNG`) is left untyped in the source text above; running shape
inference on the parsed model determines (and fills in) its type, based on
the type/shape-inference function registered for the `CreateRNG` op
schema -- intermediate and output values may always be left untyped in
this way and have their types filled in by shape inference. See
`tests/python/opaque_type_test.py` for a complete, runnable version of
this example (including the schema and type/shape-inference-function
definitions for `CreateRNG` and `RandomTensor`), which also checks that
the resulting model passes both `onnx.checker.check_model` and
`onnx.shape_inference.infer_shapes`.

## Optional Type

An optional type represents a reference to either an element (could be Tensor, Sequence, Map, or Sparse Tensor) or a null value. The optional type appears in model inputs, outputs, as well as intermediate values.

### Use-cases

Optional type enables users to represent more dynamic typing scenarios in ONNX. Similar to Optional[X] type hint in Python typing which is equivalent to Union[None, X], Optional types in ONNX may reference a single element, or null.

### Examples in PyTorch

Optional type only appears in TorchScript graphs generated by jit script compiler. Scripting a model captures dynamic types where an optional value can be assigned either None or a value.

- Example 1

        class Model(torch.nn.Module):
            def forward(self, x, y:Optional[Tensor]=None):
                if y is not None:
                    return x + y
                return x

    Corresponding TorchScript graph:

        Graph(
            %self : __torch__.Model,
            %x.1 : Tensor,
            %y.1 : Tensor?
        ):
            %11 : int = prim::Constant[value=1]()
            %4 : None = prim::Constant()
            %5 : bool = aten::__isnot__(%y.1, %4)
            %6 : Tensor = prim::If(%5)
                block0():
                    %y.4 : Tensor = prim::unchecked_cast(%y.1)
                    %12 : Tensor = aten::add(%x.1, %y.4, %11)
                -> (%12)
                block1():
                -> (%x.1)
            return (%6)

    ONNX graph:

        Graph(
            %x.1 : Float(2, 3),
            %y.1 : Float(2, 3)
        ):
            %2 : Bool(1) = onnx::OptionalHasElement(%y.1)
            %5 : Float(2, 3) = onnx::If(%2)
                block0():
                    %3 : Float(2, 3) = onnx::OptionalGetElement(%y.1)
                    %4 : Float(2, 3) = onnx::Add(%x.1, %3)
                -> (%4)
                block1():
                    %x.2 : Float(2, 3) = onnx::Identity(%x.1)
                -> (%x.2)
            return (%5)

- Example 2

        class Model(torch.nn.Module):
            def forward(
                    self,
                    src_tokens,
                    return_all_hiddens=torch.tensor([False]),
            ):
                encoder_states: Optional[Tensor] = None
                if return_all_hiddens:
                    encoder_states = src_tokens

                return src_tokens, encoder_states

    Corresponding TorchScript graph:

        Graph(
            %src_tokens.1 : Float(3, 2, 4,),
            %return_all_hiddens.1 : Bool(1)
        ):
            %3 : None = prim::Constant()
            %encoder_states : Tensor? = prim::If(%return_all_hiddens.1)
                block0():
                -> (%src_tokens.1)
                block1():
                -> (%3)
            return (%src_tokens.1, %encoder_states)

    ONNX graph:

        Graph(
            %src_tokens.1 : Float(3, 2, 4),
            %return_all_hiddens.1 : Bool(1)
        ):
            %2 : Float(3, 2, 4) = onnx::Optional[type=tensor(float)]()
            %3 : Float(3, 2, 4) = onnx::If(%return_all_hiddens.1)
                block0():
                -> (%src_tokens.1)
                block1():
                -> (%2)
            return (%3)
