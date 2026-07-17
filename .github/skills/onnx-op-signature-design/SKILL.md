---
name: onnx-op-signature-design
description: How to choose inputs/attributes when designing a new ONNX operator, especially decomposable/function-ops that add no expressive power. Use when deciding between a single packed input vs multiple inputs, or whether to encode layout/variants as attributes vs graph structure.
---

# Designing ONNX operator signatures

Guidance for choosing an op's inputs, outputs, and attributes — most relevant for
function-ops and other ops that are just a named decomposition of existing ops.

## Principles

1. **Mirror authored structure; leave packing to the backend.** ONNX is the
   export/interchange layer. A standard op should match the structure producers
   naturally hold at the graph boundary (e.g. two separate projection outputs),
   not a memory-packed form. Fusing a fixed-topology subgraph into one packed
   kernel is a solved below-graph problem for backends; do not pre-bake that
   packing into the standard op.

2. **When an op adds zero expressive power, pick the form hardest to get silently
   wrong.** If the op is a named decomposition of existing ops, its expressiveness
   is fixed — so choose the signature by *error mode*. Prefer designs whose
   mistakes fail loud (shape/type errors at build or inference time) over designs
   with a silent-wrong-answer mode. A hard-coded layout or split convention that
   yields correct *shapes* but garbage *values* is the worst case: nothing flags it.

3. **Express layout and variants as explicit graph ops, not attributes/enums.**
   Use `Split`/`Slice`/`Gather`/`Concat`/`Add`/`Clip` etc. upstream to describe
   data layout and structural variants. An attribute that encodes data layout just
   re-imports the ambiguity a downstream tool must interpret (and every backend
   must agree on); graph edges are self-describing and unambiguous.

4. **Pair multi-input elementwise ops with an explicit equal-shape / no-broadcast
   constraint** when broadcasting is never semantically valid. Enforce it in shape
   inference (fail loud on rank mismatch, conflicting static dims, and size-1
   stretch; merge only symbolic/unknown dims). This turns a wiring mistake into an
   inference error instead of a silently broadcast wrong result. Reuse shared
   infrastructure (e.g. `mergeInShapeInfo`/`mergeInDimensionInfo`) rather than
   hand-rolling dim checks.

5. **Reuse existing ops in the FunctionBody** instead of re-inlining their math, so
   there is a single source of truth and the decomposition stays consistent as the
   reused op evolves.

## Worked example — SwiGLU

An early single-input SwiGLU split one packed tensor into gate/value halves along
an axis. That produces correct output *shapes* regardless of how the producer
packed the tensor — but silently miscomputes any model that interleaves gate/value
instead of using contiguous halves. The layout ambiguity was real enough that ONNX
Runtime shipped a 3-valued `swiglu_fusion` enum and llama.cpp carried a swapped
flag to parametrize it. The redesigned **two-input** form `Y = Swish_alpha(A) * B`
takes gate `A` and value `B` as separate inputs, pushing any packing/splitting into
explicit upstream graph ops, and adds an equal-shape/no-broadcast guard — removing
the silent-miscompute mode entirely.
