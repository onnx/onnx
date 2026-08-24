<!--
Copyright (c) ONNX Project Contributors
-->

<!--- SPDX-License-Identifier: Apache-2.0 -->
- Feature Name: `symbolic_dimension_algebra`
- Start Date: 2026-08-22
- RFC PR: [onnx/onnx#0000](https://github.com/onnx/onnx/pull/0000)
- Status: under discussion
- Authors:
  - take-cheeze

## Summary
[summary]: #summary

Give the shape-inference reference implementation (`onnx/defs/shape_inference.h`,
`onnx/shape_inference/implementation.cc`) an internal, dependency-free symbolic
algebra for `TensorShapeProto::Dimension`, so that arithmetic combining two
`dim_param`s — `M + N`, `2 * M`, `M * N`, exact division — produces a real,
reproducible symbol instead of an anonymous unknown dimension (`?`). This is
purely a capability upgrade to the *inference-time* engine: the wire format
(`dim_value` / `dim_param` / unset) is unchanged, no opset bump is required,
and every existing model, producer, and consumer keeps working exactly as
today. This proposal is scoped to reference-implementation code only; it does
not add expression syntax to the ONNX IR itself.

## Motivation
[motivation]: #motivation

`docs/ShapeInference.md` documents the exact gap this proposal closes, in its
own words:

> Shape inference works only with constants and simple variables. It does not
> support arithmetic expressions containing variables. For example, `Concat`
> on tensors of shapes `(5, 2)` and `(7, 2)` can be inferred to produce a
> result of shape `(12, 2)`, but `Concat` on tensors of shapes `(5, 2)` and
> `(N, 2)` will simply produce `(M, 2)`... These limitations are a property of
> the current implementation, not fundamental constraints — if you are in
> need of something more advanced, do let us know!

This is not a new observation. Proposal
[0005](0005-SymbolicShapeInfProposal.md) ("Symbolic Shape Inference And
Partial Data Propagation", accepted and shipped in ONNX 1.10) named this exact
capability explicitly and just as explicitly deferred it:

> **Non-goals** ... *Add symbolic expressions to ONNX standard*: This is not
> necessary for accomplishing our goals... the tradeoff is the added
> complexity. So, at this point we are not considering it. This can be
> considered in future iterations.

Five years on, the gap 0005 deferred has not gone away — it has been
independently re-solved, outside `onnx` itself, at least four times:

- `onnxruntime`'s [`symbolic_shape_infer.py`](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/symbolic_shape_infer.py)
  represents every `dim_param` as a `sympy.Symbol` and re-implements ~100
  operator shape-inference rules in Python so it can compute over them.
- [`justinchuby/onnx-shape-inference`](https://github.com/justinchuby/onnx-shape-inference)
  does the same with `sympy` on top of `onnx_ir`, adding anonymous-symbol
  reconciliation against author-declared `dim_param`s.
- `onnxslim` does not implement its own algebra at all; it depends on
  whichever of the above happens to be installed, and is only as good at
  dynamic shapes as that upstream tool.
- This organization's own `onnxsim` shipped a dependency-free C++ polynomial
  engine (`SymExpr`, [onnxsim#527](https://github.com/onnxsim/onnxsim/pull/527))
  specifically because neither `onnx`'s core shape inference nor a `sympy`
  dependency (which needs GMP, impractical under Emscripten/WASM) was usable
  from onnxsim's C++/WASM build. It is now wired into onnxsim's constant
  folder (`_EvalPartialShape` in `onnxsim.cpp`) and resolves shape-scaffolding
  chains that plain ONNX data propagation cannot. The companion RFC
  describing that implementation is at
  [onnxsim/onnxsim#597](https://github.com/onnxsim/onnxsim/issues/597) /
  `docs/symexpr-shape-inference-rfc.md` in that repo.

Every one of these tools is solving the same problem 0005 named five years
ago, against the same underlying gap in `onnx`'s reference implementation,
each maintaining its own correctness and operator coverage independently.
That duplication is itself evidence that this belongs in `onnx` — the
project every one of these tools is layered on top of — rather than being
re-invented per-consumer indefinitely.

### Where this actually bites

`onnx/defs/shape_inference.h` already has overloaded arithmetic on
`TensorShapeProto::Dimension`, e.g.:

```cpp
inline TensorShapeProto::Dimension operator*(
    const TensorShapeProto::Dimension& dim1,
    const TensorShapeProto::Dimension& dim2) {
  TensorShapeProto::Dimension result;
  if (dim1.has_dim_value() && dim2.has_dim_value()) {
    result.set_dim_value(checkedMultiply(dim1.dim_value(), dim2.dim_value()));
  } else if (dim1.has_dim_value() && (dim1.dim_value() == 1)) {
    return dim2;
  } else if (dim2.has_dim_value() && (dim2.dim_value() == 1)) {
    return dim1;
  }
  return result;  // <-- neither dim_value nor dim_param: the symbol is lost
}
```

and similarly `operator+(Dimension, int64_t)` only preserves the symbol when
the constant is `0`. Any op whose shape-inference rule multiplies or adds two
*symbolic* dims — not a symbol against a literal — silently degrades to an
anonymous unknown dimension today. The same thing happens one level up in
`DataPropagationContext`'s partial data propagation (0005's second half):
`Shape -> Gather -> Add -> Concat -> Reshape` folds fine when the `Add` is
between a `dim_param` and a constant, and stalls the instant it is between
two `dim_param`s.

This is exactly the shape every KV-cache transformer decoder export produces
today (Llama/Mistral/Qwen/GPT-style models via `torch.onnx.export(...,
dynamo=True)` or `optimum.exporters.onnx`, which is the standard ONNX export
path for autoregressive LLMs):

```
Shape(past_key)              -> [batch, num_heads, past_len, head_dim]
Gather(..., axis=0, index=2) -> past_len                     (dim_param)
Shape(k_new)                 -> [batch, num_heads, seq_len, head_dim]
Gather(..., axis=0, index=2) -> seq_len                      (dim_param)
Add(past_len, seq_len)       -> total_len                    <-- symbol + symbol
Concat(past_key, k_new, axis=2)  -> present_key   # shape [batch, num_heads, total_len, head_dim]
...
Reshape(attn_out, [batch, seq_len, num_heads * head_dim])    <-- symbol * const, const
```

`Add(past_len, seq_len)` and the `num_heads * head_dim` product are both
combinations of two non-trivial dims, so both fall through the `return
result;` branch above and the anonymous-unknown-dim branch of data
propagation. The `Shape -> Gather -> Add -> Concat` subgraph computing
`total_len`, and the `Shape -> Gather -> Mul -> Concat` subgraph computing
`num_heads * head_dim`, both survive in the graph and both are recomputed at
every inference call, purely because `onnx`'s reference implementation cannot
represent "the sum/product of two symbols" — even though `num_heads` and
`head_dim` are almost always static constants once weights are fixed, and
`total_len` is exactly the quantity a KV-cache-aware runtime needs to know
symbolically for memory planning.

## Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

Nothing changes about how a model is authored, and nothing changes about the
`.onnx` file format. A `dim_param` is still just a string; `dim_value` is
still just an int64. What changes is what `onnx.shape_inference.infer_shapes`
(Python) / `shape_inference::InferShapes` (C++) is able to conclude, and what
`ShapeInferenceOptions{enable_data_propagation=true}` is able to fold.

Extending `docs/ShapeInference.md`'s own example: today,

```python
# x: [M, 2], y: [N, 2]
z = Concat([x, x], axis=0)   # -> [?, 2]   (M + M is currently unrepresentable)
w = Concat([x, y], axis=0)   # -> [?, 2]   (M + N is currently unrepresentable)
```

After this proposal:

```python
z = Concat([x, x], axis=0)   # -> [2*M, 2]   (a fresh symbol; M+M collapses to 2*M)
w = Concat([x, y], axis=0)   # -> [M + N, 2] (a fresh symbol; distinct from 2*M)
```

Concretely, "the shape carries `2*M`" or `M + N` means: shape inference mints
a new symbol via the existing `SymbolTable::createNew()` mechanism (same
mechanism 0005 already introduced for anonymous-dim symbol generation) and
records, internally, that this symbol's value is the polynomial `2*M` /
`M + N`. Two dimensions that reduce to the *same* polynomial are recognized
as equal and get the *same* symbol, even if they arrived through different
paths in the graph — this is what lets `unifyDim` treat `num_heads *
head_dim` computed two different ways in a graph as the same dimension, and
what lets the KV-cache `total_len` in the example above be resolved once
symbolic and then correctly reused for every consumer of `present_key`'s
shape.

For partial data propagation, a `Shape -> Gather -> Add -> Concat -> Reshape`
chain like the KV-cache example now folds all the way through: `Add(past_len,
seq_len)` produces the value `past_len + seq_len` (not "unknown"), so a
downstream `Reshape` whose target is `[batch, past_len + seq_len,
num_heads*head_dim]` can be recognized as having exactly the shape of
`present_key`, and a constant-folding consumer (such as `onnxsim`, or a
runtime's own graph optimizer) can eliminate the now-dead scaffolding that
used to be required to compute it at runtime.

Existing callers who only look at `dim_value`/`dim_param` on the output
model, without opting into anything new, see no behavior change beyond
*more* dims resolving to a concrete value or a stable shared symbol where they
previously saw `?` — a strict improvement on the graph-level guarantee 0005
already established (an unknown dimension may become a new symbol), not a
new category of behavior for existing consumers to handle.

## Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

### Representation

Introduce a `SymbolicExpr` type (working name) in a new header,
`onnx/shape_inference/symbolic_expr.h`, representing an integer-coefficient
polynomial over dimension-symbol names — a sparse map from monomials (sorted
symbol-name multisets) to `int64_t` coefficients:

```cpp
class SymbolicExpr {
 public:
  SymbolicExpr() = default;
  SymbolicExpr(int64_t constant);
  static SymbolicExpr Symbol(const std::string& name);

  bool IsConstant() const;              // no symbol appears
  int64_t ConstantValue() const;        // precondition: IsConstant()
  std::string ToString() const;         // "2*M + N" — for diagnostics/printing

  friend SymbolicExpr operator+(SymbolicExpr, const SymbolicExpr&);
  friend SymbolicExpr operator-(SymbolicExpr, const SymbolicExpr&);
  friend SymbolicExpr operator*(const SymbolicExpr&, const SymbolicExpr&);
  // Exact division only (single-monomial divisor); nullopt when it doesn't
  // divide evenly or the divisor is a genuine polynomial sum.
  friend std::optional<SymbolicExpr> TryDivide(const SymbolicExpr&, const SymbolicExpr&);
 private:
  std::map<Monomial, int64_t> terms_;   // canonical form: no zero coefficients
};
```

This is deliberately narrow — no `floor`/`ceil`/`min`/`max` terms, no
rational functions, no inequality solving — because it is meant to be
mandatory, always-on infrastructure that every consumer of `onnx`'s core
library links against, and the class of expression ONNX shape-inference rules
actually generate (sums and products of `dim_param`s and constants: reshape
targets, pooling/conv output-length formulas, concat/broadcast axis sizes,
cache-length arithmetic) is exactly a polynomial, never a general rational
function. This mirrors the scope onnxsim's `SymExpr` already settled on in
production (`sym_expr.h` in `onnxsim/onnxsim`), which this proposal takes as
its starting reference implementation — it is pure standard C++17 with no
external dependency (no `sympy`, no SymEngine/GMP), already unit-tested, and
already exercised against real transformer-export graphs.

### Wiring into existing infrastructure

- **`SymbolTable`** (`onnx/defs/shape_inference.h`) already mints fresh
  symbol names via `createNew(prefix)`. Extend it (or a subclass used only by
  the graph-level inference driver in `implementation.cc`) with a side table
  `std::unordered_map<std::string /*symbol*/, SymbolicExpr>` recording what
  polynomial each generated symbol stands for, and a lookup the other way
  (`SymbolicExpr -> existing symbol name`) so that two occurrences of the same
  polynomial reuse one symbol rather than minting duplicates — this is what
  gives `unifyDim` a real equality test instead of only string identity.
- **`TensorShapeProto::Dimension operator+/-/*`** in `shape_inference.h` gain
  a `Dimension op Dimension` general case: build the two dims' `SymbolicExpr`
  (a `dim_value` is a constant expr, a `dim_param` looks itself up in — or is
  freshly registered in — the active `SymbolTable`'s side map, defaulting to
  `SymbolicExpr::Symbol(name)` when not yet a compound expression), combine
  algebraically, and materialize the result: a constant result sets
  `dim_value`; a non-constant result is looked up/registered as a symbol via
  `SymbolTable` and sets `dim_param` to that symbol's name. `MaterializeSymbolicShape`
  (`implementation.h`) — which already walks inferred types converting
  placeholder symbols into real ones — is the natural place to finalize this
  for a whole `TypeProto` at the end of a node's inference call, exactly as
  it does today for plain unknown-dim symbols.
- **`DataPropagationContext`** (`shape_inference.h`, used by the
  `enable_data_propagation` path 0005 introduced): its `TensorShapeProto`
  value map already carries per-dimension `dim_value`/`dim_param` entries for
  *data* (not just shape) tensors. The arithmetic ops it dispatches through
  for `Add`/`Sub`/`Mul`/`Div`/`Equal`/`Where` on shape-family tensors route
  through the same `Dimension` operators above, so they gain the identical
  capability for free — an `Add` between two propagated `dim_param` values in
  a `Shape -> Gather -> Add -> Concat -> Reshape` chain now propagates
  `M + N` instead of stopping.
- **Backward compatibility**: no protobuf field changes, no opset version
  implications — this is exactly the same category of change as 0005 itself,
  which shipped as an ONNX 1.10 capability improvement to `InferShapes`/
  `ShapeInferenceOptions` with no IR version bump. Existing serialized models
  are unaffected; a caller who does not read the (still purely in-memory,
  never serialized) symbol-to-expression side table sees only `dim_value` /
  `dim_param` on the output exactly as today, with more of them resolved.

### What is *not* proposed

- No change to `TensorShapeProto` itself, or to any other `.proto` message —
  no expression syntax is added to the serialized IR.
  `SymbolTable`'s existing string-symbol contract for `dim_param` is
  unchanged; a generated symbol standing for `M + N` is just a normal
  `dim_param` string like `"unk__42"` to every consumer that isn't looking at
  the new in-memory side table.
- No general CAS. See "Rationale and alternatives" below.
- No change to operator semantics or op-schema versioning. Individual ops'
  `TypeAndShapeInferenceFunction`s opt into the richer arithmetic simply by
  using the (now more capable) existing `operator+`/`operator*`/`unifyDim`
  helpers; ops that don't touch multi-symbol arithmetic are unaffected.

## Drawbacks
[drawbacks]: #drawbacks

- **Maintenance surface.** `onnx/defs/shape_inference.h` and
  `onnx/shape_inference/implementation.cc` are linked into every consumer of
  the core library, including ones with no interest in symbolic algebra. A
  polynomial engine, however small, is more C++ for every downstream build
  (runtimes, converters, the Python bindings) to carry, test, and keep
  correct.
- **Merge-policy risk.** onnxruntime's own experience with
  `symbolic_shape_infer.py`'s `auto_merge_` heuristic (documented in its own
  source as capable of incorrectly equating dims that only happen to agree
  at a sampled value) shows that *equality* decisions over symbolic
  expressions are the actual hard part, not the arithmetic. This proposal
  restricts equality to *structural* equality of the canonical polynomial
  (so `M+N == N+M` but `M+N` vs `M` stays "unknown," never "assumed equal");
  getting the unify/merge call sites in `implementation.cc` to respect that
  distinction correctly everywhere is real, careful work, not a mechanical
  change.
- **Slightly more inference-time cost.** Building and canonicalizing a
  polynomial on every symbolic dimension operation is not free, though
  onnxsim's production experience with the same representation
  (`std::map<Monomial, int64_t>` insert/merge, no simplification search)
  suggests this is small relative to existing shape-inference costs.

## Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

- **Why a hand-rolled polynomial type instead of depending on a CAS?**
  `sympy` is Python-only and cannot be a dependency of `onnx`'s C++ core.
  SymEngine is C++ but its practical build needs GMP (or an experimical
  boost-multiprecision fallback) — an unattractive new mandatory dependency
  for a library many consumers build in constrained environments (WASM,
  mobile, embedded runtimes), which is precisely why onnxsim rejected the
  same option for its own C++ core. A small, dependency-free, purpose-built
  polynomial type keeps the cost proportional to the benefit.
- **Why not the full generality of a CAS (floor/ceil/min/max, inequalities,
  rationals)?** Every documented use case — this proposal's, 0005's, and the
  KV-cache example above — needs sums and products of dims and constants,
  never more. onnxruntime's tool *does* implement some of this (its own
  `int_max_` clamp heuristics stand in for a real inequality theory), and
  that extra surface is exactly where its own documentation flags the least
  confidence. Staying deliberately narrower than the state of the art among
  the external tools is a feature for something that has to be core,
  mandatory infrastructure rather than an opt-in analysis tool.
- **Why fix this in `onnx` instead of leaving it to downstream tools?** The
  Motivation section's list — onnxruntime, `justinchuby/onnx-shape-inference`,
  and onnxsim — is not evidence that the ecosystem has already solved this
  satisfactorily elsewhere; it's evidence that four separate projects
  maintain four separate, partially-overlapping implementations of the same
  documented gap in the thing they all depend on. Landing a narrow,
  dependency-free version in `onnx` itself is the one change that stops that
  duplication for all four (and any future tool) at once, rather than adding
  a fifth.
- **Impact of not doing this**: the status quo from 0005 continues —
  `onnx`'s own reference shape inference stays the one tool in this space
  that cannot do symbolic arithmetic at all, and every serious consumer that
  needs it keeps reimplementing it independently, at their own correctness
  and coverage risk, exactly as has already happened four times.

## Prior art
[prior-art]: #prior-art

- [0005 — Symbolic Shape Inference And Partial Data Propagation](0005-SymbolicShapeInfProposal.md):
  the direct predecessor. Introduced `SymbolTable`, symbol generation for
  anonymous unknown dims, and `DataPropagationContext`; explicitly deferred
  "add symbolic expressions" as a non-goal for "future iterations." This
  proposal is that future iteration.
- [`onnxruntime/python/tools/symbolic_shape_infer.py`](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/symbolic_shape_infer.py):
  `sympy`-based, ~100+ per-op `_infer_*` handlers, `sympy_data_` value
  propagation, heuristic dimension merging (`_add_suggested_merge`,
  `auto_merge_` — documented by ORT itself as a soundness tradeoff),
  `int_max_` heuristics for unbounded literals. Broader operator coverage
  than this proposal targets, at the cost of a `sympy` dependency and being
  a standalone Python tool rather than part of `onnx`'s own inference pass.
- [`justinchuby/onnx-shape-inference`](https://github.com/justinchuby/onnx-shape-inference):
  also `sympy`-based, built directly on `onnx_ir` rather than round-tripping
  protobuf; adds anonymous-engine-symbol-to-author-symbol reconciliation.
  Faces the same "not usable inside `onnx`'s C++ core" constraint as ORT's
  tool. See also the open [`onnx/ir-py#57` "Integrate symbolic shape
  inference"](https://github.com/onnx/ir-py/issues/57), which asks a version
  of the same question this proposal answers for the classic C++ core.
- `onnxslim`: no independent symbolic engine; depends entirely on whichever
  of the above is installed, illustrating the ecosystem cost of the gap
  staying unfilled in `onnx` itself.
- `onnxsim`'s `SymExpr` (`onnxsim/onnxsim#527`, and M1-M3 of
  `onnxsim/onnxsim#532`, wired into `_EvalPartialShape` in `onnxsim.cpp`):
  this proposal's reference implementation. Same representation
  (integer-coefficient polynomial, `std::map<Monomial, int64_t>`), same
  scope decision (no CAS generality), already shipped and folding real
  KV-cache-transformer export graphs in production. The companion
  onnxsim-side RFC (`onnxsim/onnxsim` `docs/symexpr-shape-inference-rfc.md`,
  closing `onnxsim/onnxsim#597`) documents that implementation in detail and
  motivated this proposal's existence.

## Unresolved questions
[unresolved-questions]: #unresolved-questions

- **Where should the polynomial type live?** A new
  `onnx/shape_inference/symbolic_expr.h`, or folded directly into
  `SymbolTable`/`shape_inference.h`? Should it be reachable from the public
  C++ API at all (for a caller who wants the human-readable formula, e.g. for
  diagnostics or a downstream optimizer like onnxsim), or stay a private
  implementation detail of `implementation.cc`? (This is the in-process
  question only — whether to *persist* a resolved formula across a
  save/reload is a separate, larger question; see "Future possibilities.")
- **Opt-in vs. always-on.** 0005 introduced `enable_data_propagation` as an
  explicit opt-in flag on `ShapeInferenceOptions` precisely because it changes
  what gets folded. Should symbolic-dimension algebra be gated behind that
  same flag (simplest — it's already the "richer inference" opt-in), a new
  flag, or always-on given it can only ever resolve *more* than today, never
  contradict a value shape inference would otherwise have produced?
- **Merge-policy strictness.** Should structural-equality-only unification
  (this proposal's default) ever be relaxed toward something like ORT's
  `auto_merge_`, or is staying strictly conservative (never assume equal
  without a structural proof) the right permanent stance for code that ships
  as part of `onnx` itself, given ORT's own experience with the risk?
- **Relationship to `onnx-ir`**: should this land in the classic protobuf-based
  C++ core (`onnx/shape_inference`), the newer `onnx_ir`/`onnx/ir-py`
  project, or both — and if both, should they share one `SymbolicExpr`
  implementation?
- Out of scope for this RFC, left for follow-ups: broader per-op coverage
  (attention/RoPE/GQA-style reshape patterns), any inequality/positivity
  reasoning for `Slice`/`Pad` clamping, and `If`/`Loop`/`Scan` subgraph
  propagation of symbolic values.

## Future possibilities
[future-possibilities]: #future-possibilities

- **A shared engine instead of four.** If this lands, `onnxruntime`'s and
  `justinchuby/onnx-shape-inference`'s tools would have the option of
  delegating to `onnx`'s own algebra for the C++-reachable subset of their
  work instead of maintaining independent `sympy` logic, and `onnxslim`
  would gain real symbolic-shape capability without an external dependency
  at all.
- **Bounded range/positivity reasoning.** A natural, separately-scoped
  follow-up for the `Slice`/`Pad` clamping cases ORT's tool currently handles
  with `int_max_` heuristics — deliberately left out of this proposal because
  it's a soundness policy question (what may be assumed about a symbol's
  sign/range), not a pure engineering extension of the polynomial algebra.
- **Exposing formulas for tooling, in-process.** Within a single
  `InferShapes`/`enable_data_propagation` call, a caller can already read
  "this dim is `M + N`" straight out of the (in-memory, non-serialized)
  symbol-table side map this proposal's reference-level explanation
  describes, before the process ends — no spec change needed for that case.
- **Persisting formulas across a save/reload, as its own follow-on
  proposal — not part of this one.** The harder version of the previous
  bullet is making a resolved formula legible to a *different* process that
  only reads the saved `.onnx` file (a downstream compiler/optimizer like
  onnxsim, without re-running graph-level symbolic shape inference itself).
  That needs a genuine wire-format addition, so it should stay out of this
  proposal's scope and be judged on its own:
  - It should be shaped as an **additive, optional, per-graph side table**
    (symbol name → expression, serializing the same `SymbolTable`-keyed map
    this proposal already builds in memory) rather than a new arm on
    `TensorShapeProto::Dimension` itself. A sibling field/message is the
    protobuf-safe kind of change — old readers skip an unrecognized field
    number, old files simply lack it — where a `dim_param`/`dim_expr`
    `oneof` would instead force a precedence question (which one is
    authoritative when both are present) that a purely additive annotation
    never raises. `dim_param` remains the sole source of dimension
    *identity*, unchanged; the new table only ever adds detail about a
    symbol that already exists. A per-graph table also avoids repeating
    (and risking disagreement among) the same formula at every one of a
    symbol's occurrences across a graph's `value_info`.
  - It must be specified as explicitly **advisory, never load-bearing** —
    the same posture `value_info` and `doc_string` already have. A
    transform that doesn't understand the annotation, or that changes what
    a symbol means, is not obligated to update or drop it; a consumer that
    trusts it without re-verifying accepts that risk, exactly as a consumer
    that trusts unverified `value_info` shapes does today.
  - It still needs its own expression grammar decision (plain-string
    polynomial vs. a small structured message) and its own referential-
    integrity story (a formula naming `M` is only meaningful if `M` is a
    real `dim_param` reachable in that graph) — real, separable design work
    that a follow-on RFC should own rather than inheriting from this one.
- **Constant folding across dynamic dims** in `onnx`'s own optimizer passes,
  using the same symbol table — today this is exactly the kind of thing a
  downstream tool like onnxsim has to implement entirely outside `onnx`
  because `onnx` itself has no such algebra to build on.
