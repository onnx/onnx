<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

- Feature Name: Extensible Symbolic Shape Annotation, Checking, and Inference
- Start Date: 2026-08-10
- RFC PR: [onnx/onnx#0000](https://github.com/onnx/onnx/pull/0000)
- Status: under discussion
- Authors: gramalingam

# Summary

This RFC proposes a modular extension of
(a) ONNX's shape annotation to allow symbolic dimension expressions, and
(b) Its utilities for checking and inference of shape annotations to support
symbolic dimension expressions.

# Motivation

ONNX already supports shape annotations in models, as well as a checker and inference,
for the following reasons:
* They serve as documentation. Users have a much easier time understanding
models when they have type and shape annotations.
* They help catch various runtime errors through the static checker.
* They enable runtime optimizations. Inference performance for modern large
(DNN) models largely hinges on statically known shapes of tensors, which are
the basis for memory allocation, memory reuse, static memory planning, etc.

This proposal assumes that the value of shape annotations, the checker and the inference
utility for shape annotations, are clear and focuses only on the proposed extensions
to these.

ONNX already supports dimensions represented by a concrete integer or a
symbolic string. This is useful for models with dynamic batch or sequence
dimensions:

```text
(BatchSize, SequenceLength, 1024 /*HiddenSize*/)
```

In practice, exporters and model tools commonly need expressions such as:

```text
past_length + current_length
2 * sequence_length
CeilDiv(sequence_length, 8)
```
Expression-like strings already occur in models, even though the current
specification says that `dim_param` should be an identifier. This is not
enforced by the checker due to a history of violations of this requirement.
These expressions improve documentation and can allow the checker and
inference engine to validate and propagate more shape relationships.

One challenge is that general symbolic arithmetic is not a single,
complete, inexpensive decision problem. For example, deciding whether two
arbitrary expressions are equal may be difficult or undecidable. A reasoning
limitation must not automatically become a false model error.
A second difficulty is in fixing the set of function symbols allowed in
symbolic dim expressions. It is likely that this set will evolve over
time. It is useful to have an extensible design that accommodates this,
allowing users to extend this on their own (with custom function symbols).

These motivations lead to the following goals and non-goals.

# Goals

* Make symbolic dimension expressions interoperable between ONNX tools.
* Allow users to extend these expressions with their own custom function (names)
* Preserve compatibility with existing `dim_param` strings.
* Improve shape checking and inference without requiring a complete theorem
  prover.
* Allow callers to extend ONNX's shape checker, inference, and data propagation
utilities by supplying more capable symbolic reasoning implementations.

# Non-goals

* Standardizing one fixed arithmetic solver.
* Requiring every backend or runtime to implement symbolic algebra.

# Overview

The proposal is organized into two parts:

* **Symbolic expressions** — the abstract syntax of dimension expressions,
  their concrete textual syntax, and their serialized/in-memory
  representation. This part is purely about *what a dimension expression
  is* and how it is written down; it introduces no new reasoning behavior.
* **Checker and inference extensions** — how the checker and inference
  engine use dimension expressions, starting from the basic injectable
  reasoning service and then covering the harder questions that arise once
  declared and inferred information must be reconciled.

# Independent design questions

This document describes and addresses the following (mostly independent)
design questions with recommendations for each.

| ID | Question | Recommended option |
| --- | --- | --- |
| Q1 | Should ONNX define a symbolic expression language? | Yes, minimal extensible language |
| Q1b | Should the textual grammar support infix sugar (e.g. `+` for `Add`)? | Yes, limited to a fixed, predefined set of ONNX-defined symbols; not user-extensible |
| Q1c | Should function symbols share the namespace of ONNX operators? | No; keep them separate, but reuse names where they denote the same function |
| Q2 | Which serialized representation should be used? | Structured form as the primary representation; text remains a required compatibility encoding at I/O boundaries |
| Q2b | Should the structured form be a new `DimExprProto`, or should existing `Dimension` be extended? | Extend `TensorShapeProto.Dimension` with a recursive function variant |
| Q2c | What about existing `dim_param` strings that already hold expressions? | Provide up/down-conversion utilities; let users control behavior via options |
| Q3 | Should protobuf objects also be the internal AST? | Yes; reuse them, and consider a separate AST later only if needed |
| Q4 | Should symbolic reasoning be injectable? | Yes |
| Q5 | How should the checker/inference layer reconcile inferred information with a pre-existing declaration, including inconclusive or contradictory reasoning results? | Delegate reconciliation to the reasoning service's `unify`/`assume`/`simplify`; preserve the `Unknown`/`Contradiction` distinction, with strictness as a caller/checker policy; treat merge-policy pluggability as a separate, optional question |

# Symbolic expressions

## Abstract syntax (Q1)

### Recommended: standardize a minimal language

The proposed abstract syntax is:

```text
dim-expr ::= non-negative-integer
           | symbolic-id
           | function-symbol "(" dim-expr-list ")"
```

Function symbols may be ONNX-defined or qualified by a user/domain namespace.
For example:

```text
Add(length, 1)              # ONNX-defined function symbol (empty domain)
com.microsoft.Sqrt(N)       # domain-qualified extension function symbol
```

The first example applies the ONNX-defined `Add` function symbol to a
symbolic-id (`length`) and a non-negative-integer literal (`1`). The second
example applies a function symbol named `Sqrt`, qualified with the
`com.microsoft` domain, to a single symbolic-id argument (`N`); the domain
qualification distinguishes it from any ONNX-defined or other extension's
`Sqrt`, mirroring how `NodeProto.domain` disambiguates operator names.

The standard should define
* A textual serialization of the abstract syntax (straightforward), and
* The set of allowed function symbols (in the ONNX domain) and their
meaning.
* The meaning of function symbols in custom-domains are defined by the
users who own that domain, and other tools (including ONNX) can handle
them as uninterpreted function symbols.

### Alternative: treat every expression as an opaque string

An alternative to defining an expression language is to allow `Dimension.dim_param`
to be an arbitrary opaque string. This maximizes short-term compatibility but
leaves parsing, function identity, and semantics inconsistent between tools.
It also makes reliable checking and structured serialization difficult.
In current usage `"past_length + current_length"` and `"past_length+current_length"`
(which differ only in the use of white-space)
are treated as distinct symbolic dimensions, which is undesirable.

## Concrete textual syntax

### Q1b: should the textual grammar support infix syntactic sugar (e.g. `+` for `Add`)?

Existing `dim_param` strings such as `past_length+curr_length` already use
infix arithmetic notation rather than the prefix `function-symbol(...)` form.
Requiring every existing expression-like model to be rewritten in prefix
form purely for parsing purposes would be an unnecessary compatibility
break, so some infix support is needed regardless of Q2's outcome.

The recommendation is to treat infix notation strictly as **textual sugar**:
it affects only the parser and printer, not the AST (Q3) or the structured
protobuf form (Q2), which both use `DimFunction(name, args)` uniformly. A
parser desugars `a+b` to `DimFunction(domain="", name="Add", args=[a, b])`
before constructing the AST; a printer may resugar the same node back to
`a+b` when printing.

As a first cut, infix sugar should be:

* **limited to a fixed, small set of ONNX-defined (empty-domain) function
  symbols** — e.g. `+` for `Add`, `-` for `Sub`, `*` for `Mul`, and possibly
  `//` for a floor-division symbol — predefined by the ONNX standard grammar,
  with fixed precedence and associativity;
* **not extensible by users.** A domain-qualified extension function symbol,
  such as `com.microsoft.Sqrt`, is always written in prefix form:
  `com.microsoft.Sqrt(N)`. Allowing user-defined infix or other custom
  syntax would fragment the grammar and complicate parsing/printing
  portability across implementations;
* **purely notational.** `a+b` and `Add(a,b)` must parse to the identical
  `DimFunction` node; the printer's choice of infix versus prefix form (if
  configurable at all) must not change the expression's meaning or its
  canonical structured (protobuf/AST) representation.

This keeps the set of textual forms a reader must recognize small and fixed,
while accommodating the arithmetic notation already present in deployed
models. Additional infix operators, if ever needed, would be a standard
grammar change, not a per-user extension.

## Function namespace

### Q1c: should function symbols share the namespace of ONNX operators?

**Recommendation: no.**

ONNX's tensor operators subsume scalar integer operators: any scalar integer
operator or function can be mapped into a corresponding element-wise tensor
operator, as is commonly done in the ONNX standard. However, the integer
functions used in dimension expressions play a different role from tensor
operators, so there is no need (yet) to conflate the two and constrain
dimension expressions to use only ONNX operators. Where possible, the same
names should be reused for the same functions (e.g. `Add`, `Sub`, `Mul`), so
that the two vocabularies stay recognizably aligned. This can be revisited
later if unifying the namespaces proves valuable.

## Serialization format and internal representation

The current representation is:

```protobuf
message TensorShapeProto {
  message Dimension {
    oneof value {
      int64 dim_value = 1;
      string dim_param = 2;
    }
    optional string denotation = 3;
  }
}
```

### Q2: which serialized representation should be used?

#### Option A: generalize `dim_param`

Standardize that `dim_param` may contain the textual `DimExpr` grammar.

**Benefits**

* existing expression-like models continue to work;
* old readers retain a useful string;
* no protobuf change is required.

**Costs**

* every consumer must parse strings;
* structure is not explicit;
* malformed or non-canonical strings remain possible;
* user-defined function identity is less strongly represented.

#### Option B: make `Dimension` recursive

Use `TensorShapeProto.Dimension` itself as the recursive expression node.
Protobuf supports recursive nested messages, so no separate expression
message is required:

```protobuf
message TensorShapeProto {
  message Dimension {
    oneof value {
      int64 dim_value = 1;
      string dim_param = 2;
      DimFunction dim_application = 4;
    }
    optional string denotation = 3;
  }

  message DimFunction {
    // Empty domain denotes an ONNX-defined function symbol, following the
    // same convention used for operator domain/name (cf. NodeProto).
    string domain = 1;
    string name = 2;
    repeated Dimension argument = 3;
  }

  repeated Dimension dim = 1;
}
```

This is wire-compatible as a protobuf schema change, but old readers may treat
the dimension as unspecified because they do not understand the new field.

**Benefits**

* structure is explicit, so consumers need not parse strings;
* function identity (including domain-qualified user extensions) is
  represented directly;
* `TensorShapeProto` continues to represent the symbolic values used by
  partial data propagation (see "Data propagation" below), so those APIs
  need not change;
* only one expression representation exists, so no conversion between a
  dimension and a separate expression message is needed.

The `denotation` field may be present on leaf nodes, where it can describe a
value such as the number of heads, but has no semantic meaning on a compound
function node. The field remains structurally available on recursive nodes and
is ignored there; restricting its meaningful use to leaves is a semantic
constraint rather than a structural requirement.

##### Q2b: should the structured form be a new `DimExprProto`, or should existing `Dimension` be extended?

**Alternative to Option B: a separate `DimExprProto`**

Instead of making `Dimension` recursive, a distinct message could carry
expressions, with `TensorShapeProto.Dimension` adding a `DimExprProto
dim_expr = 4` field:

```protobuf
message DimExprProto {
  oneof value {
    int64 constant = 1;
    string symbol = 2;
    DimFunction application = 3;
  }

  message DimFunction {
    string domain = 1;
    string name = 2;
    repeated DimExprProto argument = 3;
  }
}
```

**Pros:** it keeps expressions in a self-contained message that is
independent of `TensorShapeProto`, and it can be reused in contexts that are
not shapes without dragging in dimension-specific fields such as
`denotation`.

**Cons:** it introduces a second expression representation alongside
`Dimension`, requiring conversions at every boundary; the data-propagation
value representation ("Data propagation" below), which currently uses
`TensorShapeProto`, would need corresponding API changes to carry
`DimExprProto` values; and its leaves would likely need to grow something
like `denotation` anyway, at which point they closely resemble `Dimension`.

The recommendation remains Option B (extend `Dimension` directly) over this
alternative, for the reasons given above.

### Q2c: what about existing `dim_param` strings that already hold expressions?

Deployed models already contain `dim_param` strings such as
`"past_length+curr_length"`, and old producers will keep emitting them. With
Option B, the same information can be expressed in two ways, so the two forms
must be able to interoperate.

The recommendation is **not** to mandate a single form in the standard, but to
provide *conversion utilities*:

* **Up-conversion** (string to structure): parse a `dim_param` string using the
  Q1 grammar (including the Q1b infix sugar) and replace it with the equivalent
  `dim_application` tree. A string that does not parse as a well-formed
  expression is left untouched and continues to denote an opaque symbolic name,
  so a name that merely happens to contain unusual characters is never
  reinterpreted.
* **Down-conversion** (structure to string): print a `dim_application` tree back
  into the canonical textual form and store it in `dim_param`, so that readers
  that predate the structured field still see a meaningful, and no less
  informative, dimension.

For values intended as expressions within the standard grammar, these
conversions are syntactically reversible. Up-conversion is not necessarily
semantics-preserving for a legacy `dim_param` that was intended as one opaque
symbol despite resembling an expression; the policy for handling that
ambiguity is discussed below.

The new `dim_application` field is introduced by a new IR version. Models using
an earlier IR version must not use the field. Writers targeting the new IR
version should prefer the structured form. To downgrade a model to an earlier
IR version, a converter must first replace every `dim_application` with its
canonical textual `dim_param` encoding and verify that no other feature
prevents the downgrade. The two serialized forms are alternatives in the same
`oneof` and therefore cannot both be emitted for one dimension. An older
consumer can preserve and display the downgraded expression text, but it is not
required to interpret that text as arithmetic.

IR versioning does not, by itself, determine how a tool should interpret an
expression-like `dim_param` in an existing model: such a string may have been
intended either as an expression or as one opaque symbolic name. Because the
right choice depends on the consumer and use case, tools should expose options
controlling whether they attempt up-conversion on load and how they handle
strings that fail to parse (ignore, warn, or reject). Whether preserving all
legacy `dim_param` values as opaque symbols should be the default, or whether
well-formed expressions should be parsed by default, remains an unresolved
compatibility-policy question. ONNX supplies the conversion mechanism without
silently changing the meaning of legacy models or forcing a migration schedule
on producers and consumers.

### Q2 recommendation

Adopt Option B as the primary representation: it is what the internal AST
(Q3) and the default symbolic reasoning implementation operate on directly,
avoiding repeated parsing and printing every time an expression is
inspected, unified, or simplified. Option A (the textual `dim_param`
grammar) remains a required compatibility encoding, but only at I/O
boundaries — via the Q2c up/down-conversion utilities, not as the working
representation of the reasoning module itself. A new structured field must
not silently change the meaning of existing fields.

### Q3: should protobuf objects also be the internal AST?

#### Recommended: reuse the protobuf representation

The recursive `Dimension` representation from Q2 already provides the tree
structure needed in memory:

```text
DimExpr
  Constant(int64)          -> dim_value
  Symbol(SymbolId)         -> dim_param
  DimFunction(sym, [args]) -> dim_application
```

Function symbols are identified by a domain-qualified pair `(domain, name)`,
mirroring how ONNX already identifies operators (`NodeProto.domain` /
`NodeProto.op_type`). An empty domain denotes an ONNX-defined function symbol;
a non-empty domain identifies a user or vendor extension namespace, avoiding
collisions between independently defined function symbols.

Using this representation directly keeps the scope of this proposal limited:
it requires no conversion at model boundaries, and — importantly — it lets
partial data propagation (see "Data propagation" below) keep its current
`TensorShapeProto`-based APIs unchanged. Helper utilities can supply the
operations implementations need — structural equality, hashing, inspection,
substitution, parsing, and printing — without introducing a second
representation.

**Naming convention.** The rest of this document uses `DimExpr` as the name
of the type representing a dimension expression, for example in the
`SymbolicDimReasoning` interface (Q4). Under this recommendation, `DimExpr`
is simply an alias for `TensorShapeProto.Dimension`; the two names are used
interchangeably. Should the alternative (a separate immutable AST, or a
separate `DimExprProto`, per Q2b) be adopted instead, `DimExpr` would name
that distinct type instead.

#### Alternative: a separate immutable AST

ONNX could define its own in-memory AST (using an arena, reference-counted
nodes, or hash-consing) and convert to and from the protobuf form at model
boundaries. Such an AST would be immutable, independent of protobuf, and
better suited to caching and efficient simplification.

The cost is broader refactoring and extension: every boundary needs
conversion code, and the data-propagation APIs (see "Data propagation" below)
that currently exchange `TensorShapeProto` values would have to be extended
to carry the new type. This can be pursued later, as an internal
implementation change, if profiling or simplification needs justify it; it
is not required by the standard and should not be the only implementation
choice permitted.

# Checker and inference extensions

## Basic Extension

The checker and inference engine in ONNX is unification-based: a key primitive used by
these is the unification of two dimensions (`TensorShapeProto.Dimension`). The
generalization of shapes to allow symbolic expressions as dimensions introduces
the question of how to generalize the unification strategy to handle this case.
Unfortunately, this is a hard problem in a formal technical sense. Various forms
of arithmetic reasoning are undecidable or intractable. Specific approximate
versions are doable: for example, if we handle all function symbols as uninterpreted
functions. However, hardcoding a specific approximation into the ONNX standard itself
seems undesirable. Rejecting a model as invalid because of a limitation in the reasoning
capability of the tool (typically referred to as a "false positive") can be problematic.

### Recommended: inject a reasoning service

Hence, we propose to parametrize the checker and inference engine with a reasoning
service. The inference driver supplies a default implementation, and callers may
provide a more capable implementation. A representative interface is:

```cpp
class SymbolicDimReasoning {
 public:
  virtual ~SymbolicDimReasoning() = default;

  virtual UnifyResult unify(
      const DimExpr& lhs,
      const DimExpr& rhs,
      const ConstraintOrigin& origin = {}) = 0;

  // ... other methods discussed later
};
```

The following operation is deliberately **not** part of this interface:

* **Fresh symbol generation.** Guaranteeing that a new symbol is unique
  across a model requires visibility of all names in use, which ONNX
  already provides through `SymbolTable`. A reasoning implementation that
  needs fresh symbols should receive the symbol table as a dependency
  rather than own the naming policy.

### Implementation detail: handling nested subgraphs

`GraphInferenceContext` already threads a `SymbolTable*`, `DataValueMap*`
(generated shape data), and `ISchemaRegistry*` from an outer graph into
every node's inference/data-propagation context and into every child
context created for a graph-valued attribute (e.g., `If`/`Loop`/`Scan`, or a
called function). `SymbolicDimReasoning` should be threaded the same way:
one instance, default or caller-supplied, passed into every node and into
every child scope, so a subgraph or called function shares the same
reasoning state as its parent. Shape inference and data propagation remain
separate operator callbacks, but both draw on this same instance and the
same symbol table for a given graph scope, keeping the two extension points
independently implementable while avoiding divergent or duplicated
reasoning state.

Function-symbol metadata and evaluation rules for user-defined `DimExpr`
function symbols are not addressed here; see this section's "semantics and
metadata for function symbols" responsibility above.

### Implementation detail: data propagation

Data propagation already represents the value of an integer tensor using a
`TensorShapeProto`, whose entries (dimensions) can be a concrete integer
(`dim_value`) or a symbolic name (`dim_param`). This already lets, e.g., a
`Shape` output be propagated as concrete and/or symbolic dimensions and
consumed by later `Reshape`/`Concat`/`Slice` computations. The gap is that a
dimension entry cannot currently be a general expression, such as `N + 1`.
This proposal extends the same symbolic expression representation used for
ordinary shape annotations (Q1–Q3) to this value representation, so that
propagated values can hold arbitrary dimension expressions and not just
constants or bare symbols. Because the recommended representation (Q2, Q3)
extends `TensorShapeProto.Dimension` in place, `TensorShapeProto` can continue
to represent symbolic data values and the data-propagation APIs need no
change. (Introducing a separate expression type or AST instead would require
updating those APIs accordingly.)

The initial supported domain may remain scalar and one-dimensional integer
tensors, matching current data-propagation coverage. It should distinguish
unknown values from known symbolic values, as today.

## Advanced Extensions

In the existing implementation, unification can cause a symbolic dimension
to be bound to a known constant or to another symbolic dimension. This is
handled locally (preferring a known constant over a symbolic dimension
and preferring a pre-existing symbolic dimension over another symbolic
dimension inferred to be equal to it), without any global impact (for example,
when we infer that two symbolic dimensions M and N in the input model must be
the same). The generalization to symbolic expressions complicates this
picture, since we may infer equality of two complex symbolic expressions.

### From renaming to reasoning: the constraint and naming model

When the declared and inferred sides are both simple symbols, "merging" them
is mostly a bookkeeping problem: pick one of the two names as canonical (the
declared/anchor name is naturally preferred) and rewrite every other
occurrence of the discarded name to match. Once dimensions can be arbitrary
expressions, the same operation can require unifying, for example,
`x` (a declared anchor) with `y + 1` (an inferred expression). This is no
longer mere renaming: establishing, checking, and simplifying such a
relationship is exactly what the injectable reasoning service (Q4) is for.

We extend the API for symbolic reasoning with a method `simplify` as below
that determines the best way to represent a symbolic expression, given
the constraints that have been learnt during inference: this can do
simple renaming (replacing `x` by `y`), simple substitutions (replace
`z` by `y+1`), or more complex replacement (of an expression `u+1` by
`w`).

```cpp
class SymbolicDimReasoning {
 public:
  // ...

  virtual DimExpr simplify(
      const DimExpr& expression,
      const SimplifyOptions& options = {}) = 0;

  virtual Proof prove(
      DimRelation relation,
      const DimExpr& lhs,
      const DimExpr& rhs) const = 0;

  virtual UnifyResult assume(
      DimRelation relation,
      const DimExpr& lhs,
      const DimExpr& rhs,
      const ConstraintOrigin& origin = {}) = 0;

  // ...
};
```

The `assume` and `prove` methods extend the basic mechanism, allowing the inference
layer to record other assumptions (eg., that one expression is less than or equal to
another expression), as well as to check whether some relationship is implied by
existing constraints. Concretely, the graph inference layer should:

1. identify relationships between declared and inferred information while
   processing nodes;
2. identify authoritative graph-declared (anchor) names;
3. report each relationship, together with its origin, to the reasoning
   service via `unify`/`assume`, which is what actually collects and retains
   the resulting constraints;
4. rewrite dimensions and symbolic tensor values via `simplify`; and
5. preserve deterministic canonical output.

### Materialization: when the reconciled result is written back

Nested graphs and speculative inference need child scopes or checkpoints.
Facts learned only in a failed branch must not leak into unrelated graph
paths.

```cpp
class SymbolicDimReasoning {
 public:
  // ...

  virtual Checkpoint checkpoint() const = 0;
  virtual void rollback(Checkpoint) = 0;
};
```

Because the reasoning service takes expressions by value and records
equalities in its own state rather than mutating a `Dimension` in place (as
the current `unifyDim(const Dim&, Dim&)` helper does), inferred bindings must
be written back to the protobuf explicitly. This occurs at three points:

1. **Local write-back**, immediately after unifying a dimension, reproducing
   today's behavior for the simple cases.
2. **Scope finalization**, at the end of each graph scope, re-simplifying
   every inferred `value_info`, graph output, and propagated symbolic value
   against the final constraint state. This is what allows a constraint
   learned at a later node to refine a dimension written earlier, and it
   updates all occurrences of a symbol rather than a single mutable
   dimension.
3. **Before scope exit or rollback**, since constraints established inside a
   nested or speculative scope are discarded by `rollback` and any shape not
   yet materialized would lose the inferred information.

Finalization is idempotent and must not introduce new constraints; it is a
query of the reasoning state followed by a write to the protobuf. Symbols
local to an inner scope must not leak into an outer scope's materialized
shapes.

`checkpoint`/`rollback` give a linear undo, not a branch merge: they do not
by themselves decide what a construct like `If` should materialize for its
combined result. Branch merging is an orchestration that the graph inference
layer builds on top of these primitives: checkpoint before the branch,
explore `then` and `else` from that checkpoint in turn (rolling back between
them), capture each branch's inferred output expressions, and combine the
two captures using `prove`/`simplify`—keeping an expression where both
branches provably agree, and otherwise falling back to a weaker result (a
fresh symbol, or `Unknown`). No new primitive is required for this, though a
dedicated `fork`/`join` pair could be considered later if this pattern
proves common.

Also out of scope: once a scope's facts are materialized and the scope
exits, this design does not revisit them. If a fact is learned later in the
graph (e.g. two previously distinct symbols turn out to be equal), the
current design does not retroactively re-simplify expressions already
written back inside an already-exited branch or nested scope.

### Unknown and contradiction handling

**Recommended: preserve the distinction.**

The `unify`/`assume`/`prove` calls used throughout merging can themselves
return an inconclusive or inconsistent result, and this is where the
distinction matters most in practice. The reasoning service should
distinguish:

```text
True          relation proved
False         relation disproved
Unknown       implementation cannot decide
Contradiction accepted constraints are inconsistent
```

A merge that ends in `Contradiction` (e.g. the declared and inferred shapes
are provably incompatible) should normally produce an `InferenceError` or
checker error, with the origin of the conflicting constraints. A merge that
ends in `Unknown` (the reasoning engine cannot decide whether the declared
and inferred information agree) is not a definitive conflict: inference
should generally preserve the unknown result and continue, producing a
sound partial result, while the checker may offer strict and permissive
modes for how to treat such cases.

# Unresolved questions

* Which function symbols, including their arities and semantics, should be
  included in the initial ONNX-defined standard set?
* Should tools preserve legacy expression-like `dim_param` values as opaque
  symbols by default, or parse well-formed expressions by default with an
  option to disable up-conversion?

# Prior art

The existing ONNX symbolic shape inference proposal introduced graph-level
symbol generation, symbolic propagation, and partial data propagation. Its
data-propagation API is the basis for the symbolic value extension described
here.

`onnx-light` provides relevant prior art in:

* AST-based symbolic expressions;
* simplification and partial evaluation;
* equality and upper-bound constraints;
* user-visible anchor reconciliation;
* `value_as_shape` propagation;
* sequence and map descriptors;
* custom operator callbacks; and
* nested graph inference contexts.

The proposed RFC does not require ONNX to copy that implementation. It uses
the same separation of concerns to identify useful extension boundaries.

[`onnx-shape-inference`](https://github.com/justinchuby/onnx-shape-inference)
is another relevant prior art. It performs symbolic shape inference directly
on the [ONNX IR](https://github.com/onnx/ir-py) (rather than on protobuf),
using SymPy for symbolic dimension arithmetic, and provides:

* symbolic shape inference and shape (value) data propagation, tracking
  known element values through chains such as `Shape` → `Slice` → `Concat`
  → `Reshape`;
* symbolic constraint resolution that reconciles engine-generated dimension
  names with author-declared symbolic names on graph outputs/`value_info`,
  renaming anonymous dimensions — including compound expressions such as
  `2*_d0` or `past_seq + seq` — to the declared names;
* an extensible registry for custom operator shape-inference functions,
  with version-aware dispatch across opset history; and
* selectable merge policies (`refine`, `strict`, `override`, `skip`)
  governing how newly inferred shapes are merged with existing ones.

Its SymPy-based expression arithmetic, anonymous-to-declared dimension
renaming, and selectable merge policy are concrete instances of the
symbolic reasoning, constraint/naming, and merge-policy pluggability
questions this RFC treats as extension points (Q4, Q5) rather than fixed
ONNX behavior.
