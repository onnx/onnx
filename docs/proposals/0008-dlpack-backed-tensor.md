<!--
Copyright (c) ONNX Project Contributors
-->

<!--- SPDX-License-Identifier: Apache-2.0 -->
- Feature Name: `dlpack_backed_tensor`
- Start Date: 2026-08-21
- RFC PR: [onnxsim/onnx#0000](https://github.com/onnxsim/onnx/pull/0000)
- Status: under discussion
- Authors:
  - Takeshi Watanabe ( https://github.com/take-cheeze )

## Summary
[summary]: #summary

Replace the in-memory C++ IR's `onnx::Tensor` (`onnx/common/tensor.h`) byte storage — currently a five-way union of a raw-bytes string, five separate typed vectors, and an external-data sentinel, all deep-copied on every `Tensor` copy — with a single, explicitly-checkable, reference-counted buffer handle shaped like a `DLManagedTensor`. Copying a `Tensor` becomes a pointer-and-refcount operation instead of a full byte copy, and "no data available" becomes one checkable state (`has_data()`) instead of an implicit union that every call site has to get right on its own. This is scoped entirely to the internal C++ IR (`onnx/common/ir.h`, `onnx/common/tensor.h`); it does not touch `TensorProto`, `onnx.in.proto`, or any serialized wire format.

## Motivation
[motivation]: #motivation

Two independent problems currently point at the same storage model.

**Cost.** `Tensor` owns its bytes outright: `raw_data_` is a `std::string`, and every typed-vector field (`float_data_`, `double_data_`, `int32_data_`, `int64_data_`, `uint64_data_`) is a plain `std::vector`. There is no reference counting anywhere in the type. `Graph::initializers_` is `std::vector<Tensor>`, and `TensorAttr` (`ScalarAttributeValue<Tensor, AttributeKind::t>`) stores a `Tensor` by value — so every graph clone, every attribute copy, every node duplication in an optimization pipeline drags a full deep byte copy of every tensor's payload along with it. This is negligible for a small convolution kernel. It is the dominant cost of running an optimizer pipeline at all on a multi-gigabyte embedding table or a large language model's weight tensors, and it actively discourages tools built on this IR from doing the kind of lazy, memory-mapped model loading that large models require.

**Correctness.** This is not hypothetical. It is a real bug class found and fixed in a downstream onnx-optimizer fork while building exactly the kind of lazy loading described above. A pooled, memory-mapped model loader left large tensors in `TensorProto::EXTERNAL` state — no bytes materialized in the `Tensor` — specifically to avoid the copy cost above. Auditing the optimizer's passes against that state surfaced multiple call sites that read an `EXTERNAL` tensor as if it were fully populated:

- A batch-normalization/convolution fusion pass silently folded uninitialized memory into convolution weights.
- A duplicate-initializer elimination pass merged two unrelated external tensors as if they were bit-identical.
- A constant-condition elimination pass indexed into an empty vector and crashed the process.

All three were confirmed by reverting the fix under test and reproducing the failure. The common root cause in each case: `Tensor` has four or five mutually exclusive storage states (raw bytes, one of several typed-vector kinds, external, none), and nothing in the type forces a caller to check which one is actually live before calling `data<T>()` or one of its typed-accessor siblings. As more tools build memory-conscious loading paths on top of this IR — which large-model workflows increasingly require — this same class of silent-wrong-result or crash bug will keep recurring at every call site that was written assuming a `Tensor` always has its bytes in hand.

Both problems share a single fix: give `Tensor` one explicit, checkable buffer handle instead of an implicit union of storage strategies, and make holding that handle cheap to copy, so pipeline code stops paying for — and stops risking correctness on — a full materialize on every clone.

## Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

For code that already uses `onnx::Tensor`, almost nothing changes on the surface:

```cpp
Tensor t = LoadSomeTensor();
if (t.has_data()) {
  auto* p = t.data<float>();
  // ... read p[0 .. elem_num())
}
```

`has_data()` is new, and replaces scattered checks against `data_location_ == TensorProto::EXTERNAL`. Every other accessor in common use — `sizes()`, `elem_type()`, `data<T>()`, `raw()`, `floats()`/`int64s()`/etc. — keeps its existing signature and behavior whenever data is present. A pass author who never touches an `EXTERNAL` tensor sees no change at all beyond the new, cheap copy cost.

What does change, and is the point of this proposal, is that copying a `Tensor` is now cheap regardless of how large its payload is:

```cpp
Tensor a = LoadSomeTensor();   // owns a 4 GB buffer
Tensor b = a;                  // today: a 4 GB memcpy.
                                // proposed: refcount++, no bytes touched.
```

This means pass authors who currently avoid copying `Tensor` out of performance necessity — and who therefore sometimes hold raw pointers or indices into a parent structure instead, a pattern that itself is a source of lifetime bugs — can go back to holding a `Tensor` by value, which is both simpler to reason about and consistent with `Graph::initializers_` already being declared as `std::vector<Tensor>`.

A migration note for existing pass authors: any code that currently branches on `tensor.has_data_location() && tensor.data_location() == TensorProto::EXTERNAL` (or an equivalent construction) should switch to `!tensor.has_data()`. Code that never checked for `EXTERNAL` at all — the pattern responsible for the correctness bugs described above — will now throw a catchable exception from `data<T>()` instead of reading whatever bytes happen to be behind an unrelated, uninitialized field.

## Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

### Storage

```cpp
class TensorBuffer {  // refcounted DLManagedTensor handle
 public:
  static TensorBuffer Borrow(DLManagedTensor* t);   // no-op deleter, caller owns lifetime
  static TensorBuffer Adopt(DLManagedTensor* t);    // takes deleter ownership
  static TensorBuffer Allocate(size_t nbytes);       // fresh owned storage

  const void* data() const;
  size_t nbytes() const;
  bool empty() const { return dl_ == nullptr; }      // replaces the EXTERNAL sentinel

  TensorBuffer(const TensorBuffer&);   // atomic refcount++ (shared_ptr)
  TensorBuffer& operator=(const TensorBuffer&);
  ~TensorBuffer();                     // atomic refcount--, deleter runs at 0

 private:
  std::shared_ptr<DLManagedTensor> dl_;
};

class Tensor final {
  // sizes_, elem_type_, name_, segment fields: unchanged, unaffected by this proposal
  TensorBuffer buffer_;   // replaces float_data_/double_data_/.../raw_data_/external_data_
};
```

`float_data_`, `double_data_`, `int32_data_`, `int64_data_`, `uint64_data_`, `raw_data_`, `is_raw_data_`, `data_location_`, and `external_data_` are all removed as separate members. Every element type is read through `data<T>()` reinterpreting the one buffer — close to what the raw-data path already does today, just made universal instead of one case among several.

`string_data_` is the one exception and is *not* folded into `TensorBuffer`; see Unresolved questions.

### dtype, shape, strides

Only bytes move into `TensorBuffer`. `elem_type_` (the `TensorProto::DataType` wire enum) and `sizes_` (`vector<int64_t>`) stay exactly where they are today as separate `Tensor` members — both are metadata needed even when `buffer_.empty()` is true, for instance an `EXTERNAL` tensor mid–shape-inference with no bytes pooled yet.

| Concept | Lives in | Converts at the DLPack boundary via |
|---|---|---|
| dtype | `Tensor::elem_type_` (unchanged) | ONNX-dtype ↔ `DLDataType` map, see Prior art |
| shape | `Tensor::sizes_` (unchanged) | direct: `sizes_.data()/.size()` ↔ `DLTensor::shape/ndim` |
| strides | not tracked — implied contiguous | `DLTensor::strides` always `nullptr` |

ONNX has no stride concept today — a tensor is always dense row-major, both in the proto and in the current `Tensor`. `DLTensor` does carry an optional `strides` field. This proposal keeps `TensorBuffer` compact-only as a hard invariant, not a runtime-checked convention: `strides` is always `nullptr` at construction. A non-contiguous borrow — a strided view into someone else's buffer — is explicitly out of scope for this proposal, not a case silently mishandled.

### Accessor compatibility

Existing accessor names are preserved as thin views over `buffer_`:

- `data<T>()` — throws if `buffer_.empty()`; otherwise reinterprets, as the raw-data path does today.
- `raw()` — returns a `string_view` over `buffer_.data()`/`nbytes()`; no allocation.
- `floats()`/`doubles()`/`int32s()`/etc. — become typed spans over the same buffer; kept for source compatibility, documented as legacy in favor of `data<T>()`.
- `set_raw_data(std::string)` — allocates a new owned `TensorBuffer` and moves bytes in once; the one place a real copy still happens, same as today.
- `has_data() const` — new; the canonical replacement for `data_location_ == TensorProto::EXTERNAL` checks scattered through pass code.

### Migration path

1. Land `TensorBuffer` and the accessor shims, removing the old five-vector storage but keeping every existing accessor signature. Gate this step on re-auditing every call site that currently branches on `data_location_` or assumes a specific typed vector is populated.
2. Convert pass code that depends on `Tensor` (in scope for the motivating downstream fork: on the order of thirty files under `passes/`) to `has_data()` in place of the old `EXTERNAL` check, one file at a time, backed by behavior-level regression tests rather than storage-layout tests, so they survive the migration unmodified.
3. Any project maintaining a side table to work around `Tensor`'s current inability to hold a borrowed view — for instance a `shared_ptr<const char[]>` + `string_view` per external tensor, kept in sync with the IR by convention — can fold that bookkeeping directly into `TensorBuffer::Borrow()` once this lands, removing a second, hand-synchronized representation of "does this tensor have bytes, and who owns them."

Each step is independently buildable and shippable; this proposal does not require landing all three at once.

## Drawbacks
[drawbacks]: #drawbacks

- **Shared mutable state.** Because copies now alias the same buffer, an in-place mutation through one `Tensor` becomes visible through every other `Tensor` copy that shares it. Today's deep-copy semantics made this impossible by accident, not by design, so no pass should be relying on it — but that needs to be verified across the ecosystem, not assumed, before this ships. See Unresolved questions.
- **DLPack C-ABI surface.** `DLManagedTensor` is a C-ABI struct with a manual `deleter` function pointer. Every construction path (`Borrow`/`Adopt`/`Allocate`) has to get that deleter right, or a borrowed buffer outlives its source and dangles. This is a well-understood but real class of bug to get exactly right in three places.
- **Narrower dtype support at the buffer boundary.** DLPack's `DLDataType` does not cover every value `TensorProto::DataType` can hold with a clean 1:1 mapping — string types in particular do not fit a flat typed buffer at all (see below). `elem_type_` itself is untouched by this proposal, but any code that goes through `TensorBuffer`'s DLPack shape inherits DLPack's narrower support.
- **Migration surface.** Even scoped to the internal IR, this is a breaking change to a class that downstream C++ consumers link against directly, not merely an internal implementation detail of this repository — and it touches every pass in at least one known downstream fork. That is real review and testing cost, proportional to how widely `onnx::Tensor` is depended on outside this repository, which this proposal does not have full visibility into (see Unresolved questions).

## Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

**Why DLPack specifically, rather than a bespoke refcounted buffer type?** DLPack is already reachable prior art: a downstream project's constant-folding / tensor-exchange boundary already uses `DLManagedTensor` as its ABI, including a working ONNX-dtype ↔ `DLDataType` map and little-endian-on-disk-vs-host-order handling. Reusing that shape collapses the conversion logic at the `Tensor` ↔ external-executor boundary to nothing, instead of maintaining two different "how do I describe a tensor buffer" conventions in the same ecosystem.

**Why not just wrap `raw_data_` in a `shared_ptr<std::string>` and stop there?** That fixes the copy-cost half of the motivation without fixing the correctness half: it does nothing about "no data" being an implicit condition spread across an enum comparison and a set of vector-emptiness checks. Collapsing to one buffer type with one `empty()`/`has_data()` check addresses both halves with one change instead of leaving the second half unaddressed.

**Why not leave large tensors as `EXTERNAL` and require every caller to special-case them, i.e. do nothing?** That is the status quo, and it is the thing that produced the bug class described in Motivation. Pushing the correctness burden onto every call site — of which there are dozens today and more as memory-conscious tooling grows — does not scale; making the empty state impossible to overlook does.

**Impact of not doing this.** Tools built on this IR that need to handle large models without materializing every tensor in memory will keep re-discovering the same class of silent-wrong-result and crash bugs independently, one pass at a time, as this proposal's motivating case did.

## Prior art
[prior-art]: #prior-art

- **A downstream constant-folding bridge already in this ecosystem** does exactly this kind of conversion at its executor boundary: it wraps a `DLManagedTensor` with a custom deleter via manager-context structs, provides a bijective (intentionally restricted to `lanes == 1`) ONNX-dtype ↔ `DLDataType` map, and handles the little-endian-on-disk vs. host-order mismatch for raw bytes. This proposal reuses those idioms rather than inventing new ones.
- **PyTorch's `Tensor`/`Storage` split** and **TensorFlow's `Tensor`/`TensorBuffer` split** both separate tensor metadata (shape, dtype) from a refcounted byte buffer, for the same reason: cheap views and cheap copies. Both predate, and motivate, DLPack's existence as a common tensor-exchange ABI between frameworks built this way.
- **Apache Arrow's `Buffer`** is a similar refcounted-buffer-plus-metadata split in a different serialization ecosystem, for the same underlying reason: zero-copy sharing across process and library boundaries.
- **No existing workaround inside this codebase fully solves the problem** — the motivating downstream fork's own workaround (a side `TensorPool` mapping tensors to memory-mapped byte ranges, kept in sync with the IR by convention rather than by the type system) is exactly the kind of parallel bookkeeping this proposal is meant to make unnecessary.

## Unresolved questions
[unresolved-questions]: #unresolved-questions

- **Copy-on-write, or push mutability out of the type?** Cheap copies plus in-place mutation is a classic aliasing hazard (see Drawbacks). Two options: (a) `Tensor` is logically immutable once `buffer_` is set — "mutation" always means constructing a new `Tensor` — matching how the prior-art DLPack bridge already treats DLPack tensors as effectively immutable views; (b) real copy-on-write, cloning on first mutable access when the refcount is greater than one. This proposal leans toward (a) for simplicity but expects this to be resolved through RFC discussion before implementation.
- **Does `string_data_` fit this model at all?** String tensors are variable-length per element — DLPack has no native representation for that. It likely stays a distinct `std::vector<std::string>` path outside `TensorBuffer`, meaning `Tensor` keeps one narrow, permanent exception to "single buffer handle." This should be settled by this RFC rather than discovered mid-implementation.
- **Segmented tensors (`is_segment_`/`segment_begin_`/`segment_end_`)** have not yet been examined against this design. Implementation needs to confirm segments compose cleanly as a sub-view of a parent `TensorBuffer`, or determine they need their own borrowing mechanism — expected to be resolved during implementation rather than blocking RFC acceptance.
- **Alignment contract.** Memory-mapped external buffers are page-aligned; today's typed-vector storage has no alignment guarantee beyond the standard allocator's. If any consumer wants SIMD-aligned access via `data<T>()`, that needs to be a documented contract of `TensorBuffer::Allocate`, decided as part of this RFC rather than assumed later.
- **Actual blast radius outside this repository.** This proposal was written with visibility into one downstream fork's dependence on `Tensor`'s accessor surface (on the order of thirty files). It does not have visibility into how many other projects link this repository's C++ library directly and depend on `Tensor`'s current copy semantics or storage layout — surfacing that is exactly what discussion on this RFC, and outreach to the relevant SIG, is for.

## Future possibilities
[future-possibilities]: #future-possibilities

- If this proves out, the same buffer-handle approach could extend to a first-class, zero-copy `TensorProto` ↔ `Tensor` ↔ DLPack round-trip for constant folding and external-executor handoff across the ecosystem generally, rather than each downstream tool building its own bridge.
- A stride-carrying `TensorBuffer` variant (explicitly deferred above) could later let passes represent common view operations — transpose, slice, broadcast — without a copy, if a concrete pass workload demonstrates the need. This is out of scope here to keep the initial change reviewable.
- Once `Tensor` has a real refcounted buffer, a similar pass over `Graph`/`Node`/`Value`'s other value-copied members could be worth a follow-up look, though none of them carry `Tensor`'s multi-gigabyte payload problem today.
