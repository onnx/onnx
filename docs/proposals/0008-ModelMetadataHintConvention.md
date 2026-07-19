<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->
- Feature Name: `model_metadata_hint_convention`
- Start Date: 2026-07-19
- RFC PR: [onnx/onnx#0000](https://github.com/onnx/onnx/pull/0000)
- Status: under discussion
- Authors: justinchuby

## Summary
[summary]: #summary

Define a structured namespace convention for runtime-advisory metadata on ONNX
nodes and models, enabling model exporters to embed deployment hints (placement,
memory tiering, precision, expert scheduling) that any runtime can optionally
consume — without changing the protobuf schema.

## Motivation
[motivation]: #motivation

### The Problem

ONNX models today carry structural information (graph topology, tensor shapes,
data types) and, since IR v11, optional multi-device annotations
(`ShardingSpecProto`, `DeviceConfigurationProto`). However, there is a large
class of **runtime-advisory information** that model exporters know at export
time but have no standardized way to communicate:

1. **Memory tier preference.** A model exporter that profiled activation patterns
   knows which weights are "hot" (accessed every token) vs "cold" (rarely used
   MoE experts). A runtime with tiered storage (VRAM → host RAM → disk) would
   benefit from this information, but today must re-discover it through profiling.

2. **Expert affinity.** MoE models (Mixtral, DeepSeek-V3, Qwen-MoE) have experts
   that are frequently co-activated. Runtimes that cache experts on-device could
   use affinity hints to pre-group co-activated experts, avoiding redundant
   loading.

3. **Precision preferences.** Some layers tolerate FP8 inference better than
   others. An exporter that ran calibration can annotate per-node precision
   recommendations.

4. **Deployment-specific sharding.** The same model may be deployed on 8×H200
   (TP8, NVLink) or 4×Mac Studio (TP4, Thunderbolt). The existing
   `ShardingSpecProto` supports only one configuration per model. Exporters
   need a way to embed multiple deployment profiles.

### Current Workaround

Runtimes today use:
- Custom config files alongside the model (not portable)
- Hard-coded heuristics (not model-specific)
- Runtime profiling (expensive, especially for MoE models with 100s of experts)

### Use Cases

**Use Case 1: MoE Expert Placement.** A model exporter (e.g., Mobius) profiles
expert activation frequency during calibration. It embeds
`ai.onnx.hint.expert.activation_frequency: "0.73"` on each expert subgraph. A
runtime reads these hints to decide which experts to keep resident in VRAM vs
offload to host RAM.

**Use Case 2: Multi-Deployment Sharding.** A model is exported once, but
deployed in three configurations. The exporter embeds generic placement hints
plus specialized overrides:
- Default: `ai.onnx.hint.placement.shard_count: "8"`
- Mac cluster: `ai.onnx.hint.config.mac_cluster.placement.shard_count: "4"`
- Single GPU: `ai.onnx.hint.config.single_gpu.memory.offload_priority: "1"`

**Use Case 3: Mixed Precision.** Calibration shows attention layers lose quality
at FP8 but FFN layers are fine. Exporter annotates:
- Attention nodes: `ai.onnx.hint.compute.precision: "fp16"`
- FFN nodes: `ai.onnx.hint.compute.precision: "fp8"`

## Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

### Convention Overview

Model producers MAY add key-value pairs to the existing `metadata_props` field
on `NodeProto` (or `ModelProto` for model-level hints) using the following
namespace:

```
ai.onnx.hint.{category}.{key}
```

For deployment-specific specializations:

```
ai.onnx.hint.config.{config_name}.{category}.{key}
```

All values are strings. Interpretation is type-specific (documented per key).

### Standard Categories

| Category | Description | Example Keys |
|---|---|---|
| `placement` | Device affinity & parallelism | `device_type`, `shard_axis`, `shard_count`, `pipeline_stage` |
| `memory` | Memory tier & residency | `tier` (hot/warm/cold), `offload_priority`, `pin` |
| `expert` | MoE expert scheduling | `affinity_group`, `activation_frequency`, `prefetch_window` |
| `compute` | Precision & kernel selection | `precision` (fp32/fp16/bf16/fp8/int8), `kernel_hint` |

### Semantics

- **All hints are advisory.** A runtime MUST NOT fail if it cannot honor a hint.
  Hints influence optimization decisions but never change correctness.
- **Unknown hints are ignored.** Runtimes SHOULD silently skip hints in
  categories or keys they do not recognize.
- **Config resolution order:** specialized config hint → generic hint → no hint
  (runtime decides autonomously).

### Example

A MoE model with 64 experts, exported for both data center and Mac cluster:

```
# On expert subgraph nodes 0-15 (high frequency cluster):
metadata_props:
  ai.onnx.hint.expert.affinity_group: "hot_cluster_0"
  ai.onnx.hint.expert.activation_frequency: "0.82"
  ai.onnx.hint.memory.tier: "hot"
  ai.onnx.hint.config.mac_cluster.memory.tier: "hot"

# On expert subgraph nodes 48-63 (rare experts):
metadata_props:
  ai.onnx.hint.expert.affinity_group: "cold_tail"
  ai.onnx.hint.expert.activation_frequency: "0.03"
  ai.onnx.hint.memory.tier: "cold"
  ai.onnx.hint.memory.offload_priority: "9"
```

## Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

### Namespace Grammar

```
hint_key       := "ai.onnx.hint." category "." specific
config_key     := "ai.onnx.hint.config." config_name "." category "." specific
category       := placement | memory | expert | compute | <future>
config_name    := [a-z][a-z0-9_]*    (lowercase, underscore-separated)
specific       := [a-z][a-z0-9_]*    (lowercase, underscore-separated)
```

### Resolution Algorithm

A runtime resolving a hint for a node:

```python
def resolve_hint(node_metadata: dict, key: str, active_config: str | None) -> str | None:
    """Resolve a hint key with optional config specialization."""
    if active_config:
        specialized = f"ai.onnx.hint.config.{active_config}.{key}"
        if specialized in node_metadata:
            return node_metadata[specialized]
    generic = f"ai.onnx.hint.{key}"
    return node_metadata.get(generic)
```

### Interaction with ShardingSpecProto (IR v11+)

The existing `ShardingSpecProto` provides **structural** parallelism information
(how tensor axes are partitioned across devices). The `ai.onnx.hint.*` convention
provides **advisory** information for runtime optimization. They are
complementary:

| Aspect | ShardingSpecProto | ai.onnx.hint.* |
|---|---|---|
| Scope | Tensor partitioning | Broad (memory, scheduling, precision) |
| Authority | Normative (runtime should honor) | Advisory (runtime may ignore) |
| Schema | Protobuf fields | metadata_props (string KV) |
| Multi-config | One per model | Multiple via config.{name} overlay |

A model MAY have both. When both provide placement information, `ShardingSpecProto`
takes precedence for tensor partitioning; `ai.onnx.hint.placement.*` supplements with
device-type preferences, pipeline stage assignments, and other non-sharding hints.

### Standard Key Registry

#### placement

| Key | Type | Description |
|---|---|---|
| `device_type` | string | Preferred device type: `gpu`, `npu`, `cpu` |
| `shard_axis` | int | Tensor axis to shard for tensor parallelism |
| `shard_count` | int | Number of shards |
| `pipeline_stage` | int | Pipeline parallelism stage index |

#### memory

| Key | Type | Description |
|---|---|---|
| `tier` | enum | `hot` (device), `warm` (host), `cold` (disk) |
| `offload_priority` | int | Higher = evict sooner (0 = never evict) |
| `pin` | bool | `true` = suggest pinning in fastest available memory |

#### expert

| Key | Type | Description |
|---|---|---|
| `affinity_group` | string | Group ID for co-activated experts |
| `activation_frequency` | float | [0, 1] probability of activation per token |
| `prefetch_window` | int | Layers ahead to prefetch this expert |

#### compute

| Key | Type | Description |
|---|---|---|
| `precision` | enum | `fp32`, `fp16`, `bf16`, `fp8_e4m3`, `fp8_e5m2`, `int8`, `int4` |
| `kernel_hint` | string | Runtime-specific kernel selection hint |

### Extensibility

The category list is open. Future proposals may add categories (e.g., `energy`,
`latency`, `batch`) by following the same namespace convention. Runtimes MUST
ignore categories they do not recognize.

Runtime-specific hints that are not suitable for standardization SHOULD use a
vendor prefix instead:

```
com.{vendor}.hint.{category}.{key}
```

For example: `com.nxrt.hint.expert.streaming_priority: "2"`

## Drawbacks
[drawbacks]: #drawbacks

1. **metadata_props pollution.** Models with many nodes and many hints could have
   large metadata. Mitigation: hints are sparse (only annotate nodes where the
   hint differs from default behavior).

2. **String typing.** All values are strings, requiring parsing. This is an
   inherent limitation of `metadata_props`. A future protobuf extension could
   add typed hint fields, but the string convention works today without schema
   changes.

3. **Hint staleness.** If a model is re-quantized or modified post-export, hints
   may become stale. Mitigation: tools that modify models should strip or
   regenerate hints.

4. **Fragmentation risk.** Without governance, runtimes might define overlapping
   semantics for the same key. Mitigation: the standard key registry (this RFC)
   defines canonical semantics.

## Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

### Why metadata_props and not new protobuf fields?

- **Zero schema change.** This convention works with all existing ONNX versions
  that support `metadata_props`.
- **Incremental adoption.** Runtimes can start consuming hints without waiting
  for a new IR version.
- **Extensibility.** New hint keys are added by documentation, not protobuf
  evolution.

### Alternative: External sidecar file

Some runtimes use JSON/YAML sidecar files for deployment configuration.

- Pro: No model modification.
- Con: Fragile (file can get separated from model), not portable across tools,
  requires out-of-band coordination.

### Alternative: Extend ShardingSpecProto

Could add memory/precision/expert fields to the existing sharding protobuf.

- Pro: Structured, typed.
- Con: Mixes orthogonal concerns (sharding vs memory vs precision), requires
  IR version bump for each new field, heavyweight for advisory information.

### Impact of not doing this

Each runtime continues inventing its own metadata conventions, leading to:
- Models that only work well on one specific runtime
- Exporters that need runtime-specific plugins for each target
- Redundant runtime profiling that the exporter already did

## Prior art
[prior-art]: #prior-art

### ONNX Multi-Device Proposal (RFC 0006)

The existing multi-device proposal by @kevinch-nv added `ShardingSpecProto` and
`DeviceConfigurationProto` to ONNX IR v11.

| Aspect | RFC 0006 (ShardingSpec) | This RFC (ai.onnx.hint.*) |
|---|---|---|
| **Scope** | Tensor partitioning only | Memory, precision, scheduling, placement |
| **Authority** | Normative — runtime should honor | Advisory — runtime may ignore |
| **Schema** | Protobuf fields (typed) | metadata_props (string KV) |
| **Multi-config** | One sharding per model | Multiple via `config.{name}` overlay |
| **Extensibility** | Requires protobuf change + IR bump | Add keys by documentation |
| **Communication** | Implicit (no explicit collectives) | Same — no explicit collectives |

**Relationship:** Complementary. RFC 0006 answers "how is this tensor split?"
(structural). This RFC answers "what should the runtime *prefer* for this
node?" (advisory). A model may have both.

### Google GSPMD (XLA Compiler)

GSPMD annotates XLA HLO ops with per-tensor sharding specs on a logical device
mesh, then auto-propagates incomplete annotations through the graph via a
compiler pass.

| Aspect | GSPMD | This RFC |
|---|---|---|
| **Granularity** | Per-tensor sharding | Per-node, multi-category |
| **Propagation** | Compiler auto-completes partial annotations | No propagation — each hint is self-contained |
| **Scope** | Parallelism only (shard/replicate) | Parallelism + memory + precision + scheduling |
| **Binding** | Compile-time, enforced | Runtime, advisory |
| **Format** | XLA internal proto (not portable) | ONNX metadata_props (portable) |
| **Multi-config** | One mesh per compilation | Multiple configs in one model |

**Key insight adopted:** GSPMD shows that sparse annotations (annotate a few ops,
propagate the rest) are ergonomic. We adopt the same philosophy — annotate only
where the hint differs from what the runtime would auto-decide.

**Key difference:** GSPMD is a compiler that *enforces* annotations and inserts
communication ops. Our hints are purely advisory — the runtime makes the final
decision and never fails if it cannot honor a hint.

### PyTorch DTensor

DTensor represents distributed tensors with a `(DeviceMesh, [Placement])`
tuple where Placement is `Shard(dim)`, `Replicate`, or `Partial`.

| Aspect | DTensor | This RFC |
|---|---|---|
| **Representation** | Runtime object (in-memory) | Serialized metadata (in model file) |
| **Scope** | Tensor parallelism | Broad (memory, precision, expert, placement) |
| **Lifecycle** | Created at runtime, used for dispatch | Created at export, consumed at load |
| **Portability** | PyTorch-only | Any ONNX runtime |
| **Mesh concept** | Explicit n-d mesh | No mesh — hints are device-agnostic |
| **Communication** | Explicit (all_reduce inserted automatically) | No communication semantics |

**Key insight adopted:** DTensor's Placement vocabulary (`Shard(dim)`,
`Replicate`) is clean and minimal. Our `ai.onnx.hint.placement.shard_axis` /
`shard_count` is the serialized equivalent for ONNX.

**Key difference:** DTensor is a runtime dispatch mechanism that changes
semantics. Our hints are metadata that do not change graph semantics.

### TensorRT Optimization Profiles

TensorRT serializes per-layer precision (`setPrecision(FP16)`), workspace sizes,
tactic selection caches, and dynamic shape profiles into engine plans.

| Aspect | TensorRT | This RFC |
|---|---|---|
| **Portability** | NVIDIA-only, hardware-specific engine plan | Portable across runtimes |
| **Granularity** | Per-layer precision + kernel tactic | Per-node multi-category |
| **Binding** | Baked into compiled engine | Advisory, runtime decides |
| **Dynamic shapes** | Optimization profiles (min/opt/max) | Not addressed (orthogonal) |
| **Source** | Auto-tuning during build | Calibration data at export |
| **Extensibility** | Closed (NVIDIA internal) | Open namespace |

**Key insight adopted:** TensorRT shows that per-layer precision hints have real
value — some layers tolerate INT8, others need FP16. Our
`ai.onnx.hint.compute.precision` brings this to the portable ONNX level.

**Key difference:** TensorRT profiles are *results* of hardware-specific
auto-tuning (not portable). Our hints are *inputs* to any runtime's optimization
(portable). A TensorRT build could consume `ai.onnx.hint.compute.precision` as
initial precision constraints, then run its own tactic search within those bounds.

### SafeTensors Metadata

SafeTensors files include a JSON header with a `__metadata__` dict containing
arbitrary string key-value pairs. In practice, models on HuggingFace Hub use
this for:
- `format`: framework origin (`"pt"`, `"tf"`)
- `total_size`: aggregate model size
- Quantization info (ad-hoc, no standard)

| Aspect | SafeTensors __metadata__ | This RFC |
|---|---|---|
| **Schema** | Completely unstructured JSON dict | Structured namespace convention |
| **Scope** | File-level only | Model-level + per-node |
| **Standard keys** | None (organic/ad-hoc) | Registered categories + keys |
| **Governance** | Community convention | RFC-governed registry |
| **Runtime use** | Rarely consumed for optimization | Designed for runtime consumption |

**Lesson learned:** SafeTensors demonstrates that even a minimal metadata
mechanism gets widely adopted (every HuggingFace model uses `__metadata__`).
But without namespace structure, it devolves into fragmented ad-hoc keys.
Our proposal avoids this with a governed namespace hierarchy.

### vLLM / SGLang Deployment Configuration

vLLM uses CLI args and `config.json` fields for deployment decisions:
`--tensor-parallel-size 8`, `--quantization awq`, `--max-model-len 4096`,
`--kv-cache-dtype fp8`.

| Aspect | vLLM config | This RFC |
|---|---|---|
| **Location** | External (CLI / env / JSON) | Embedded in model |
| **Portability** | Runtime-specific flags | Cross-runtime |
| **Granularity** | Global (whole model) | Per-node |
| **Multi-config** | Separate launch commands | Single model, multiple configs |
| **Per-layer precision** | Not supported | Supported via `compute.precision` |

**Key insight:** vLLM's approach works for a single runtime at a single
deployment target. But when the same model is served by ORT, TensorRT, and
vLLM on different hardware, having hints *in the model* eliminates configuration
duplication and drift.

**Key difference:** vLLM configs are imperative ("use TP=8"). Our hints are
advisory ("this model works well with TP=8 on this config"). A runtime can
ignore hints that don't match its capabilities.

### exo (Distributed Consumer Inference)

exo discovers peers via UDP broadcast on the local network and assigns pipeline
stages based on detected device capabilities (memory, FLOPS).

| Aspect | exo | This RFC |
|---|---|---|
| **Placement decisions** | Runtime-only (automatic) | Model can suggest (hints) |
| **Hardware awareness** | Probes at startup | Embedded in model per-config |
| **MoE support** | None | Expert affinity + frequency hints |
| **Security** | None (open broadcast) | N/A (model metadata, not networking) |

**Key difference:** exo makes all placement decisions at runtime with no
model input. For simple pipeline parallelism this works, but for MoE expert
placement or mixed-precision per-layer decisions, runtime-only heuristics
cannot match the quality of exporter-time profiling data embedded as hints.

## Unresolved questions
[unresolved-questions]: #unresolved-questions

1. **Governance for new standard keys.** Should new keys require an RFC, or is a
   lighter-weight process (e.g., PR to a registry doc) sufficient?

2. **Model-level vs node-level hints.** Should `ai.onnx.hint.*` on `ModelProto`
   metadata serve as defaults that node-level hints override? This RFC assumes
   yes, but the override semantics need specification.

3. **Hint validation in checker.** Should the ONNX checker validate hint key
   syntax (namespace structure) without validating values? Or should hints be
   fully opaque to the checker?

4. **Interaction with graph optimization.** When a graph optimizer fuses or splits
   nodes, how should hints be propagated? Options: drop, merge (intersection),
   inherit from dominant operand.

## Future possibilities
[future-possibilities]: #future-possibilities

1. **Typed hint extension.** A future IR version could add an optional
   `HintProto` field to `NodeProto` with typed fields for the most commonly
   used hints, while keeping `metadata_props` as the extensibility mechanism.

2. **Hint generators.** Standard tools that profile a model on reference hardware
   and annotate it with `ai.onnx.hint.*` keys. This could be part of the ONNX
   ecosystem tooling.

3. **Cross-model hints.** For multi-model pipelines (e.g., speculative decoding
   with draft + target model), hints about inter-model relationships.

4. **Energy/latency hints.** Categories for power-constrained deployment:
   `ai.onnx.hint.energy.budget_watts: "15"` or latency SLO hints.

5. **Formal hint inheritance spec.** Define how hints propagate through function
   expansion, subgraph inlining, and graph optimization passes.
