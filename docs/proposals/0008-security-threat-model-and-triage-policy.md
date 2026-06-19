<!--
Copyright (c) ONNX Project Contributors
-->

<!--- SPDX-License-Identifier: Apache-2.0 -->
- Feature Name: `security_threat_model`
- Start Date: 2026-06-11
- RFC PR: [onnx/onnx#8081](https://github.com/onnx/onnx/pull/8081)
- Status: under discussion
- Authors: titaiwangms

## Summary
[summary]: #summary

ONNX **already has** a threat model — `docs/AssuranceCase.md` §1 (v1.0 *(DRAFT)*): a T1–T6 threat table
(malicious-model code execution, deserialization memory corruption, resource-exhaustion DoS,
supply-chain compromise, information disclosure, compromised packages) plus a trust-boundary
diagram — **and** a published disclosure process (`SECURITY.md`). The gap is that **neither is
dispute-deciding**: for a given advisory they do not settle whether a finding is **in scope to
fix**, **how severe** it is, or **whether it warrants embargo**, so advisories are scoped, rated,
and disclosed ad hoc and equally competent maintainers reach differing conclusions. The existing
threat model also **over-claims assurances** that day-to-day triage and shipped fixes contradict:
`docs/AssuranceCase.md` §1.2 Boundary #1 advertises "size limits" at the untrusted→core boundary,
and §3 maps integer overflow (CWE-190) to "checked size arithmetic in tensor allocation" — yet
GHSA-538c-55jv-c5g9 (CWE-400 over-allocation) had to SHIP the external-data offset/length bound
those assurances implied already existed.

This RFC asks maintainers to **ratify and upgrade** that threat model into one that **decides cases
by rule** — a **component × primitive scope taxonomy**, a CVSS v3.1-anchored severity rubric, and a
public-by-default disclosure stance — and to **reconcile the over-claims to a best-effort stance**.
This ratified model is **Layer 0** (the normative layer that requires a maintainer vote). Once
ratified, it is recorded in a single citable home — this RFC **recommends** a clearly-titled "Threat
model" section **extending `SECURITY.md`** (fewer root files; it reconciles with the existing
`SECURITY.md`, `docs/AssuranceCase.md`, and `INCIDENT_RESPONSE.md`), with a dedicated
`THREAT_MODEL.md` as the alternative (see Unresolved questions). The proposed verbatim document
edits, and an operational triage guide (**Layer 1**, derived from Layer 0), follow in a follow-up PR.

## The decision we're asking for
[the-decision-we-are-asking-for]: #the-decision-we-are-asking-for

This RFC is a **decision request**, not a code change. The substance is three yes/no questions
for maintainers to ratify; everything else is the rationale a "yes" would adopt.

1. **Reference-runtime execution scope.** Do we ratify that the ONNX **reference runtime is not
   intended for untrusted input** and is therefore out of scope for execution-safety
   vulnerabilities (treated as documented limitations / public robustness bugs)?

2. **Resource-exhaustion vs. memory-safety.** Do we ratify the **behavior-preserving-bound rule**
   (stated in full under [Scope rule for resource-exhaustion](#scope-rule-for-resource-exhaustion-the-behavior-preserving-bound-rule))
   — a resource-exhaustion is in scope **only if a constant number of cheap, behavior-preserving O(1)
   bounds** (rejecting only impossible / malformed inputs, never legitimate large ones) remove the
   unbounded growth, putting **size-driven over-allocation in scope** (precedent GHSA-538c-55jv-c5g9)
   and **ReDoS / quadratic compute out** — together with the in-scope memory-safety corruption (OOB
   write, or info-leaking OOB read)? A crash *claimed* as memory-corruption is in scope only on
   **reporter-supplied** evidence (ASan/Valgrind trace, write-primitive PoC, or a reproducer run
   under ONNX's existing ASan/UBSan CI); absent that it is tracked as a suspected-corruption or
   out/low provisional state (auditable, reopenable), never silently closed.

3. **Disclosure & workflow.** Do we ratify that **tool-found, easily reproduced findings without a
   demonstrated practical exploit are handled publicly** (normal one-advisory-per-PR, no embargo),
   with embargo reserved for the two narrow triggers below — a demonstrated practical exploit
   (regardless of how easily it reproduces), or a non-obvious finding plausibly exploitable beyond
   a crash — and with maintainers holding **cited authority to downgrade or close** an advisory by
   referencing a specific clause?

A "yes" to all three gives maintainers a citable basis for consistent triage.

## Motivation
[motivation]: #motivation

ONNX's existing threat model (`docs/AssuranceCase.md` §1, T1–T6) identifies *what can go wrong* but
not *how the project decides what to do about a given report*, and in places **over-claims
assurances** that triage and shipped fixes (e.g. GHSA-538c-55jv-c5g9) contradict. Because the
boundary has never been ratified into a decision procedure, three independent questions get
collapsed into a single ad hoc judgment for every advisory:

- **Is this even something ONNX is responsible for fixing?** (scope)
- **How severe is it?** (severity)
- **Must it be embargoed, or handled in public?** (disclosure)

When these three are decided together and case-by-case, outcomes are inconsistent: robustness
improvements get rated "critical," routine tool-found issues get embargoed, and reviewers reach
opposite conclusions on the same advisory. The need is **governance, not process tooling**: a
ratified, *decidable* statement of what ONNX defends against, so most contested cases are resolved
by definition rather than debate and any remaining triage is a rule application a maintainer can
cite. This RFC deliberately does **not** re-litigate any specific past advisory; prior reports are
referenced only as evidence that a shared, operational boundary is missing.

## Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

The model rests on a single principle: **three orthogonal axes, decided independently.**

- **TIER** — in scope to *fix*? (a component × primitive lookup, not a judgment)
- **SEVERITY** — how bad? (CVSS + reachability)
- **DISCLOSURE** — public or embargo? (a demonstrated practical exploit, **or** a non-obvious +
  plausibly-exploitable finding)

Collapsing these three ("touches memory → critical → embargoed") is the engine of every inflated
advisory; keeping them orthogonal is what makes the policy both inflation-proof and consistently
decidable. The reference-level section states the scope taxonomy (tier table + the behavior-preserving-bound rule), the
severity baseline, and the disclosure principle.

## Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

### Trust tiers — component × primitive (with repo entry points)

| Tier | Component | Repo entry point(s) | In scope? (by primitive) |
|---|---|---|---|
| **1** | C++ checker | `onnx/checker.cc` / `onnx/checker.h` | OOB-**write** → in. OOB-**read** → in only if it leaks info. Pure crash (SIGSEGV/abort without corruption) → out. A crash *claimed* as memory-corruption needs reporter evidence (ASan/Valgrind trace, write-primitive PoC, or a reproducer run under the existing ASan/UBSan CI); absent that it is tracked as an auditable, reopenable provisional state, not closed. Bypass (malformed model accepted) → rated by downstream reachability, typically low. |
| **1** | Shape inference | `onnx/defs/**/*.cc` (`*ShapeInference*`), `onnx/shape_inference/` | Same primitive rule. Native OOB-write / overflow-with-corruption on a crafted graph = in; bare SIGSEGV = out. |
| **1** | Version converter | `onnx/version_converter/` | Same. OOB-**write** (e.g. heap overflow during conversion) = in, high priority. |
| **2** | External-data / path handling | `onnx/checker.cc`, `onnx/external_data_helper.py`, `onnx/model_container.py` | **In** — filesystem escape (read/write/exfil), symlink/hardlink/TOCTOU, path traversal. (Partially hardened today; see `docs/Security.md`.) |
| **3** | Reference evaluator | `onnx/reference/reference_evaluator.py`, `onnx/reference/ops/**` | **Out entirely** — not intended for untrusted input. Crash / RCE claims → documented limitation. |
| **4** | Resource-exhaustion (classify by **primitive, not component**) | cross-cutting (recursion / nesting-depth, over-allocation, regex, quadratic passes) | Defers to **the behavior-preserving-bound rule** below. |

*(The former model-hub component (`onnx/hub.py`) was removed in #7678 and is not part of this
taxonomy; the reference evaluator is the sole Tier-3 example.)*

### Scope rule for resource-exhaustion (the behavior-preserving-bound rule)

> **The behavior-preserving-bound rule (resource-exhaustion scope).** A resource-exhaustion / DoS is **in scope if and only if a constant number of cheap, behavior-preserving O(1) bounds — rejecting only impossible or malformed inputs (e.g. a declared length exceeding the bytes that exist), never legitimate large inputs — removes the unbounded growth.** By this rule, **size-driven over-allocation is in scope** (precedent: the shipped external-data bounds fix, GHSA-538c-55jv-c5g9 — a constant set of four O(1) field-checks). The "constant number" is counted per parsed object/field, before any attacker-amplified allocation — not across the whole model: validating N external-data records with the same four per-record O(1) checks is still a constant set, so a per-object over-allocation cannot be reclassified "O(n), therefore out." **Attacker-controlled-depth recursion / stack overflow is in scope only when its depth bound is physically grounded in the message's own byte size**; a configurable depth cap (e.g. protobuf `SetRecursionLimit`, default 100) is a defense knob, not behavior-preserving, so whether a bare depth cap qualifies is a **maintainer-ratification call**. **Out of scope:** resource-exhaustion closable only by a global cap that rejects legitimate large models (ReDoS, quadratic / exponential compute).

This is the only scope rule stated inline; the full numbered clause set (Clauses 1, 3, and 4) that operationalizes
the model lands in the follow-up PR.

**Why the rule is drawn here (defense of Clause 2).** This is the part that needs ratification,
so the reasoning behind each boundary is set out here rather than in the landing docs:

- **The strongest in-scope case is grounded in the input itself.** The clearest "in" is a bound
  *physically determined by the input* — a declared length versus the bytes that actually exist —
  where "impossible" is decided by the file, not by policy. That is exactly the shape of
  GHSA-538c-55jv-c5g9's shipped fix, which is why naming size-driven over-allocation as an *in*
  primitive is load-bearing: without it, a plain "classify by primitive" routes ONNX's own shipped
  fix to "DoS → out" and **contradicts the project's own security history**. The named primitive,
  not the component, carries the decision.
- **A configurable cap is a defense knob, not a behavior-preserving bound.** A tunable recursion
  cap (e.g. protobuf's `SetRecursionLimit`, default 100) is a hardening knob; exceeding it is not
  by itself a security bug. A bare `MAX_DEPTH` chosen by fiat can reject a legitimately deep
  machine-generated graph, so it is not automatically behavior-preserving — hence a depth cap's
  status is a maintainer-ratification call rather than an automatic "in."
- **Anti-gaming: the global-cap argument is closed in both directions.** A reporter cannot argue
  "one global `MAX_NODES` cap is an O(1) bound, so my quadratic blowup is in" — a global cap
  rejects legitimate large models, so it is not behavior-preserving and does not qualify; conversely
  ReDoS stays out for the same reason. `memory-safety-adjacency` is the *rationale* for where the
  line sits, not a second required gate.
- **Dependency policy is separate from the scope test.** ONNX also declines to add a new
  third-party dependency solely for DoS hardening; this is a policy stance, not the reason ReDoS is
  out of scope. It is stated here only to keep the two from being conflated.
- **Clean sanitizer runs are not dispositive (evidentiary limit).** Even under the existing ASan/UBSan
  CI, a clean run does not prove benignity — it only fails to confirm on the executed path, and
  ASan/UBSan in particular miss in-allocation OOB reads and uninitialized-memory leaks (which need
  targeted review or MSan). This is why the burden of corruption evidence rests on the reporter and
  why undetermined memory bugs are tracked as reopenable provisional states rather than closed.

### Severity baseline

> **Severity.** Findings are scored with **CVSS v3.1**, anchored **per-primitive** with an **`AV:L` baseline** (an ONNX model is local input the victim opens, not a network service); conservative by default, escalate on demonstrated reachability/exploit. The full per-primitive rubric (bands, vectors, worked rows) lives in the follow-up PR.

Severity is **independent of tier and disclosure**, and **reachability is a within-band modifier,
never a scope gate** — a malformed-model-only finding sits at the low end of its primitive's band, a
benign-production-model finding at the high end, but reachability never moves a finding into or out
of scope and never decides public vs embargo.

### Disclosure principle

Disclosure is **public by default**. Embargo applies when **either** (a) a **demonstrated practical
exploit** exists — a working PoC for remote code execution, privilege escalation, memory corruption
with control-flow hijack or information leak, or sandbox / filesystem escape — **regardless of how
easily it reproduces**; **or** (b) a finding is **non-obvious** (not reproducible with widely
available tooling) **and** plausibly exploitable beyond a crash. A tool-found, easily reproduced
out-of-bounds *crash* with no demonstrated exploit is therefore **public**; a finding carrying a
working RCE / control-flow-hijack PoC is **embargoed** until a fix ships. Disclosure is decided
**independently** of tier and severity.

### Implementing text

The proposed verbatim document edits implementing these decisions — the full `SECURITY.md`
boundary statement and supporting clauses, the "Reporting a Vulnerability" replacement, the
complete severity rubric table, the operational workflow, the one-time backlog re-classification
pass, the `INCIDENT_RESPONSE.md` deltas, the publish-or-dismiss SLA, the AI-assisted triage
guardrails, and the Layer-1 triage guide — are in a **separate follow-up PR (applied only after
ratification)**.

## Drawbacks
[drawbacks]: #drawbacks

- Declaring components out of scope (the reference runtime, algorithmic resource-exhaustion) could
  be read as ONNX "not caring" about robustness. Mitigation: out-of-scope-as-a-*vulnerability*
  still allows fixing issues as ordinary public robustness bugs; the model changes the *channel and
  severity*, not whether bugs get fixed.
- A ratified boundary is a commitment that is harder to change later than ad hoc practice — this is
  intentional (the predictability is the benefit), but it raises the bar for getting the wording
  right now, which is why ratification (not assertion) is requested.
- The taxonomy adds vocabulary maintainers must learn. Mitigation: the entire decision reduces to
  one table lookup plus the behavior-preserving-bound rule; the worked example in the follow-up PR
  shows it end to end.
- Conservative severity scoring (capping a not-yet-RCE OOB-write at High) risks later criticism if
  such a write is shown RCE-able. Mitigation: the bands are **maximum-justifiable** caps with
  explicit **re-scoring to Critical on RCE / CFH evidence** — restrained-by-default,
  escalated-on-proof, never permanently low.

## Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

**Why this design.** The recurring problem is *definitional*, not procedural: maintainers have
disagreed about whether a given report is even a vulnerability. A ratified threat model is the only
artifact that resolves that disagreement; a procedural checklist built on an unratified boundary
would simply relocate the same argument into the checklist's review.

**Alternatives considered:**

- **Triage-guide only (no ratified boundary).** Insufficient — a checklist has no normative anchor,
  so "is this in scope?" stays unsettled and the guide re-litigates severity ad hoc.
- **Do nothing / per-advisory ad hoc.** The status quo that produced inconsistent outcomes and
  repeated maintainer disagreement; it does not scale.
- **Full automation (bot-driven triage).** A bot is only as good as the boundary it encodes;
  automation must be *subordinate* to a ratified human stance, never a substitute for it.
- **Copy TensorFlow / PyTorch wholesale.** Their model treats an untrusted model as
  code-execution-equivalent — out of scope to defend, protection reserved for benign-model-reachable
  corruption. ONNX differs: its checker / shape-inference / version-converter are advertised
  validation tooling **run on untrusted models**, so Tier-1 memory-safety is in scope for ONNX even
  though the analogous runtime crash is not — a divergence this RFC makes explicit rather than
  inheriting.

## Unresolved questions
[unresolved-questions]: #unresolved-questions

To be resolved through RFC discussion (these are maintainer decisions, not RFC-author assertions):

- **Resolved (recorded here for traceability):** the **CVSS version is v3.1** (per maintainer
  andife) — the severity rubric is anchored on v3.1, not v4.0.
- **Tier classification of each component** (the table above) — especially the Tier-1 memory-safety
  carve-out and the cheap-bound discriminator that routes some resource-exhaustion in scope.
- **Scope gate for malformed-model-only memory issues.** This RFC's **chosen (stricter) position**
  is *primitive-based scope* (an OOB write / info-leaking OOB read is in scope regardless of
  reachability, with malformed-only reachability handled as a severity reducer). The **discussable
  looser compromise** is TensorFlow's gate — benign-production-model reachability as a *necessary
  condition* for scope. Maintainers ratify which.
- **The behavior-preserving-bound rule** — which exhaustion cases it genuinely
  closes, and specifically **whether any given recursion / nesting-depth cap qualifies** (a
  byte-size-grounded depth bound is behavior-preserving; a bare policy `MAX_DEPTH` is a judgment
  call).
- **The evidence burden and the provisional (auditable, reopenable) disposition** — confirming the
  reporter carries the burden of establishing corruption, and that an unproven crash is tracked
  out/low, not closed. (No new CI job is proposed; the Ubuntu debug leg already runs ASan + UBSan.)
- **The exact embargo trigger list**, and **appeal / escalation authority** — final call
  when a reporter disputes a downgrade (proposed: infra SIG → Steering Committee).
- **The publish-or-dismiss window** — the fixed number of days (e.g. 90 / 120 / 180, best-effort
  90-day target after PyTorch) within which an embargoed advisory must be published or dismissed,
  **and the bounded embargo-extension cap** beyond which a still-unfixed in-scope advisory escalates
  rather than extends again.
- **Home of the threat model** — a clearly-titled section extending `SECURITY.md` (recommended) vs.
  a dedicated top-level `THREAT_MODEL.md` (PyTorch / TensorFlow precedent). Either way the boundary
  statement is written to be quotable as one anchor-able unit, and `docs/Security.md` /
  `docs/AssuranceCase.md` are cross-linked as subordinate so the repository does not carry
  overlapping security postures.
