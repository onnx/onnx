<!--
Copyright (c) ONNX Project Contributors
-->

<!--- SPDX-License-Identifier: Apache-2.0 -->
- Feature Name: `security_threat_model`
- Start Date: 2026-06-11
- RFC PR: [onnx/onnx#0000](https://github.com/onnx/onnx/pull/0000)
- Status: under discussion
- Authors: titaiwangms

## Summary
[summary]: #summary

ONNX has a published disclosure process (`SECURITY.md`) but no **ratified threat
model**: there is no agreed statement of what ONNX defends against, which components
are a security boundary, how severity is assigned, or when a finding warrants embargo.
As a result, incoming advisories are scoped, rated, and disclosed ad hoc, and equally
competent maintainers have reached differing conclusions on the same report. This RFC
proposes a threat model — a **component × primitive scope taxonomy**, a CVSS-anchored
severity rubric, and a public-by-default disclosure stance — for maintainers to ratify.
This ratified model is **Layer 0** (the normative layer that requires a maintainer vote).
Once ratified, it is recorded in a single citable home. This RFC **recommends** a
clearly-titled "Threat model" section **extending `SECURITY.md`** (fewer root files; it
reconciles directly with the existing `SECURITY.md` and `INCIDENT_RESPONSE.md`), with the
existing `docs/Security.md` retained as a **sub-component design document that inherits** the
global tiers rather than a parallel policy; a dedicated `THREAT_MODEL.md` remains the
alternative if maintainers prefer it (see Unresolved questions). An operational triage guide
(**Layer 1** — derived from Layer 0, consistency only, no separate vote) follows separately.

## The decision we're asking for
[the-decision-we-are-asking-for]: #the-decision-we-are-asking-for

This RFC is a **decision request**, not a code change. The substance is three yes/no
questions for maintainers to ratify. Everything else in this document is the rationale
and the proposed text that a "yes" would adopt.

1. **Reference-runtime execution scope.** Do we ratify that the ONNX **reference runtime
   is not intended for untrusted input** and is therefore out of scope for execution-safety
   vulnerabilities (treated as documented limitations / public robustness bugs)?

2. **Resource-exhaustion vs. memory-safety.** Do we ratify the **Clause-2 behavior-preserving-bound
   discriminator** — a resource-exhaustion is in scope **if and only if a *constant number* of
   cheap, *behavior-preserving* O(1) bounds** (rejecting only impossible / malformed inputs,
   e.g. a declared length exceeding the bytes available, never legitimate large inputs) remove
   the unbounded growth — covering size-driven over-allocation (precedent: the shipped
   external-data bounds fix, GHSA-538c-55jv-c5g9) and attacker-controlled-depth recursion /
   stack overflow, **while a resource-exhaustion closable only by a global cap that rejects
   legitimate large models (ReDoS, quadratic / exponential compute) is out of scope**, alongside
   the in-scope memory-safety corruption (out-of-bounds
   write, or info-leaking out-of-bounds read)? A crash *claimed* as memory-corruption is in
   scope only on **reporter-supplied** evidence (ASan/Valgrind trace, write-primitive PoC, or a
   reproducer run under ONNX's existing ASan/UBSan CI); absent that it is tracked as
   `provisional_in` (corruption suspected, sanitizer CI pending) or `provisional_dos_out` (out
   / low priority, auditable, reopenable), never silently closed.

3. **Disclosure & workflow.** Do we ratify that **tool-found, easily reproduced findings
   are handled publicly** (normal one-advisory-per-PR, no embargo), with embargo reserved
   for the narrow practical-exploit triggers listed below, and with maintainers holding
   **cited authority to downgrade or close** an advisory by referencing a specific clause?

A "yes" to all three gives maintainers a citable basis for consistent triage. Applied to the
current open backlog as a worked example (see [Appendix: backlog dry-run](#appendix-backlog-dry-run)),
the model routes the large majority of advisories into policy-defined dispositions — cleared
to public robustness bugs, in-scope fixes, or an auditable provisional state — leaving only a
small residue (attacker-controlled recursion / nesting-depth) to a genuine maintainer
judgment call.

## Motivation
[motivation]: #motivation

Security advisories arrive for ONNX with no shared definition of scope. Because the
project has never ratified a security boundary, three independent questions get collapsed
into a single ad hoc judgment for every report:

- **Is this even something ONNX is responsible for fixing?** (scope)
- **How severe is it?** (severity)
- **Must it be embargoed, or handled in public?** (disclosure)

When these three are decided together and case-by-case, outcomes are inconsistent:
findings that are robustness improvements get rated "critical," routine tool-found issues
get embargoed, and reviewers reach opposite conclusions on the same advisory. The cost
falls on maintainers, who must re-argue the project's security philosophy on every report,
and on reporters, who receive inconsistent severity and disclosure decisions.

The need is **governance, not process tooling**: a ratified statement of what ONNX
defends against. With that statement in place, most contested cases are resolved by
definition rather than by debate, and any remaining triage is a rule application a
maintainer (or a contributor) can cite.

This RFC deliberately does **not** re-litigate any specific past advisory or pull request.
Prior reports are referenced only as evidence that a shared boundary is missing.

## Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

If accepted, ONNX will have a threat model built on **three axes that are decided
independently**. Keeping them separate is the core idea: collapsing them is what produces
inflated and inconsistently-disclosed advisories.

> **The load-bearing principle:** *Being in scope to fix does NOT imply high severity, and
> neither implies embargo — tier, severity, and disclosure are decided independently.*

### TIER — is it in scope to fix? (a lookup, not a judgment)

Tier is determined by **which component** the finding is in, then **which primitive**
(root cause) it exhibits. It is a table lookup, not an opinion.

| Tier | Components | Stance |
|---|---|---|
| **Tier-1** | checker / shape-inference / version-converter | Memory-corruption **in scope** (see primitive rule below) |
| **Tier-2** | path / external-data handling | In scope — filesystem-escape primitives (traversal, symlink/hardlink/TOCTOU) |
| **Tier-3** | reference runtime | **Out of scope** — not a boundary; do not run untrusted input |
| **Tier-4** | resource-exhaustion / DoS — **classify by primitive, not component** | Defers to the **Clause-2 behavior-preserving-bound discriminator**: size-driven over-allocation (GHSA-538c-55jv-c5g9) is **in**, and attacker-controlled-depth recursion is **in** only when its bound is physically grounded in message size (a bare depth cap is a maintainer-judgment call — see Clause 2); ReDoS / quadratic / exponential compute is **out**. |

*(This is the at-a-glance view; the authoritative component × primitive table with repo
entry points is in the [reference-level explanation](#reference-level-explanation).)*

Within the Tier-1 / Tier-2 validation tooling, scope is a **per-primitive partition**:

- **In scope (memory corruption):** corruption in ONNX's own code while loading / checking /
  converting — an **out-of-bounds write or undefined-behavior overflow always**, an
  **out-of-bounds read only if it leaks information**.
- **In scope (resource-exhaustion the Clause-2 discriminator admits):** size-driven
  over-allocation (precedent: the shipped external-data bounds fix, GHSA-538c-55jv-c5g9) and
  attacker-controlled-depth recursion / stack overflow, where a constant set of cheap,
  **behavior-preserving** O(1) bounds — rejecting only impossible / malformed inputs, never
  legitimate large ones — removes the unbounded growth. The *memory-safety-adjacency* is the
  **rationale** for where the line sits, not a second gate.
- **Out (consumer's contract):** a validation **bypass** (a malformed model is *accepted*)
  that is harmful only if a downstream consumer blindly trusts checker output — rated by
  downstream reachability, **usually low**.
- **Out (non-goal):** a resource-exhaustion the Clause-2 discriminator excludes — ReDoS,
  quadratic / exponential compute — **and** a pure crash with no underlying memory-safety
  violation.

> **The sole scope gate is the Clause-2 behavior-preserving-bound discriminator** (see
> [Supporting clauses](#reference-level-explanation)) — stated in full once there to avoid
> drift; the guide-level summary above defers to it.

> **Classify by root-cause primitive, not observed symptom.** A crash shown to be an
> out-of-bounds read/write is memory-corruption (in scope) even if the only demonstrated
> effect is a crash. A "pure crash" (out of scope) means there is **no** underlying
> memory-safety violation — a null dereference, uncaught exception, or assertion/abort. The
> burden to establish corruption rests on the **reporter** (an ASan/Valgrind trace, a
> write-primitive PoC, or a minimal reproducer the triager runs under ONNX's ASan/UBSan CI);
> a crash with neither evidence nor a reproducer is tracked as `provisional_dos_out` — never
> silently closed. See Clause 4.

### SEVERITY — how bad? (CVSS + reachability)

Severity is assigned from the primitive and its reachability using a CVSS-anchored rubric
(see [reference-level-explanation](#reference-level-explanation)). It is **independent of
tier**: a Tier-1 in-scope finding is *not* automatically high or critical.

**Reachability is a severity *modifier*, not a scope gate** — this is the deliberate
divergence from TensorFlow, whose security-relevant set is gated on *benign-model
reachability* (a "production-grade, benign model"). Here a finding reachable only through a
**malformed** model sits at the **low end** of its primitive's severity band; reachability
through a **benign, production-grade** model sits at the **high end**. So a controllable
OOB-write reached via a malformed model is **Tier-1, in scope to fix**, and is *rated* by
reachability rather than excluded by it. **Disclosure is decided separately**, by Clause 3
(see below) — reachability moves severity *inside* a band, it does not decide public vs
embargo. This sidesteps TensorFlow's benign-model scope gate, whose "production-grade, benign
model" test is inherently judgment-laden, by demoting reachability to a severity input.

### DISCLOSURE — public or embargo? (public is the rebuttable default)

Disclosure is **public by default**. Embargo applies when **either** (a) a **demonstrated
practical exploit** exists — a working PoC for remote code execution, privilege escalation,
memory corruption with control-flow hijack or information leak, or sandbox / filesystem
escape — **regardless of how easily it reproduces**; **or** (b) a finding is **non-obvious**
(not reproducible with widely available tooling) **and** plausibly exploitable beyond a
crash. A tool-found, easily reproduced out-of-bounds *crash* with no demonstrated exploit is
therefore **public**; but a tool-found, easily reproduced finding carrying a **working RCE or
control-flow-hijack PoC is embargoed** until a fix ships — ease of reproduction increases
attacker utility and must not force premature disclosure of a proven exploit.

### Why this helps

A maintainer can triage a typical advisory by walking three short steps and **cite the
rule for each**. The same inputs yield the same answer whether a human or an automated
assistant performs the triage — and a maintainer can **downgrade or close** an inflated
report by citing the specific clause, rather than arguing philosophy case by case.

## Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

This section specifies the proposed text. If the RFC is accepted, the clauses below are
added to `SECURITY.md` verbatim (the boundary statement, Clauses 1–4, and the
downstream-relations policy); the tier table,
severity rubric, and workflow inform the derived Layer-1 triage guide.

### The threat model — proposed `SECURITY.md` boundary statement
[the-threat-model]: #the-threat-model

This boundary statement **is the threat model**: a single, self-contained, quotable unit.
Whichever home maintainers choose for it (see Unresolved questions — dedicated
`THREAT_MODEL.md` vs. a titled section of `SECURITY.md`), it is designed to be cited as one
anchor-able block.

> **Boundary.** For ONNX's validation tooling (checker, shape-inference, version-converter):
> a memory-corruption **OOB-WRITE** is in scope and will be fixed; an **OOB-READ** is in
> scope only if it produces an information leak; a validation **BYPASS** (malformed input
> accepted) is rated by downstream reachability, typically low; a **resource-exhaustion** is
> in scope **if and only if a constant number of cheap, behavior-preserving O(1) bounds**
> (rejecting only impossible / malformed inputs, never legitimate large inputs) remove the
> unbounded growth — covering size-driven over-allocation (precedent: the shipped external-data
> bounds fix, GHSA-538c-55jv-c5g9) and attacker-controlled-depth recursion / stack overflow; a
> resource-exhaustion closable only by a global cap that rejects legitimate large models
> (ReDoS, quadratic / exponential compute) and a **pure crash** (SIGSEGV/abort without memory
> corruption) are out of scope. The
> **reference runtime is not intended for untrusted input and is out of scope entirely.**

### Supporting clauses (proposed `SECURITY.md` text)

> **Clause 1 — best-effort validators; neither a correctness nor a security guarantee.**
> The ONNX checker, shape-inference, and version-converter are best-effort **structural
> validators**, not a guarantee of either model **validity** or **safety**. Passing them is
> **not** a correctness guarantee and **not** a security guarantee, and must **not**
> substitute for sandboxing untrusted inputs. Downstream consumers MUST NOT assume a passing
> model is 100% correct or safe, and MUST validate structural fields (dims, indices, offsets)
> they rely on — e.g. iterate buffers rather than indexing by a dims-derived count. This is a
> non-guarantee, **not** a narrowing of Tier-1 scope: memory-corruption in the checker /
> shape-inference / version-converter themselves remains in scope and will be fixed. (Any
> repository documentation describing this validation as "comprehensive" or "complete
> mediation," or as fully mitigating memory-safety / integer-overflow weakness classes — e.g.
> `docs/AssuranceCase.md` — is to be reconciled to this best-effort statement: triage
> experience and published advisories show these are not guarantees, and the repository must
> not advertise an assurance the threat model explicitly disclaims.)

> **Clause 2 — resource-exhaustion (one decidable gate: the behavior-preserving-bound
> discriminator).** A resource-exhaustion / DoS is **in scope if and only if a *constant
> number* of O(1) bounds** — a small fixed set of size / length / offset / count /
> nesting-depth comparisons, or one constant allocation / recursion-depth cap — that are
> **cheap *and* behavior-preserving** remove the unbounded growth. "Behavior-preserving" means
> the bound rejects **only impossible / malformed inputs** (e.g. a declared `length` exceeding
> the bytes actually available), **never legitimate large inputs**. **In** (named primitives):
> size-driven **over-allocation** — precedent GHSA-538c-55jv-c5g9, whose shipped fix is exactly
> this shape: a **constant set of four** O(1) field-checks (`offset ≥ 0`, `length ≥ 0`,
> `offset ≤ file_size`, `length ≤ file_size − offset` in `onnx/external_data_helper.py`), each
> rejecting only a physically-impossible read. The strongest IN case is therefore one whose
> bound is **physically determined by the input itself** (declared length vs. bytes that
> actually exist), where "impossible" is decided by the file, not by policy. **Attacker-controlled-depth
> recursion / stack overflow** is IN **only when** its depth bound is likewise physically
> grounded — i.e. when a maximum nesting depth is **implied by the message's own byte size**
> (a deeper structure cannot be encoded in the bytes present), so the bound rejects only
> structures that cannot physically exist. This is **distinct from a *configurable* recursion
> cap** (e.g. protobuf's tunable `SetRecursionLimit`, default 100): a configurable policy cap
> is a *defense knob*, not a behavior-preserving bound, and exceeding it is not by itself a
> security bug. A **bare policy depth-constant** (`MAX_DEPTH` chosen by fiat) is therefore
> **not** automatically behavior-preserving: it can reject a legitimately deep machine-generated
> graph, so whether a given depth cap qualifies is a **maintainer-ratification call** (see
> Unresolved questions) — this RFC does **not** claim recursion is as cleanly decidable as
> over-allocation. **Unit of "constant number of bounds":** a **fixed set of O(1) checks
> applied per parsed object / field *before* any attacker-amplified allocation** — *not* a
> count over the whole model. Validating N external-data records with the same four per-record
> O(1) checks is still a "constant set" (the check template is fixed; it runs once per object
> it guards), so a reporter cannot reclassify a per-object over-allocation as "O(n), therefore
> out." ("Single O(1) bound" was the wrong count;
> the precedent uses a constant *set* of checks.) **Out:** ReDoS and quadratic / exponential
> compute — because **no cheap behavior-preserving bound exists**: the only available "bound"
> is a **global size cap** (a max input length / `MAX_NODES`) that rejects **legitimate large
> models**, not merely impossible ones, so it is **not** behavior-preserving (the backtracking
> / blowup occurs at input sizes a real producer emits). **Discriminator (sole gate,
> agent-decidable):** *does a fixed set of cheap, behavior-preserving O(1) checks — applied
> per parsed object / field, rejecting only impossible / malformed inputs, never legitimate
> large inputs — remove the unbounded growth? YES → in; NO → out.* Anti-gaming: a reporter **cannot** argue "one global
> `MAX_NODES` cap is an O(1) bound, so my quadratic blowup is IN" — a global cap rejects
> legitimate large models, so it is not behavior-preserving and does not qualify; conversely
> ReDoS stays out for the same reason. `memory-safety-adjacency` is the **rationale** for where
> the line is drawn, **not** a second required gate.
> *(Separate **policy** line, not part of this scope test: ONNX also declines to add a new
> third-party dependency — e.g. the vetoed `regex` dep — solely for DoS hardening; this is a
> policy stance, not the reason ReDoS is out of scope.)*
>
> **Why the IN primitive must be named (this is what fixes C1):** a Tier-2 *component*
> carve-out **alone does not suffice** — without naming size-driven over-allocation as an IN
> primitive, a plain "classify by primitive" routes GHSA-538c-55jv-c5g9's over-allocation to
> "DoS → out" and would **contradict ONNX's own shipped fix**. The named primitive, not the
> component, is load-bearing.

> **Clause 3 — disclosure & embargo trigger.** Tool-found, easily reproduced findings are
> handled **publicly**, via a normal one-advisory-per-PR fix. Embargo applies on **either**
> trigger. **(A) A demonstrated practical exploit beyond a crash** — **regardless of how
> easily it reproduces**: (i) remote code execution, (ii) privilege escalation, (iii) memory
> corruption with a working control-flow-hijack or information-leak proof of concept (not a
> bare SIGSEGV), or (iv) sandbox / filesystem escape via path or external-data handling.
> **(B) A non-obvious finding** (not reproducible with widely-available tooling) that is
> **plausibly exploitable beyond a crash** — handled privately until triaged, so a
> not-yet-weaponized but non-trivially-discovered issue is not 0-dayed by a public report.
> (Trigger B is the same precaution stated in the guide-level disclosure rule and the
> proposed `SECURITY.md` reporting text; it is listed here so the verbatim clause and the
> surrounding prose embargo on the same two prongs.)

> **Clause 4 — memory-safety vs. pure crash (evidence & disposition).** A crash whose root
> cause is shown to be an OOB read/write is memory-corruption (IN scope) even if the only
> demonstrated effect is a crash. "Pure crash" (out of scope) means NO underlying
> memory-safety violation — null deref, uncaught exception, assertion/abort. To claim
> memory-safety / in-scope status, the **reporter** supplies corruption evidence: either an
> ASan/Valgrind trace, a write-primitive PoC, **or** a minimal reproducer. When a reproducer
> is provided, the triager runs it under ONNX's **existing** sanitizer CI: the Ubuntu debug
> leg of `.github/workflows/main.yml` runs the test suite under ASan + UBSan
> (`-DONNX_USE_ASAN=ON`, `ONNX_HARDENING=ON`; CMake links `Sanitizer::address` +
> `Sanitizer::undefined`). The reproducer runs under **that** job — no new CI job is needed.
> Disposition is a state machine. **Reporter-supplied corruption evidence that itself proves
> the violation** — an ASan/Valgrind trace or a write-primitive PoC — routes **directly to
> `in_scope` (Tier-1)** even when no *shareable* reproducer accompanies it (e.g. a downstream
> vendor reporting an ASan heap-overflow from a proprietary model they cannot redistribute);
> credible corruption evidence is **never** down-ranked merely for lack of a public
> reproducer, and confidential artifacts are handled privately. A report with **a reproducer
> but no trace** sits in **`provisional_in`** (corruption *suspected* — the visible "might be
> a real OOB-write" state) while the triager runs it under the sanitizer CI just described
> (lines above). A
> **sanitizer-confirmed** corruption routes to **`in_scope` (Tier-1)**; a report with
> **neither evidence nor reproducer** routes to **`provisional_dos_out`** (tracked,
> reopenable). A **clean** sanitizer run does **not** prove benignity — it only fails to
> confirm on the *executed path*, and ASan/UBSan in particular **miss in-allocation OOB reads
> and uninitialized-memory leaks** (which need targeted review or MSan) — so for an
> OOB-read / info-leak claim a clean run triggers root-cause review, and otherwise routes back
> to **`provisional_dos_out`** (reopenable via `reopen_if`), **never** to terminal
> `out_of_scope`.
> `out_of_scope` is terminal **only** for a primitive proven benign (null deref / uncaught
> exception / assertion-abort, or genuinely-algorithmic DoS the Clause-2 discriminator
> excludes). Both
> provisional states are **tracked, auditable** — an undetermined memory bug is
> relocated-to-tracked, **never silently dropped**. The burden to establish corruption rests
> on the reporter, not on the maintainer to disprove it. Routing a report that *claims*
> memory-safety / corruption to `out_of_scope` or `provisional_dos_out` is a status-changing,
> down-ranking call that **requires human maintainer sign-off** — never an autonomous AI
> decision (see AI-assisted triage guardrails). Every `provisional_dos_out` item is
> **re-reviewed within the publish-or-dismiss window** (a fixed number of days set at
> ratification; **90 days** is the proposed default, after PyTorch — see the SLA subsection
> and Unresolved questions) — reopened on new corruption evidence or confirmed `out_of_scope`
> — so "tracked, reopenable" is enforceable on a concrete clock, not aspirational.

> **Downstream-relations policy (ratified; not a triage clause).** For a verified crash-only, malformed-model
> robustness bug, ONNX upstream will typically handle it as a public robustness issue rather
> than embargo it or request a CVE. This reflects ONNX's own threat model and is **not** a
> judgment that the bug is unimportant downstream: redistributors, distributions, and CNAs
> whose deployment threat models differ MAY treat it as a vulnerability, and ONNX will
> **cooperate** with them — sharing reproducers, fixes, and references.

### Proposed replacement for the `SECURITY.md` "Reporting a Vulnerability" section

This **amends** (does not append to) the existing "Reporting a Vulnerability" section so the
disclosure stance is one self-consistent policy rather than a blanket "do not disclose until
fixed." Proposed verbatim text:

> **Reporting a Vulnerability**
>
> Whether to report a finding privately or publicly depends on the finding, per ONNX's
> security threat model — it is not a blanket rule.
>
> **Reporting safely is the first rule: when in doubt, report privately.** You do **not** have
> to determine a finding's exploitability or severity to report it — that classification is
> the maintainers' job, not yours. If a finding *might* involve native memory corruption (any
> out-of-bounds read/write or overflow), filesystem access outside the model directory, a
> provenance/trust bypass, or any potential remote code execution or privilege escalation —
> **or if you are unsure** — report it **privately** using GitHub Security Advisories and do
> **not** open a public issue or disclose it until a maintainer has classified it. This
> "fail-closed" default exists so that an uncertain reporter never accidentally 0-days a real
> vulnerability by self-classifying it as harmless.
>
> - **Public handling is the default *outcome*, decided after triage.** Most reports are
>   handled in public, but it is the **maintainer**, after classifying the finding against the
>   threat model, who moves an out-of-scope or non-exploitable finding to the public channel —
>   not the reporter up front. The one case where you may safely report directly in public is
>   a finding you are **confident** is an out-of-scope robustness bug (e.g. a reproducible pure
>   crash on a deliberately malformed model with no memory-corruption claim); open a normal
>   public GitHub issue and, where possible, a pull request.
> - **Private coordinated disclosure is required for the embargo class.** A finding with a
>   **demonstrated practical exploit** — a working PoC for RCE, privilege escalation,
>   exploitable memory corruption with control-flow hijack or info leak, or sandbox /
>   filesystem escape — MUST be reported privately via GitHub Security Advisories **regardless
>   of how easily it reproduces**, and not disclosed publicly until a fix and advisory have
>   been released. A finding that is **non-obvious** (not reproducible with widely-available
>   tooling) **and** plausibly exploitable beyond a crash is likewise handled privately until
>   triaged.
>
> If unable to use GitHub, contact onnx-security@lists.lfaidata.foundation. After receipt, a
> maintainer will acknowledge, classify it, work with you on impact / remediation, and — for
> embargoed findings — keep you informed toward a fix and coordinated disclosure.

### Reconciling the existing `SECURITY.md` "Security Requirements" section

`SECURITY.md` today carries a "Security Requirements" paragraph stating that *"ONNX does not
guarantee that models or inputs are trustworthy, and operators are responsible for validating
provenance and applying appropriate isolation, resource limits, and runtime safeguards when
executing untrusted workloads."* Read alone, this can be mistaken for a blanket "all
untrusted-model handling is out of scope," which **contradicts** this RFC's Tier-1 / Tier-2
in-scope stance (memory-corruption in the checker / shape-inference / version-converter, and
filesystem escapes in external-data handling, **are** ONNX's responsibility). Ratification
therefore **amends** this paragraph so it scopes operator-responsibility correctly — to
**running** models (the reference runtime, Tier-3, is **not** intended for untrusted input)
and to algorithmic resource-exhaustion the Clause-2 discriminator excludes — without
disclaiming memory-safety in ONNX's validation tooling or parse-time filesystem safety.
Proposed amended sentence:

> ONNX does not guarantee that models or inputs are trustworthy; operators **running** models
> (including via the reference runtime, which is **not** intended for untrusted input) are
> responsible for provenance validation, isolation, resource limits, and runtime safeguards.
> This operator responsibility does **not** narrow ONNX's own threat model: memory-safety in
> the checker, shape-inference, and version-converter, and parse-time filesystem safety in
> external-data handling, **remain in scope** per ONNX's security threat model.

### The three orthogonal axes

The model must state, and the triage guide must operationalize, that the three rules are
decided independently:

- **TIER** — in scope to *fix*? (component × primitive lookup)
- **SEVERITY** — how bad? (CVSS + reachability)
- **DISCLOSURE** — public vs embargo? (demonstrated practical exploit, **or** non-obvious + plausibly exploitable)

Collapsing these three ("touches memory → critical → embargoed") is the engine of every
inflated advisory; keeping them orthogonal is what makes the policy both inflation-proof
and consistently decidable.

### Trust tiers — component × primitive (with repo entry points)

| Tier | Component | Repo entry point(s) | In scope? (by primitive) |
|---|---|---|---|
| **1** | C++ checker | `onnx/checker.cc` / `onnx/checker.h` | OOB-**write** → in. OOB-**read** → in only if it leaks info. Pure crash (SIGSEGV/abort without corruption) → out. A crash *claimed* as memory-corruption needs reporter evidence (ASan/Valgrind trace, write-primitive PoC, or a reproducer run under the existing ASan/UBSan CI); absent that it is tracked `provisional_dos_out`, not closed (Clause 4). Bypass (malformed model accepted) → rated by downstream reachability, typically low. |
| **1** | Shape inference | `onnx/defs/**/*.cc` (`*ShapeInference*`), `onnx/shape_inference/` | Same primitive rule. Native OOB-write / overflow-with-corruption on a crafted graph = in; bare SIGSEGV = out. |
| **1** | Version converter | `onnx/version_converter/` | Same. OOB-**write** (e.g. heap overflow during conversion) = in, high priority. (`ParseData` in `onnx/defs/tensor_util.cc` is a general-purpose tensor-parse macro reached from multiple components, not just conversion.) |
| **2** | External-data / path handling | `onnx/checker.cc` (`open_external_data`, `resolve_external_data_location`), `onnx/external_data_helper.py`, `onnx/model_container.py` | **In** — filesystem escape (read/write/exfil), symlink/hardlink/TOCTOU, path traversal. (Partially hardened today; see `docs/Security.md`.) |
| **3** | Reference evaluator | `onnx/reference/reference_evaluator.py`, `onnx/reference/ops/**` | **Out entirely** — not intended for untrusted input. Crash / RCE claims → documented limitation. |
| **4** | Resource-exhaustion (classify by **primitive, not component**) | cross-cutting (recursion / nesting-depth, over-allocation, regex, quadratic passes) | Defers to the **Clause-2 behavior-preserving-bound discriminator**. **In:** size-driven over-allocation (precedent GHSA-538c-55jv-c5g9) and attacker-controlled-depth **recursion / nesting-depth** *only when its depth bound is physically grounded in the message's own byte size* (a bare policy `MAX_DEPTH` is a maintainer-ratification call, not automatically in — see Clause 2). **Out:** ReDoS and quadratic / exponential compute (closable only by a global cap that rejects legitimate large models — not behavior-preserving). |

*(The former model-hub component (`onnx/hub.py`) was removed from the repository in #7678
and is therefore not part of this taxonomy; the reference evaluator is the sole Tier-3
example.)*

### Severity rubric (CVSS-anchored, by primitive)

Severity is independent of tier and disclosure. CVSS bands are keyed on the
**maximum-justifiable** confidentiality / integrity / availability (C/I/A) impact for each
primitive. **Baseline Attack Vector is Local (`AV:L`)**: the attack vector is a model file
the victim opens, not a network service, so the bands below are computed under `AV:L`. A
finding genuinely reachable over the network (`AV:N`) is scored on its own vector and can
exceed these bands — the table is the *local-model* baseline, not an absolute cap.

**Posture: strict on scope, conservative on severity, escalate on evidence.** ONNX is
deliberately **stricter on *scope*** than peer projects (Tier-1 memory-safety is in scope,
broader than PyTorch / TensorFlow) **and conservative on *severity***: under the `AV:L`
local-model baseline, an OOB / memory-corruption write is capped at **High (7.0–8.9)**
absent a *working* RCE / control-flow-hijack PoC, rather than auto-rated Critical. These are
**maximum-justifiable** caps, not permanent ceilings — a finding is **re-scored to Critical
(9.0+) the moment an RCE / CFH PoC appears** (and an `AV:N`-reachable case is scored on its
own vector, per above). The wide scope and the restrained scoring are complementary: pulling
more into scope makes anti-inflation scoring *more* necessary, not less, so a volunteer
project is not flooded with reflexively-Critical robustness findings.

| Primitive | CVSS band (max-justifiable C/I/A, `AV:L` baseline) | Required grounding |
|---|---|---|
| **OOB / memory-corruption write** (Tier 1/2) | High **7.0–8.9** (~7.8 typical); Critical **9.0+** only with a working CFH/RCE PoC. Max **C:H/I:H/A:H** | Repro + named reachable entry point + CVSS vector |
| **OOB read that yields an information leak** (Tier 1/2) | Medium–High **5.0–7.5**. Max **C:H/I:N/A:N** | Repro showing an observable / exfiltrable out-of-bounds read (a non-leaking OOB read is out of scope) |
| **Filesystem escape** (Tier 2) | Read-only escape Medium **5.5–6.5** (max **C:H/I:N/A:N**); arbitrary write/delete outside the model directory High **7.1–8.1** (max **C:H/I:H/A:H** — overwriting / deleting files is an availability impact) | PoC reading, or writing / deleting, outside the model directory |
| **Resource-exhaustion the Clause-2 discriminator admits** (Tier 1/2) | Low–Medium **3.1–6.5** (`AV:L`; an `AV:N`-reachable variant scores ~7.5 on its own vector). Max **A:H only** | A constant set of cheap, behavior-preserving O(1) bounds (rejecting only impossible / malformed inputs, never legitimate large ones) removes the unbounded growth (Clause 2; precedent GHSA-538c-55jv-c5g9) |
| **Validation bypass requiring downstream misuse** (Tier 1) | Low / informational **0.0–3.9**. **C:N/I:N/A:N** at the component | Must state the downstream-misuse precondition explicitly (Clause 1) |
| **Pure crash / SIGSEGV / abort without corruption** (Tier 1) | **Out of scope** — not assigned | — |
| **Non-leaking OOB read** (Tier 1) | **Out of scope** — not assigned | — |
| **Resource-exhaustion the Clause-2 discriminator excludes** (ReDoS, quadratic / exponential) | **Out of scope** — not assigned | — |
| **Reference runtime on untrusted input** (Tier 3) | **Out of scope** (documented limitation) — not assigned | — |

**Anti-inflation lookup.** A maintainer rebuts an inflated CVSS vector by checking the
claimed C/I/A against the primitive's max-justifiable column: a bypass or pure crash is
**C:N/I:N** and therefore cannot reach High. This is a **lookup, not a negotiation**.

**Reachability modifier (within a band, not across scope).** Within each row, malformed-model-only
reachability sits at the **low end** of the band; benign-production-model
reachability sits at the **high end**. Reachability moves severity *inside* the primitive's
band — it **never** moves a finding *into or out of* scope (that is the primitive + Clause-2/4
gate's job), and it **never** decides public vs embargo (that is Clause 3's job alone).

**Report-quality filter.** A report claiming severity above informational MUST supply
(1) a reproducing PoC (or an explicit "not reproduced" flag), (2) a cited threat-model
clause, and (3) a named reachable entry point. Absent these, severity is capped and the
report is returned for evidence. This is a **rule application, not a judgment about the
reporter.**

### Workflow (operational — derives from the model)

- **One advisory → one focused PR.** Clustering is permitted only as a *tracking* artifact,
  never as a merged bundle. Identical-pattern fixes MAY batch with an explicit note.
- **Disposal authority.** A maintainer MAY downgrade or close an advisory by citing the
  specific clause, tier, or missing-grounding item.
- **Disclosure routing.** Default to a public PR; embargo only on a Clause-3 trigger,
  opened privately and moved public on fix.
- **Appeal / escalation chain.** Maintainer triage → (dispute) infra SIG → (unresolved)
  Steering Committee.

### One-time backlog re-classification pass

On ratification, the maintainers run a **single tracked effort** to re-triage every open
security advisory (~3 dozen today) against this model. Each advisory is tagged with its
**tier + primitive classification** and a **disposition**, and the result — classification
plus rationale — is recorded **on the advisory itself** for auditability:

- **Genuine in-scope** (Tier-1 / Tier-2 memory-safety; Tier-2 filesystem / external-data
  **path-traversal, symlink / hardlink, TOCTOU**) — kept embargoed and **fixed individually**,
  one advisory → one PR, with a confirmed CVE.
- **Ambiguous** (crash without corruption evidence, undetermined root cause) — the Clause-4
  **provisional** logic applies (`provisional_in` / `provisional_dos_out`); tracked,
  reopenable, never silently closed.
- **Out-of-scope** (pure crash, algorithmic DoS the Clause-2 discriminator excludes) — **dismissed**
  with a standard disposition note (clause + tier + missing-grounding) and **re-filed as a
  public robustness issue**.

This converts the contested all-in-one batch into a **transparent, per-advisory public
record**. (A competing draft proposes category buckets; those are a triage **front-end**, and
their crosswalk onto this tier × primitive model is kept in the Layer-1 guide so **no parallel
taxonomy enters this RFC** — the L0 decision stays on tier × primitive.)

### Changes to `INCIDENT_RESPONSE.md`

`INCIDENT_RESPONSE.md` already exists at the repository root and is the operational
counterpart to this model. Ratification implies the following deltas (its **14-business-day**
response target and volunteer framing are unchanged):

- **"Response Steps → 1. Confirm / 2. Triage"** gain an explicit **classify-against-the-threat-model
  gate before escalation**: the Incident Lead applies the tier/primitive lookup first and
  applies the Clause-4 **hold** (`provisional_in`) for ambiguous, native-write, filesystem,
  or trust-related reports rather than escalating immediately.
- The Triage step's existing line — *"reports … outside the project's threat model may be
  closed without a CVE"* — is upgraded from **optional to the DEFAULT disposition** for
  out-of-scope categories, with this model's standard disposition note attached.
- **"4. Disclose"** gains the **publish-or-dismiss SLA** (below).
- **"Escalation" (currently an empty / TBD section)** is filled with this model's
  already-specified appeal chain — **maintainer triage → infra SIG → Steering Committee** —
  the same routing in the Workflow subsection above (subject to ratification of the final
  authority; see Unresolved questions).
- **Intake reconciliation (so `0008`, `SECURITY.md`, and `INCIDENT_RESPONSE.md` do not
  contradict).** The IRP's *"Do NOT open a public issue for security vulnerabilities"* line is
  **retained**, scoped to the **embargo class**: a reporter who suspects RCE, privilege
  escalation, exploitable memory corruption, or a filesystem escape — or who is **unsure** —
  reports **privately first**. For findings the reporter is confident are out-of-scope
  robustness bugs (and other non-embargo findings), the public channel is correct. After
  classification, it is the **maintainers** who re-file dismissed out-of-scope advisories
  publicly. This RFC's public-by-default stance thus **amends** the IRP's former blanket
  private-first intake to **the embargo class plus a fail-closed "when unsure, report
  privately" default**: intake stays private-first for findings that *might* be
  memory-corruption / filesystem-escape / RCE or whose class is uncertain, while the **public
  channel is the default *outcome*** after a maintainer classifies. This **narrows, but does
  not eliminate,** private-first intake — a deliberate revision, not a contradiction.

### Publish-or-dismiss window (SLA)

Every embargoed advisory is, within a **fixed window**, either **published** (fix shipped +
CVE) or **dismissed** (downgraded to a public robustness bug and re-filed). PyTorch's
**90-day** publish-or-dismiss flow is the precedent. A `provisional_in` item must reach a
sanitizer verdict within the window; a `provisional_dos_out` item is re-reviewed within the
same window (Clause 4) — reopened on new corruption evidence or confirmed `out_of_scope`. The
exact number is the **proposed default 90 days**, left for maintainer ratification (see
Unresolved questions) — a volunteer project may choose 90 / 120 / 180 days or a "best-effort
90-day target."

A **confirmed in-scope but not-yet-fixed** advisory is **not** force-disclosed unfixed when
the window elapses: the maintainers may grant a **logged, bounded embargo extension**
(recorded on the advisory with a new target date). This extension valve is ONNX's **own**
addition — PyTorch's published policy has no such mechanism (it is a hard publish-or-dismiss);
to prevent an indefinite silent hold, an extension **requires documented maintainer approval,
is recorded privately on the advisory while the embargo holds (and the extension log is
published when the advisory is disclosed, so it is not pre-announced as an unfixed
vulnerability), and is capped at a bounded cumulative total before mandatory escalation to
the infra SIG** (the exact cap is a maintainer-ratification item — see Unresolved questions).
The
default remains publish-or-dismiss; the logged extension is the explicit, auditable exception
for a genuine fix still in progress — never an indefinite silent hold.

### AI-assisted triage guardrails

Because this model is meant to be applied **with AI assistance**, two rules are **normative**:

- **No embargoed content to third-party AI.** Embargoed or private advisory content MUST NOT
  be sent to third-party AI services or any external API. AI-assisted triage is permitted
  **only** on already-public reports, or run **entirely locally / on-device**.
- **Human sign-off on status-changing dispositions.** A human maintainer MUST sign off on any
  disposition that changes embargo/public status or dismisses a plausible security report.
  **AI assistance drafts; it does not decide.** In particular, routing a report that **claims**
  memory-safety or memory-corruption to `out_of_scope` or `provisional_dos_out` always requires
  human sign-off — de-prioritizing a plausible OOB-write is status-changing. (This subsumes the
  agent STOP-gates carried in the Layer-1 wrapper.)

### Layer 1 (follows ratification)

Once this model is ratified, a thin operational guide (`docs/SecurityTriage.md`) and an
agent wrapper (`.agents/skills/security-triage/SKILL.md`) derive from it. The guide
restates no normative claim; it applies the ratified model as a one-screen decision tree
plus the rubric above. The agent wrapper is a thin pointer to the guide (single source of
truth) that adds only a triage-record schema and hard human-escalation gates. These ship
as a separate, low-risk PR **after** this RFC is accepted.

## Drawbacks
[drawbacks]: #drawbacks

- Declaring components out of scope (the reference runtime, algorithmic resource-exhaustion)
  could be read as ONNX "not caring" about robustness. Mitigation: out-of-scope-as-a-*vulnerability*
  still allows fixing issues as ordinary public robustness bugs; the model changes the
  *channel and severity*, not whether bugs get fixed.
- A ratified boundary is a commitment that is harder to change later than ad hoc practice.
  This is intentional — the predictability is the benefit — but it raises the bar for
  getting the wording right now, which is why ratification (not assertion) is requested.
- The taxonomy adds vocabulary maintainers must learn. Mitigation: the entire decision is
  three table lookups; the derived Layer-1 triage guide (`docs/SecurityTriage.md`) shows a
  worked example end to end.
- Conservative severity scoring (capping a not-yet-RCE OOB-write at High) risks later
  criticism if such a write is subsequently shown to be RCE-able. Mitigation: the bands are
  **maximum-justifiable** caps with explicit **re-scoring to Critical on RCE / CFH
  evidence**, so conservative means *restrained-by-default, escalated-on-proof* — never
  permanently low. This is the deliberate cost of the anti-inflation goal that motivates the
  RFC.

## Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

**Why this design.** The recurring problem is *definitional*, not procedural: maintainers
have disagreed about whether a given report is even a vulnerability. A ratified threat
model is the only artifact that resolves that disagreement; a procedural checklist built
on an unratified boundary would simply relocate the same argument into the checklist's
review.

**Alternatives considered:**

- **Triage-guide only (no ratified boundary).** Insufficient — a checklist has no
  normative anchor, so "is this in scope?" stays unsettled and the guide re-litigates
  severity ad hoc.
- **Do nothing / per-advisory ad hoc.** This is the status quo that produced inconsistent
  outcomes and repeated maintainer disagreement; it does not scale.
- **Full automation (bot-driven triage).** A bot that forces advisories through fixed
  questions is only as good as the boundary it encodes; automation must be *subordinate* to
  a ratified human stance, never a substitute for it. (It also risks repeating skepticism
  toward automated security handling.)
- **Copy TensorFlow / PyTorch wholesale.** Their model treats an untrusted model as
  **code-execution-equivalent** (TF: "using untrusted models or graphs is equivalent to
  running untrusted code") — out of scope to defend, protection reserved for benign-model-reachable
  corruption. ONNX differs: its checker / shape-inference / version-converter are
  advertised validation tooling **run on untrusted models** to decide whether to trust them.
  So Tier-1 memory-safety is in scope for ONNX even though the analogous runtime crash is
  not — a divergence this RFC makes explicit rather than inheriting.

**Impact of not doing this.** Advisories continue to be triaged inconsistently, maintainers
keep re-arguing scope, and reporters receive unpredictable severity and disclosure outcomes.

## Prior art
[prior-art]: #prior-art

ONNX's stance is deliberately aligned with the **community ML projects** most comparable to
it, and diverges only where ONNX's architecture demands it.

- **PyTorch** (security policy). Maintains an explicit *"Issues That Are NOT Security
  Vulnerabilities"* section and the principle that **an attacker who already has local
  code-execution / file-write gains no additional capability** — such reports are not
  vulnerabilities. **Denial-of-service via resource consumption** is explicitly **not** a
  vulnerability. Operates a **90-day publish-or-dismiss** flow: fixed-and-disclosed, or
  dismissed and **re-filed as a public issue**, within the window.
- **TensorFlow** (`SECURITY.md`). Treats using an untrusted model as **equivalent to running
  untrusted code** (recommends sandboxing); memory corruption is a security issue **only if
  reachable and exploitable through production-grade, benign models**, and
  **resource-allocation DoS is out of scope**. TF's "production-grade benign model" test is
  definitive in wording but inherently **judgment-laden** — that imprecision is what this RFC
  narrows, not a gap TF itself flags as open.
- **Protobuf**: ships **default** recursion (default limit 100) and total-size limits and
  documents **tuning** them (`SetRecursionLimit` / `SetTotalBytesLimit`) as the consumer's
  responsibility; exceeding a configured limit is **not** treated as a parser vulnerability.

**Our principled divergences (deliberate, not oversights):**

- **Filesystem / provenance stays IN scope** (PyTorch excludes the analogous category) on
  ONNX's own merits: ONNX performs **parse-time filesystem I/O** (external-data resolution),
  so a path / symlink / TOCTOU escape grants capability at *load / validate* time — before any
  "attacker already has code-exec" precondition would apply, so the no-additional-capability
  principle does not neutralize it.
- **TensorFlow's benign-reachability is a SEVERITY input, not a scope gate.** We feed
  "reachable through a benign model?" into the **CVSS / reachability** axis of the severity
  rubric; we do **not** use it to decide scope. Our scope gate is the **root-cause primitive
  + the behavior-preserving-bound discriminator** (Clauses 2 and 4). For the strongest cases
  (memory-corruption primitives; physically-bounded over-allocation) this is a genuinely
  decidable test rather than TF's judgment-laden "production-grade benign model" call. We do
  **not** claim it removes *all* judgment: a bare policy depth-cap (the Clause-2 recursion
  case) remains a maintainer call — the same class of judgment TF's gate requires — so this
  RFC **narrows where** that judgment is needed rather than eliminating it.
- **We diverge from a *named* item in PyTorch's "NOT vulnerabilities" list — deliberately.**
  PyTorch enumerates "crashes and out-of-bounds access" as **not** security vulnerabilities
  ("the caller is already running native code"). ONNX keeps **OOB-write (and info-leaking
  OOB-read) in scope** for its Tier-1 validation tooling — directly diverging from that named
  item — on the same checker-runs-on-untrusted-models grounds: ONNX's checker /
  shape-inference / version-converter are advertised to run **on untrusted models to decide
  whether to trust them**, so PyTorch's "caller already runs native code" premise does not
  hold at validation time. We surface this divergence explicitly rather than cite PyTorch
  selectively.

**Why not a corporate-style funnel.** ONNX-Runtime routes through **Microsoft MSRC**, and
NVIDIA's ONNX tooling through **NVIDIA PSIRT** — closed, vendor-owned disclosure funnels. As a
vendor-neutral **LF AI & Data** project, ONNX deliberately follows the *open community* model
(PyTorch / TensorFlow): public-by-default, narrow embargo, transparent per-advisory record —
not a single-vendor security queue. The Tier-1 memory-safety carve-out is ONNX's one
documented divergence from the trusted-input model, justified because its checker /
shape-inference / version-converter are intentionally **run on untrusted models**.

## Unresolved questions
[unresolved-questions]: #unresolved-questions

To be resolved through the RFC discussion (these are maintainer decisions, not crew
assertions):

- **Tier classification of each component** (the table above) — especially the Tier-1
  memory-safety carve-out and the cheap-bound discriminator that routes some
  resource-exhaustion in scope (Clause 2).
- **Scope gate for malformed-model-only memory issues — chosen position vs. discussable
  compromise.** This RFC's **chosen (stricter) position** is *primitive-based scope*: an
  out-of-bounds write (or info-leaking OOB read) in the checker / shape-inference /
  version-converter is **in scope to fix regardless of reachability**, with malformed-model-only
  reachability handled as a *severity* reducer (low end of band, public) rather than a scope
  exclusion. The **discussable looser compromise** is to adopt TensorFlow's gate directly —
  require **benign-production-model reachability as a necessary condition for scope**, so a
  memory issue reachable *only* through a deliberately malformed model is treated as an
  ordinary public robustness bug, not a Tier-1 security finding. The chosen position defends
  more (consistent with ONNX's checker being run on untrusted models); the compromise reduces
  maintainer burden by excluding malformed-only crashes from the security channel entirely.
  Maintainers ratify which.
- **The behavior-preserving-bound discriminator** (Clause 2) — which exhaustion cases a
  constant set of cheap, behavior-preserving O(1) bounds (rejecting only impossible / malformed
  inputs, never legitimate large ones) genuinely closes — **and, specifically, whether any
  given recursion / nesting-depth cap qualifies**: a depth bound physically derived from the
  message's own byte size is behavior-preserving, but a bare policy `MAX_DEPTH` constant is a
  judgment call (it can reject a legitimately deep machine-generated graph) and is not claimed
  as cleanly decidable as physically-bounded over-allocation.
- **The evidence burden and `provisional_dos_out` disposition** (Clause 4) — confirming that
  the reporter carries the burden of establishing corruption (trace / PoC / reproducer run
  under the existing ASan/UBSan CI), and that an unproven crash is tracked out/low, not
  closed. (No new CI job is proposed; the Ubuntu debug leg already runs ASan + UBSan.)
- **The exact embargo trigger list** (Clause 3, items i–iv).
- **Appeal / escalation authority** — who has the final call when a reporter disputes a
  downgrade (proposed: infra SIG → Steering Committee).
- **The publish-or-dismiss window** — the fixed number of days (e.g. 90 / 120 / 180, or a
  best-effort 90-day target after PyTorch) within which an embargoed advisory must be
  published or dismissed; a volunteer project may need longer. **And the embargo-extension
  cap** — the bounded cumulative extension total (and approval / escalation threshold) beyond
  which a still-unfixed in-scope advisory must escalate to the infra SIG rather than extend
  again.
- **Home of the threat model** — this RFC **recommends** a clearly-titled section extending
  `SECURITY.md` (fewer root files; reconciles with the existing `SECURITY.md` and
  `INCIDENT_RESPONSE.md`), with `docs/Security.md` retained as a sub-component design document
  that **inherits** the global tiers (not a parallel policy). A dedicated top-level
  `THREAT_MODEL.md` (PyTorch / TensorFlow precedent; a single citable source) is the
  alternative if maintainers prefer it. Either way the boundary statement above is written to
  be quotable as one anchor-able unit, and `docs/Security.md` must be cross-linked as
  subordinate so the repository does not carry three overlapping security postures.

## Future possibilities
[future-possibilities]: #future-possibilities

- The Layer-1 triage guide and agent wrapper described above, shipped after ratification.
- A GitHub issue-form that prompts reporters for the three grounding items (PoC, cited
  clause, reachable entry point) at intake, improving report quality without adding
  automation that acts autonomously.
- Periodic review of the tier table as new components are added to the `onnx` package, so
  the boundary stays current with the codebase.

## Appendix: backlog dry-run (illustrative)
[appendix-backlog-dry-run]: #appendix-backlog-dry-run

> **Illustrative only.** These are **aggregate** counts from a dry-run of the *proposed*
> (not-yet-ratified) model against the open advisory backlog. They **disclose no advisory
> identities**, **change no advisory state**, and are **contingent on ratification** — this
> is "what ratification would buy," not triage already performed. The specific open
> (non-public) advisories are deliberately not enumerated here, consistent with Clause 3 and
> the AI-assisted-triage guardrails.

Applying the three axes to the current open backlog (~30 advisories) routes them as follows:

| Disposition under the model | Count | Meaning |
|---|---:|---|
| **Cleared-out** (→ public robustness bug) | ~⅓ | pure crash / non-leaking OOB-read / ReDoS / reference-runtime — out of the security channel, fixed as ordinary public bugs |
| **In-scope fix** (stays an advisory) | ~⅓ | OOB-write, info-leaking OOB-read, filesystem escape, Clause-2-admitted over-allocation — fixed one-advisory-per-PR |
| **Provisional** (tracked, auditable) | ~¼ | claimed memory-corruption (often integer-overflow → asserted downstream OOB) **without** sanitizer/PoC evidence — held in `provisional_*`, never silently closed, re-reviewed within the SLA window |
| **Maintainer judgment** (genuine residue) | 2 | attacker-controlled recursion / nesting-depth, where a bare-policy depth cap is not cleanly decidable (Clause 2) |

The headline is **not** that the model auto-resolves everything: it is that the large
majority land in a **policy-defined disposition** (cleared, fixed, or auditable-provisional)
by rule rather than by re-argued judgment, leaving a **small, explicitly-named residue** (the
recursion cases) for a maintainer call the RFC openly declines to pre-decide.

**Worked examples (already-published advisories only).** The model's dispositions are
illustrated by advisories ONNX has *already* disclosed publicly:

- **GHSA-538c-55jv-c5g9** (over-allocation via unbounded external-data offset/length) — the
  **Clause-2 in-scope precedent**: its shipped fix is the constant set of four O(1)
  file-size bounds in `onnx/external_data_helper.py`.
- **GHSA-3r9x-f23j-gc73**, **GHSA-p433-9wv8-28xj** (symlink path traversal),
  **GHSA-cmw6-hcpp-c6jp** (hardlink bypass), **GHSA-q56x-g2fj-4rj6** (TOCTOU read/write) —
  **Tier-2 filesystem-escape, in scope**.
- **GHSA-hqmj-h5c6-369m** (hub trust-check bypass) — note the `onnx.hub` component was
  **removed in #7678** and is **not** part of the current tier table; included only to show
  the trust-bypass class, not as a live component.
