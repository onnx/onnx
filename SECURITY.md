<!--
SPDX-FileCopyrightText: Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# Security Policy

## Threat model

This section is ONNX's security threat model: the boundary statement below defines what ONNX's validation tooling defends against, and the scope clauses make each decision rule explicit.

> **Boundary.** For ONNX's validation tooling (checker, shape-inference, version-converter):
> a memory-corruption **OOB-WRITE** is in scope and will be fixed; an **OOB-READ** is in
> scope only if it produces an information leak; a validation **BYPASS** (malformed input
> accepted) is rated by downstream reachability, typically low; a **resource-exhaustion** is
> in scope **if and only if a constant number of cheap, behavior-preserving O(1) bounds**
> (rejecting only impossible / malformed inputs, never legitimate large inputs) remove the
> unbounded growth — covering size-driven over-allocation (precedent: the shipped external-data
> bounds fix, GHSA-538c-55jv-c5g9) and attacker-controlled-depth recursion / stack overflow *whose
> depth bound is physically grounded in the message's own byte size* (see Clause 2); a
> resource-exhaustion closable only by a global cap that rejects legitimate large models
> (ReDoS, quadratic / exponential compute) and a **pure crash** (SIGSEGV/abort without memory
> corruption) are out of scope. The
> **reference runtime is not intended for untrusted input and is out of scope entirely.**

### Scope clauses

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
> `docs/AssuranceCase.md` — has been reconciled to this best-effort statement: triage
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
> graph, so whether a given depth cap qualifies is a **maintainer-ratification call** — ONNX does **not** claim recursion is as cleanly decidable as
> over-allocation. **Unit of "constant number of bounds":** a **fixed set of O(1) checks
> applied per parsed object / field *before* any attacker-amplified allocation** — *not* a
> count over the whole model. Validating N external-data records with the same four per-record
> O(1) checks is still a "constant set" (the check template is fixed; it runs once per object
> it guards), so a reporter cannot reclassify a per-object over-allocation as "O(n), therefore
> out." **Out:** ReDoS and quadratic / exponential
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
> third-party dependency solely for DoS hardening; this is a
> policy stance, not the reason ReDoS is out of scope.)*
>
> **Why the IN primitive must be named:** a Tier-2 *component*
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
> (Trigger B is the same precaution stated in the Reporting a Vulnerability section above,
> restated here so the clause and the reporting guidance embargo on the same two prongs.)

> **Clause 4 — memory-safety vs. pure crash (evidence & disposition).** A crash whose root
> cause is shown to be an OOB read/write is memory-corruption (IN scope) even if the only
> demonstrated effect is a crash. "Pure crash" (out of scope) means NO underlying
> memory-safety violation — null deref, uncaught exception, assertion/abort.
>
> **Evidence.** To claim memory-safety / in-scope status, the **reporter** supplies corruption
> evidence: an ASan/Valgrind trace, a write-primitive PoC, **or** a minimal reproducer. The
> burden to establish corruption rests on the reporter, not on the maintainer to disprove it.
> When a reproducer is provided, the triager runs it under ONNX's **existing** sanitizer CI —
> the Ubuntu debug leg of `.github/workflows/main.yml` runs the test suite under ASan + UBSan
> (`-DONNX_USE_ASAN=ON`, `ONNX_HARDENING=ON`; CMake links `Sanitizer::address` +
> `Sanitizer::undefined`); no new CI job is needed.
>
> **Disposition** is a state machine:
>
> - **Corruption evidence that itself proves the violation** (ASan/Valgrind trace or
>   write-primitive PoC) → **`in_scope` (Tier-1)** — even with no *shareable* reproducer (e.g. a
>   downstream vendor reporting an ASan heap-overflow from a proprietary model they cannot
>   redistribute). Credible evidence is **never** down-ranked merely for lack of a public
>   reproducer; confidential artifacts are handled privately.
> - **Reproducer but no trace** → **`provisional_in`** (corruption *suspected* — the visible
>   "might be a real OOB-write" state) while the triager runs it under the sanitizer CI above.
> - **Sanitizer-confirmed** corruption → **`in_scope` (Tier-1)**.
> - **Neither evidence nor reproducer** → **`provisional_dos_out`** (tracked, reopenable).
> - **A clean sanitizer run** does **not** prove benignity — it only fails to confirm on the
>   *executed path*, and ASan/UBSan in particular **miss in-allocation OOB reads and
>   uninitialized-memory leaks** (which need targeted review or MSan). For an OOB-read /
>   info-leak claim a clean run triggers root-cause review; otherwise it routes back to
>   **`provisional_dos_out`** (reopenable via `reopen_if`). `out_of_scope` is reserved as
>   **terminal only** for a primitive proven benign (null deref / uncaught exception /
>   assertion-abort, or genuinely-algorithmic DoS the Clause-2 discriminator excludes).
>
> Both provisional states are **tracked, auditable** — an undetermined memory bug is
> relocated-to-tracked, **never silently dropped**. Routing a report that *claims*
> memory-safety / corruption to `out_of_scope` or `provisional_dos_out` is a status-changing,
> down-ranking call that **requires human maintainer sign-off** — never an autonomous AI
> decision (see AI-assisted triage guardrails). Every `provisional_dos_out` item is
> **re-reviewed within the publish-or-dismiss window** (90-day default, after PyTorch; see the
> publish-or-dismiss SLA below) — reopened on new corruption evidence or confirmed
> `out_of_scope` — so "tracked, reopenable" is enforceable on a concrete clock, not aspirational.

> **Downstream-relations policy (ratified; not a triage clause).** For a verified crash-only, malformed-model
> robustness bug, ONNX upstream will typically handle it as a public robustness issue rather
> than embargo it or request a CVE. This reflects ONNX's own threat model and is **not** a
> judgment that the bug is unimportant downstream: redistributors, distributions, and CNAs
> whose deployment threat models differ MAY treat it as a vulnerability, and ONNX will
> **cooperate** with them — sharing reproducers, fixes, and references.

### Severity rubric (CVSS v3.1, by primitive)

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

### Publish-or-dismiss window (SLA)

Every embargoed advisory is, within a **fixed window**, either **published** (fix shipped +
CVE) or **dismissed** (downgraded to a public robustness bug and re-filed). PyTorch's
**90-day** publish-or-dismiss flow is the precedent. A `provisional_in` item must reach a
sanitizer verdict within the window; a `provisional_dos_out` item is re-reviewed within the
same window (Clause 4) — reopened on new corruption evidence or confirmed `out_of_scope`. The
exact number is the **proposed default 90 days**, left to maintainer discretion — a volunteer project may choose 90 / 120 / 180 days or a "best-effort
90-day target."

A **confirmed in-scope but not-yet-fixed** advisory is **not** force-disclosed unfixed when
the window elapses: the maintainers may grant a **logged, bounded embargo extension**
(recorded on the advisory with a new target date). This extension valve is ONNX's **own**
addition — PyTorch's published policy has no such mechanism (it is a hard publish-or-dismiss);
to prevent an indefinite silent hold, an extension **requires documented maintainer approval,
is recorded privately on the advisory while the embargo holds (and the extension log is
published when the advisory is disclosed, so it is not pre-announced as an unfixed
vulnerability), and is capped at a bounded cumulative total before mandatory escalation to
the infra SIG** (the exact cap is a maintainer decision).
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
  human sign-off — de-prioritizing a plausible OOB-write is status-changing.

## Reporting a Vulnerability

Whether to report a finding privately or publicly depends on the finding, per ONNX's
security threat model — it is not a blanket rule.

**Reporting safely is the first rule: when in doubt, report privately.** You do **not** have
to determine a finding's exploitability or severity to report it — that classification is
the maintainers' job, not yours. If a finding *might* involve native memory corruption (any
out-of-bounds read/write or overflow), filesystem access outside the model directory, a
provenance/trust bypass, or any potential remote code execution or privilege escalation —
**or if you are unsure** — report it **privately** using GitHub Security Advisories and do
**not** open a public issue or disclose it until a maintainer has classified it. This
"fail-closed" default exists so that an uncertain reporter never accidentally 0-days a real
vulnerability by self-classifying it as harmless.

- **Public handling is the default *outcome*, decided after triage.** Most reports are
  handled in public, but it is the **maintainer**, after classifying the finding against the
  threat model, who moves an out-of-scope or non-exploitable finding to the public channel —
  not the reporter up front. The one case where you may safely report directly in public is
  a finding you are **confident** is an out-of-scope robustness bug (e.g. a reproducible pure
  crash on a deliberately malformed model with no memory-corruption claim); open a normal
  public GitHub issue and, where possible, a pull request.
- **Private coordinated disclosure is required for the embargo class.** A finding with a
  **demonstrated practical exploit** — a working PoC for RCE, privilege escalation,
  exploitable memory corruption with control-flow hijack or info leak, or sandbox /
  filesystem escape — MUST be reported privately via GitHub Security Advisories **regardless
  of how easily it reproduces**, and not disclosed publicly until a fix and advisory have
  been released. A finding that is **non-obvious** (not reproducible with widely-available
  tooling) **and** plausibly exploitable beyond a crash is likewise handled privately until
  triaged.

If unable to use GitHub, contact onnx-security@lists.lfaidata.foundation. After receipt, a
maintainer will acknowledge, classify it, work with you on impact / remediation, and — for
embargoed findings — keep you informed toward a fix and coordinated disclosure.

## Security announcements
Security advisories are published via GitHub Security Advisories. Users depending on ONNX will be notified automatically via GitHub's dependency graph.

## Security Requirements

Open Neural Network Exchange (ONNX) manages reported vulnerabilities according to its documented security policy and delivers remediations in maintained releases. The project employs established secure development practices such as automated testing, continuous integration, and tooling intended to identify defects during development. Third-party dependencies and build components are periodically reviewed and updated to address known issues and to mitigate supply-chain risk. ONNX does not guarantee that models or inputs are trustworthy; operators **running** models (including via the reference runtime, which is **not** intended for untrusted input) are responsible for provenance validation, isolation, resource limits, and runtime safeguards. This operator responsibility does **not** narrow ONNX's own threat model: memory-safety in the checker, shape-inference, and version-converter, and parse-time filesystem safety in external-data handling, **remain in scope** per ONNX's security threat model.

## Supply Chain Security

ONNX release artifacts (wheels and source distributions) meet [SLSA Build Level 2](https://slsa.dev/spec/v1.0/levels#build-l2). Artifacts published to PyPI from this repository's GitHub Actions workflows have a corresponding signed provenance attestation generated by GitHub Actions and stored in GitHub's attestation store.

### Verifying attestations

Install the [GitHub CLI](https://cli.github.com/) and run:

```bash
gh attestation verify <artifact> --owner onnx
```

For example:

```bash
pip download onnx --no-deps -d ./dist
gh attestation verify ./dist/onnx-*.whl --owner onnx
```

A successful verification confirms that the artifact was built by GitHub Actions in the `onnx/onnx` repository and has not been tampered with since it was built.

### Software Bill of Materials (SBOM)

Each wheel also embeds a [CycloneDX 1.7](https://cyclonedx.org/) SBOM (`.cdx.json`) listing the bundled third-party components shipped inside the wheel (e.g. statically linked C++ libraries).
