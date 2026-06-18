<!--
SPDX-FileCopyrightText: Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# ONNX Incident Response Plan

**Repository:** [github.com/onnx/onnx](https://github.com/onnx/onnx) | **License:** Apache 2.0 | **Last Reviewed:** April 2026

This plan defines how the ONNX team receives vulnerability reports, assesses severity, ships fixes, and communicates with the community.

---

## Reporting a Vulnerability

- Use the **GitHub Private Vulnerability Reporting** feature (Security tab) for any finding that *might* be memory corruption, a filesystem escape, RCE, or privilege escalation — **or whenever you are unsure** (fail-closed default)
- **Do NOT** open a public issue for findings in the **embargo class** (a demonstrated practical exploit, or a non-obvious finding plausibly exploitable beyond a crash). A finding you are **confident** is an out-of-scope robustness bug (e.g. a reproducible pure crash on a deliberately malformed model, with no memory-corruption claim) may be reported directly in public. See [SECURITY.md](SECURITY.md) for the full reporting policy.
- **Response target:** within 14 business days — ONNX is a volunteer-driven open-source project, so response times may vary

---

## Response Steps

### 1. Confirm
- Confirm this is a genuine security issue (not a bug or feature request)
- Assign an **Incident Lead** from the security team (the GitHub team with access to private advisories)
- Use the GitHub Security Advisory draft as the private coordination channel
- **Classify against the threat model before escalating:** apply the tier/primitive lookup from [SECURITY.md](SECURITY.md) and place ambiguous native-write, filesystem, or trust-related reports on a Clause-4 `provisional_in` hold rather than escalating immediately

### 2. Triage

**Classify against the threat model first (three axes).** Before assigning a CVE or escalating, the Incident Lead applies the model in [SECURITY.md](SECURITY.md):

1. **Scope** — tier + primitive lookup (is this in scope to fix?)
2. **Severity** — the CVSS v3.1 severity rubric, by primitive
3. **Disclosure** — the Clause-3 public-vs-embargo trigger

Ambiguous reports (suspected memory corruption without an ASan/Valgrind trace or write-primitive PoC) are placed on a Clause-4 **`provisional_in`** hold and run under the existing sanitizer CI, rather than escalated immediately. Routing a report that *claims* memory-safety/corruption to `out_of_scope` or `provisional_dos_out` is a status-changing, down-ranking call that **requires human maintainer sign-off** — never an autonomous AI decision.

Severity is based on [CVSS v3.1](https://www.first.org/cvss/) scores and assessed case by case. The security team (the GitHub team with access to private advisories) decides per incident whether the fix warrants an out-of-cycle patch release or can be included in the next scheduled release.

**Not every report results in a CVE.** A CVE is issued when there is a confirmed, exploitable vulnerability with real-world impact. Reports describing expected behavior, unrealistic preconditions, or issues outside the project's threat model **are closed without a CVE by default**, with this model's standard disposition note (clause + tier + missing-grounding) recorded on the advisory and the finding re-filed as a public robustness issue.

### 3. Fix
- Develop the patch privately (private fork or Security Advisory draft)
- Second maintainer reviews; run full test suite
- Prepare backport to current stable branch if needed

### 4. Disclose
1. Merge fix and release patched version
2. Publish the GitHub Security Advisory — this requests a CVE and serves as the public announcement

**Publish-or-dismiss SLA.** Every embargoed advisory is, within a fixed window (**90-day** default, after PyTorch), either **published** (fix shipped + CVE) or **dismissed** (downgraded to a public robustness bug and re-filed). A confirmed in-scope but not-yet-fixed advisory may receive a **logged, bounded embargo extension** (documented maintainer approval, recorded on the advisory) rather than being force-disclosed unfixed. A `provisional_dos_out` item is re-reviewed within the same window. See [SECURITY.md](SECURITY.md) for the full SLA.

### 5. Learn
- Blameless postmortem within 2 weeks
- Update this IRP if the process revealed gaps

---

## One-time backlog re-classification pass

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

This converts a contested all-in-one batch into a **transparent, per-advisory public record**.

---

## Quarterly Release Integration

| Phase | Security Action |
|-------|----------------|
| **Start of quarter** | Review open advisories. Update this IRP. |
| **Mid-quarter** | Develop fixes. Backport critical patches. |
| **Release candidate** | Final security review. Dependency audit. |
| **Release** | Note security fixes in changelog. Close advisories. |

**Out-of-cycle releases** are triggered for confirmed Critical/High vulnerabilities or active exploitation.

---

## Escalation

Disputes over triage, scope, severity, or disposition follow this chain:

**Maintainer triage → infra SIG → Steering Committee.**

A maintainer MAY downgrade or close an advisory by citing the specific clause, tier, or missing-grounding item. An unresolved dispute escalates to the infra SIG, and then to the Steering Committee if still unresolved.

---

*This is a living document, reviewed at the start of every quarterly release cycle. It fulfills the [OpenSSF OSPS Baseline](https://baseline.openssf.org/) requirement for coordinated vulnerability disclosure and incident response.*
