<!--
SPDX-FileCopyrightText: Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# ONNX Incident Response Plan

**Repository:** [github.com/onnx/onnx](https://github.com/onnx/onnx) | **License:** Apache 2.0 | **Last Reviewed:** April 2026

This plan defines how the ONNX team receives vulnerability reports, assesses severity, ships fixes, and communicates with the community.

---

## Reporting a Vulnerability

- Use the **GitHub Private Vulnerability Reporting** feature (Security tab)
- **Do NOT** open a public issue for security vulnerabilities
- **Response target:** within 14 business days — ONNX is a volunteer-driven open-source project, so response times may vary

---

## Response Steps

### 1. Confirm
- Confirm this is a genuine security issue (not a bug or feature request)
- Assign an **Incident Lead** from the security team (the GitHub team with access to private advisories)
- Use the GitHub Security Advisory draft as the private coordination channel

### 2. Triage

Severity is based on [CVSS](https://www.first.org/cvss/) scores (v4.0 or v3.1) and assessed case by case. The security team (the GitHub team with access to private advisories) decides per incident whether the fix warrants an out-of-cycle patch release or can be included in the next scheduled release.

**Not every report results in a CVE.** A CVE is issued when there is a confirmed, exploitable vulnerability with real-world impact. Reports describing expected behavior, unrealistic preconditions, or issues outside the project's threat model may be closed without a CVE.

### 3. Fix
- Develop the patch privately (private fork or Security Advisory draft)
- Second maintainer reviews; run full test suite
- Prepare backport to current stable branch if needed

### 4. Disclose
1. Merge fix and release patched version
2. Publish the GitHub Security Advisory — this requests a CVE and serves as the public announcement

### 5. Learn
- Blameless postmortem within 2 weeks
- Update this IRP if the process revealed gaps

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

The escalation path will be confirmed and documented here.

---

*This is a living document, reviewed at the start of every quarterly release cycle. It fulfills the [OpenSSF OSPS Baseline](https://baseline.openssf.org/) requirement for coordinated vulnerability disclosure and incident response.*
