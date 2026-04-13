# ONNX Incident Response Plan

**Repository:** [github.com/onnx/onnx](https://github.com/onnx/onnx) | **License:** Apache 2.0 | **Last Reviewed:** April 2026

This plan exists so the ONNX team can respond to security incidents calmly and consistently — without making decisions under pressure. It defines how we receive reports, assess severity, ship fixes, and communicate with the community. Having this in place before something goes wrong means we can focus on the problem, not the process.

---

## Reporting a Vulnerability

- **Use:** GitHub Private Vulnerability Reporting (Security tab) or [Huntr](https://huntr.com/) (AI/ML bug bounty platform)
- **Do NOT** open a public issue for security vulnerabilities
- **Response:** Within 7 business days

---

## Response Steps

### 1. Breathe & Confirm
- Confirm this is a genuine security issue (not a bug or feature request)
- Assign an **Incident Lead** from the security team
- Open a private channel (Slack DM or GitHub Security Advisory draft)

### 2. Triage

Severity is based on [CVSS](https://www.first.org/cvss/) scores (v4.0 or v3.1):

| Severity | CVSS | Examples | Fix Target |
|----------|------|----------|------------|
| **Critical** | 9.0–10.0 | RCE via crafted `.onnx` files, heap corruption bypassing checker | **14 days** — out-of-cycle patch |
| **High** | 7.0–8.9 | Integer overflow bypassing bounds checks, OOB read/write in shape inference, stack overflow DoS | **30 days** — patch or next release |
| **Medium** | 4.0–6.9 | Null pointer dereference via malformed nodes, crashes requiring unusual opset configurations | **60 days** — next quarterly release |
| **Low** | 0.1–3.9 | Crashes requiring local access and crafted inputs with no security impact | **90 days** — next quarterly release |

**Not every report results in a CVE.** A CVE is issued when there is a confirmed, exploitable vulnerability with real-world impact. Reports that describe expected behavior, require unrealistic preconditions, or fall outside the project's threat model may be closed without a CVE.

Ask: Which versions and opsets? Triggered by crafted model files? Which C++ component (shape inference, version converter, checker)? Requires attacker-controlled model input?

### 3. Fix
- Develop the patch in a **private fork or Security Advisory draft**
- Second maintainer reviews; run full test suite
- Prepare backport to current stable branch if needed

### 4. Disclose
1. Merge fix, release patched version to PyPI
2. Request CVE via GitHub CNA and publish GitHub Security Advisory
3. **Critical/High only:** notify downstream runtimes (ONNX Runtime, TensorRT) under **7-day embargo before** the above steps, and announce via Slack and social channels after

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

Reporter → Security Contacts → Core Maintainers → Linux Foundation (infrastructure compromise only)

---

*This is a living document, reviewed at the start of every quarterly release cycle. It fulfills the [OpenSSF OSPS Baseline](https://baseline.openssf.org/) requirement for coordinated vulnerability disclosure and incident response.*
