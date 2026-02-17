<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# Security Policy

## Reporting a Vulnerability
If you believe you have discovered a security vulnerability in ONNX, please report it privately using GitHub Security Advisories.

👉 Open a private report: https://github.com/onnx/onnx/security/advisories/new

This allows maintainers to triage the issue, collaborate on a fix, and coordinate disclosure.

If you are unable to use GitHub for reporting, you may contact the maintainers at onnx-security@lists.lfaidata.foundation as a fallback.

After your report is received, a maintainer will acknowledge it, work with you to understand impact and remediation, and keep you informed about progress toward a fix and public disclosure.

Please do not disclose the vulnerability publicly until a fix and advisory have been released.

## Security announcements
Please subscribe to the [announcements mailing list](https://lists.lfaidata.foundation/g/onnx-announce), where we post notifications and remediation details for security vulnerabilities.

## Security Requirements

Open Neural Network Exchange (ONNX) manages reported vulnerabilities according to its documented security policy and delivers remediations in maintained releases. The project employs established secure development practices such as automated testing, continuous integration, and tooling intended to identify defects during development. Third-party dependencies and build components are periodically reviewed and updated to address known issues and to mitigate supply-chain risk. ONNX does not guarantee that models or inputs are trustworthy, and operators are responsible for validating provenance and applying appropriate isolation, resource limits, and runtime safeguards when executing untrusted workloads.
