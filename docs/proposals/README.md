<!--
Copyright (c) ONNX Project Contributors
-->

<!--- SPDX-License-Identifier: Apache-2.0 -->

# ONNX RFCs
[ONNX RFCs]: #onnx-rfcs

The "RFC" (request for comments) process is intended to provide a consistent
and controlled path for changes to ONNX (such as new features) so that all
stakeholders can be confident about the direction of the project.

Many changes, including bug fixes and documentation improvements can be
implemented and reviewed via the normal GitHub pull request workflow without an RFC.

Some changes though are "substantial", and we ask that these be put through a
bit of a design process and produce a consensus among the ONNX community and
the relevant [Special Interest Groups](https://github.com/onnx/sigs/tree/main).

The template found in this folder is the (strongly recommended) starting point for new RFCs, but authors may deviate from it if needed.

## Life-cycle of an RFC

Before drafting up an RFC, it is recommended to first get in touch with relevant people and groups.
This may happen by creating a small issue in [`onnx/onnx`](https://github.com/onnx/onnx), asking questions on [slack](https://app.slack.com/client/TPUCV58TG/CPS6Q1600), or by joining relevant [sig meetings](https://github.com/onnx/sigs/tree/main).

After this initial phase, authors are encouraged to draft the RFC based on the template found in this folder and to open a PR.
The proposal is then reviewed and discussed within that PR.
The outcome of this process may either lead to the proposal being accepted or rejected, but it should be merged either way for future reference.

Generally, an accepted RFC should be a fairly stable and final affair due to a rigorous review process leading to the acceptance in the first place.
However, new circumstances and ideas may arise after an RFC has been accepted.
In such cases we may either choose to re-open the accepted RFC, or to create a new RFC.
