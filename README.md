<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

<p align="center"><img width="40%" src="https://github.com/onnx/onnx/raw/main/docs/onnx-horizontal-color.png" /></p>

[![PyPI - Version](https://img.shields.io/pypi/v/onnx.svg)](https://pypi.org/project/onnx)
[![CI](https://github.com/onnx/onnx/actions/workflows/main.yml/badge.svg)](https://github.com/onnx/onnx/actions/workflows/main.yml)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/3313/badge)](https://bestpractices.coreinfrastructure.org/projects/3313)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/onnx/onnx/badge)](https://api.securityscorecards.dev/projects/github.com/onnx/onnx)
[![REUSE compliant](https://api.reuse.software/badge/github.com/onnx/onnx)](https://api.reuse.software/info/github.com/onnx/onnx)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Open Neural Network Exchange (ONNX)](https://onnx.ai) is an open ecosystem that empowers AI developers
to choose the right tools as their project evolves. ONNX provides an open source format for AI models, both deep learning and traditional ML. It defines an extensible computation graph model, as well as definitions of built-in operators and standard
data types. Currently we focus on the capabilities needed for inferencing (scoring).

ONNX is [widely supported](http://onnx.ai/supported-tools) and can be found in many frameworks, tools, and hardware. Enabling interoperability between different frameworks and streamlining the path from research to production helps increase the speed of innovation in the AI community. We invite the community to join us and further evolve ONNX.

# Use ONNX

* [Documentation of ONNX Python Package](https://onnx.ai/onnx/)
* [Tutorials for creating ONNX models](https://github.com/onnx/tutorials)
* [Pre-trained ONNX models](https://github.com/onnx/models)

# Learn about the ONNX spec

* [Overview](https://github.com/onnx/onnx/blob/main/docs/Overview.md)
* [ONNX intermediate representation spec](https://github.com/onnx/onnx/blob/main/docs/IR.md)
* [Versioning principles of the spec](https://github.com/onnx/onnx/blob/main/docs/Versioning.md)
* [Operators documentation](https://github.com/onnx/onnx/blob/main/docs/Operators.md)
* [Operators documentation](https://onnx.ai/onnx/operators/index.html) (latest release)
* [Python API Overview](https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md)

# Programming utilities for working with ONNX Graphs

* [Shape and Type Inference](https://github.com/onnx/onnx/blob/main/docs/ShapeInference.md)
* [Graph Optimization](https://github.com/onnx/optimizer)
* [Opset Version Conversion](https://github.com/onnx/onnx/blob/main/docs/docsgen/source/api/version_converter.md)

# Contribute

ONNX is a community project and the open governance model is described [here](https://github.com/onnx/onnx/blob/main/community/readme.md). We encourage you to join the effort and contribute feedback, ideas, and code. You can participate in the [Special Interest Groups](https://github.com/onnx/onnx/blob/main/community/sigs.md) and [Working Groups](https://github.com/onnx/onnx/blob/main/community/working-groups.md) to shape the future of ONNX.

Check out our [contribution guide](https://github.com/onnx/onnx/blob/main/CONTRIBUTING.md) to get started.

If you think some operator should be added to ONNX specification, please read
[this document](https://github.com/onnx/onnx/blob/main/docs/AddNewOp.md).

# Community meetings

The schedules of the regular meetings of the Steering Committee, the working groups and the SIGs can be found [here](https://onnx.ai/calendar)

Community Meetups are held at least once a year. Content from previous community meetups are at:

* 2020.04.09 <https://wiki.lfaidata.foundation/display/DL/LF+AI+Day+-ONNX+Community+Virtual+Meetup+-+Silicon+Valley+-+April+9>
* 2020.10.14 <https://wiki.lfaidata.foundation/display/DL/LF+AI+Day+-+ONNX+Community+Workshop+-+October+14>
* 2021.03.24 <https://wiki.lfaidata.foundation/pages/viewpage.action?pageId=35160391>
* 2021.10.21 <https://wiki.lfaidata.foundation/pages/viewpage.action?pageId=46989689>
* 2022.06.24 <https://wiki.lfaidata.foundation/display/DL/ONNX+Community+Day+-+June+24>
* 2023.06.28 <https://wiki.lfaidata.foundation/display/DL/ONNX+Community+Day+2023+-+June+28>

# Discuss

We encourage you to open [Issues](https://github.com/onnx/onnx/issues), or use [Slack](https://lfaifoundation.slack.com/) (If you have not joined yet, please use this [link](https://join.slack.com/t/lfaifoundation/shared_invite/zt-o65errpw-gMTbwNr7FnNbVXNVFkmyNA) to join the group) for more real-time discussion.

# Follow Us

Stay up to date with the latest ONNX news. [[Facebook](https://www.facebook.com/onnxai/)] [[Twitter](https://twitter.com/onnxai)]

# Roadmap

A roadmap process takes place every year. More details can be found [here](https://github.com/onnx/steering-committee/tree/main/roadmap)

# Installation

ONNX released packages are published in PyPi.

```sh
pip install onnx # or pip install onnx[reference] for optional reference implementation dependencies
```

[ONNX weekly packages](https://pypi.org/project/onnx-weekly/) are published in PyPI to enable experimentation and early testing.

Detailed install instructions, including Common Build Options and Common Errors can be found [here](https://github.com/onnx/onnx/blob/main/INSTALL.md)

# Testing

ONNX uses [pytest](https://docs.pytest.org) as test driver. In order to run tests, you will first need to install `pytest`:

```sh
pip install pytest nbval
```

After installing pytest, use the following command to run tests.

```sh
pytest
```

# Development

Check out the [contributor guide](https://github.com/onnx/onnx/blob/main/CONTRIBUTING.md) for instructions.

# License

[Apache License v2.0](LICENSE)

# Code of Conduct

[ONNX Open Source Code of Conduct](https://onnx.ai/codeofconduct.html)
