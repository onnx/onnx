<!--
Copyright (c) ONNX Project Contributors
-->

<!--- SPDX-License-Identifier: Apache-2.0 -->

# ONNX Community Involvement and Contribution Guidelines

ONNX is a community project and we welcome your contributions! In addition to contributing code, you can also contribute in many other ways:

- Meetings and Discussions

    Join SIGS, Working Groups, Community meetings to learn about what is needed and then where there is a good fit to interest and areas of expertise, find ways to actively contribute.  Participate in [ONNX technical discussions](https://github.com/onnx/onnx/discussions) on GitHub.  Join the ONNX Slack channels at LF AI and Data, help answer questions and welcome new members.

- Use Cases and Tools

    Develop use cases for ONNX and advocate for ONNX in developer conferences and meetups.  Develop tools that import and export using the ONNX spec, and help grow the community of ONNX users.  Become a champion for ONNX in your company or organization.

- Roadmap and Features

    Understand the ONNX roadmap document, feature priorities, and help implement them.  Become an ONNX code and documentation contributor, and work towards committer status on important repos.

- Releases and Model Zoo

    Help in achieving a release of ONNX, including increasing the number of models in the ONNX Model Zoo that exercise ONNX features.

- Publications and Blogs

    Add to the growing number of arXiv papers that refer to ONNX.  Create blogs, presentations, books, articles and other materials that help increase the adoption of ONNX, and grow the community of users and contributors.

- Steering Committee

    Attend ONNX Steering Committee meetings - they are open to all in the community. Help out where needed and appropriate on SC to-do items. Note that SIG and Working Groups leaders as well as others with demonstrated commitment and contributions to ONNX community may want to self-nominate during the annual SC election cycle.

## Adding a new operator or creating a new version of an existing operator

ONNX is an open standard, and we encourage developers to contribute high
quality operators to ONNX specification.

Before proposing a new operator, please read [the tutorial](docs/AddNewOp.md).

## Contributing code

You can submit a pull request (PR) with your code. The [SIG](community/sigs.md) or [Working Group](community/working-groups.md) that is responsible for the area of the project your PR touches will review it and merge once any comments are addressed.

### Development

To build ONNX from source please follow the instructions listed [here](https://github.com/onnx/onnx#build-onnx-from-source).

Then, after you have made changes to Python and C++ files:

- `Python files`: The changes are effective immediately in your installation. You don't need to install these again.
- `C++ files`: You need to install these again to trigger the native extension build.

Assuming build succeed in the initial step, simply running

```sh
pip install -e .
```

from onnx root dir should work.

### Folder structure

- `onnx/`: the main folder that all code lies under
  - `onnx.proto`: the protobuf that contains all the structures
  - `checker.py`: a utility to check whether a serialized ONNX proto is legal
  - `shape_inference.py`: a utility to infer types and shapes for ONNX models
  - `version_converter.py`: a utility to upgrade or downgrade version for ONNX models
  - `parser.py`: a utility to create an ONNX model or graph from a textual representation
  - `hub.py`: a utility for downloading models from [ONNX Model Zoo](https://github.com/onnx/models)
  - `compose.py`: a utility to merge ONNX models
  - `helper.py`: tools for graph operation
  - `defs/`: a subfolder that defines the ONNX operators
  - `test/`: test files

### Generated operator documentation

Operator docs ([Operators.md](Operators.md), [Operators-ml.md](Operators-ml.md)) and Changelog docs ([Changelog.md](Changelog.md), [Changelog-ml.md](Changelog-ml.md)) are automatically generated based on C++ operator definitions and backend Python snippets. To refresh all these docs, run the following commands from the repo root and commit the results by setting "ONNX_ML=1". By contrast, setting `ONNX_ML=0` will only update `Operators.md` and `Changelog.md`.

```pwsh
# Windows
set ONNX_ML=1
```

```sh
# UNIX
export ONNX_ML=1
pip install -e .
python onnx/defs/gen_doc.py
```

### Coding style

We use `lintrunner` to drive multiple linters defined in `.lintrunner.toml` to lint the codebase.

To run these checks locally, install `lintrunner` and the linters with

```sh
pip install lintrunner lintrunner-adapters
lintrunner init
```

Then lint with

```sh
lintrunner
```

format with

```sh
# Display all lints and apply the fixes
lintrunner -a
# Or apply fixes only (faster)
lintrunner f
```

Run `lintrunner --help` and see the `.lintrunner.toml` file for more usage examples, as well as instructions on how to adopt new linters.

### Testing

ONNX uses [pytest](https://docs.pytest.org) as a test driver. To run tests, you'll first need to install pytest:

```sh
pip install pytest nbval
```

After installing pytest, run from the root of the repo:

```sh
pytest
```

to run the tests.

<!-- TODO(justinchuby): Get rid of the need for manually running stat_coverage -->

You'll need to regenerate test coverage too, by running this command from the root of the repo:

```sh
python onnx/backend/test/stat_coverage.py
```

#### Cpp tests (googletest)

Some functionalities are tested with googletest. Those tests are listed in `test/cpp`, and include tests for shape inference, data propagation, parser, and others.

To run them, first build ONNX with `-DONNX_BUILD_TESTS=1` or `ONNX_BUILD_TESTS=1 pip install -e .`.

##### Linux and MacOS

The cpp tests require dynamically linking to built libraries.

```sh
export LD_LIBRARY_PATH="./.setuptools-cmake-build/:$LD_LIBRARY_PATH"
.setuptools-cmake-build/onnx_gtests
```

##### Windows

```pwsh
# If you set DEBUG=1, use `.setuptools-cmake-build\Debug\onnx_gtests.exe` instead
.setuptools-cmake-build\Release\onnx_gtests.exe
```

### DCO

ONNX has adopted the [DCO](https://en.wikipedia.org/wiki/Developer_Certificate_of_Origin). All code repositories under ONNX require a DCO. (ONNX previously used a CLA, which is being replaced with the DCO.)

DCO is provided by including a sign-off-by line in commit messages. Using the `-s` flag for `git commit` will automatically append this line. For example, running `git commit -s -m 'commit info.'` it will produce a commit that has the message `commit info. Signed-off-by: My Name <my_email@my_company.com>`. The DCO bot will ensure commits are signed with an email address that matches the commit author before they are eligible to be merged.

If you are using a GUI like the GitHub web site or GitHub Desktop, you'll need to append the `Signed-off-by: My Name <my_email@my_company.com>` manually to each commit message. For the onnx organization [sign-off](https://github.blog/changelog/2022-06-08-admins-can-require-sign-off-on-web-based-commits/) for web based commits is enabled. When this is activated you will see "Sign off and propose changes" instead of "Propose changes" when you are editing files directly at github. It is recommended to set this setting for your own fork as well. Since in the review process commits are made on this fork.

NOTE: the sign-off is needed for each commit in the PR, not at the PR level.

If you have old commits that are not signed, use the following commands to squash the old PR (original branch) into a single commit. This is an easier way to signoff old commits in old PR.

```bash
git checkout main
git checkout -b temporary_patch              # create a new branch as temporary
git merge --squash original_patch            # copy from old branch
git branch -d original_patch                 # remove old branch
git checkout -b original_patch               # create a new branch with the same name (override)
git commit -m 'type your own commit msg' -s  # signoff that single commit
git push origin original_patch -f            # forcibly override the old branch`
```

## CI Pipelines

Every PR needs to pass CIs before merge. CI pipelines details are [here](docs/CIPipelines.md).

## Other developer documentation

- [How to implement ONNX backend (ONNX to something converter)](docs/ImplementingAnOnnxBackend.md)
- [Backend test infrastructure and how to add tests](docs/OnnxBackendTest.md)

## License

[Apache License v2.0](/LICENSE)

## Code of Conduct

[ONNX Open Source Code of Conduct](http://onnx.ai/codeofconduct.html)
