<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# ONNX Releases

The ONNX project, going forward, will plan to release roughly on a four month cadence. We follow the [Semver](https://semver.org/) versioning approach and will make decisions as a community on a release by release basis on whether to do a major or minor release.

## Preparation
* Install Twine, a utility tool to interact with PyPI. Do  - ``pip install twine``
* Get hold of the username and password for the ‘onnx’ PyPI account. Release manager should get onnx pypi account credentials from steering committee or from previous release manager.
* Bump the LAST_RELEASE_VERSION in [version.h](/onnx/common/version.h). Make sure that the IR version number and opset version numbers are up-to-date in
[ONNX proto files](/onnx/onnx.in.proto),
[Versioning.md](Versioning.md),
[schema.h](/onnx/defs/schema.h),
[helper.py](/onnx/helper.py) and [helper_test.py](/onnx/test/helper_test.py). Please note that this also needs to be happened in the main branch before creating the release branch.
* Pick a release tag (v.1.X.X) for the new release through mutual consent – Slack channel for Releases (https://lfaifoundation.slack.com/archives/C018VGGJUGK)
* Create a release branch (please use rel-* as the branch name) from main.
    * Make sure all tests pass on that branch.
    * Draft a release based on the branch:
        * https://github.com/onnx/onnx/releases/new
            * DO NOT click `Publish release`. Use `Save Draft` for now.
                * Publishing will create a tag which we don't want yet as there maybe bug fixes added during validation
        * git tag v1.X.X
        * Draft a new release statement listing out the new features and bug fixes, and potential changes being introduced in the release.
            * Use [pervious releases](https://github.com/onnx/onnx/releases) as a template
            * Use information from [Release logistics wiki](https://github.com/onnx/onnx/wiki) which should have been created prior to branch cut. (ie https://github.com/onnx/onnx/wiki/Logistics-for-ONNX-Release-1.XX.0)
* After cutting a release branch, bump [VERSION_NUMBER file](/VERSION_NUMBER) (next version number for future ONNX) in the `main` branch.
* Prepare a change log for the release –
    * ``git log --pretty=format:"%h - %s" <tag of the previous release>...<new tag>``
* Please use target VERSION_NUMBER with `rc` (e.g., `1.x.0rc1`) to test TestPyPI in advance before using target VERSION_NUMBER (e.g., `1.x.0`) for final release.
* Create an issue in onnxruntime repo. See [a sample issue](https://github.com/microsoft/onnxruntime/issues/11108) for details. The issue is to request onnxruntime to update with the onnx release branch and to run all CI and packaging pipelines ([How_To_Update_ONNX_Dev_Notes](https://github.com/microsoft/onnxruntime/blob/main/docs/How_To_Update_ONNX_Dev_Notes.md)). It is possible that onnx bugs are detected with onnxruntime pipeline runs. In such case the bugs shall be fixed in the onnx main branch and cherry-picked into the release branch. Follow up with onnxruntime to ensure the issue is resolved in time before onnx release.

## Upload to TestPyPI
**Wheels**
* In release branch update the version number in file [VERSION_NUMBER] to something like `1.x.0rc1` as release candidate for verification before finally using the targeted version number for this release.
* Windows
  * Use GitHub Action (`.github/workflows/release_win.yml`) under onnx repo to produce wheels for Windows.

* Linux
  * Use GitHub Action (`.github/workflows/release_linux_x86_64.yml`) and (`.github/workflows/release_linux_aarch64.yml`) under onnx repo to produce x64/aarch64 wheels for Linux.

* Mac
  * Use GitHub Action (`.github/workflows/release_mac.yml`) under onnx repo to produce wheels for Mac.

* After success, upload the produced wheels manually to TestPyPI: `twine upload --verbose *.whl --repository-url https://test.pypi.org/legacy/ -u PYPI_USERNAME -p PYPI_PASSWORD`.


**Source Distribution**
* Make sure all the git submodules are updated
    * ``git submodule update --init``
* Make sure the git checkout is clean –
    * Do ``git clean -nxd`` and make sure that none of the auto-generated header files *like* the following are not present.
        * onnx/onnx-operators.pb.cc
        * onnx/onnx-operator.pb.h
        * onnx/onnx.pb.cc
        * onnx/onnx.pb.h
    * If they are present run ``git clean -ixd`` and remove those files from your local branch
* Do ``python -m build --sdist`` to generate the source distribution.
* Do ``twine upload dist/* --repository-url https://test.pypi.org/legacy/ -u PYPI_USERNAME -p PYPI_PASSWORD`` to upload it to the test instance of PyPI.

## TestPyPI package verification
**Test ONNX itself**
* Test the PyPI package installation with different combinations of various Python versions, Protobuf versions and platforms.
  * Python versions : Applicable python versions for the release.
  * Protobuf versions : Latest protobuf version at the time of the release + protobuf version used for previous release


* After installing the PyPI package, run `pytest` in the release branch.

**Partner Validation**

 * Test with onnxruntime package: To test the interaction with onnxruntime, use ONNX functions like `load`, `checker.check_model`, `shape_inference.infer_shapes`, `save` with onnxruntime functions like `InferenceSession` and `InferenceSession.run` on certain example ONNX model. For example, run the test script from [test_with_ort.py](/onnx/test/test_with_ort.py) with installed onnxruntime package.

 * Test with ONNX converters: Create GitHub issues in converters repos to provide them the package links and have them test the TestPyPI packages.
   * https://github.com/pytorch/pytorch
   * https://github.com/onnx/onnx-tensorflow (not actively maintained)
   * https://github.com/onnx/tensorflow-onnx
   * https://github.com/onnx/sklearn-onnx
   * https://github.com/onnx/onnxmltools
   * https://github.com/onnx/onnx-tensorrt


**Source distribution verification**
* Test the source distribution by doing ``pip install --index-url https://test.pypi.org/simple --no-binary onnx onnx`` in a new environment.

## Upload to official PyPI
**NOTE: Once the packages are uploaded to PyPI, you cannot overwrite it on the same PyPI instance. Please make sure everything is good on TestPyPI before uploading to PyPI**

**Wheels**
* Windows/Linux_x86_64/Linux_aarch64/Mac
  * Create a new API token of onnx scope for uploading onnx wheel in https://pypi.org/manage/account (section of API tokens). Remove the created token after the release.
  * Similar to TestPyPI, use `twine upload --verbose *.whl --repository-url https://upload.pypi.org/legacy/ -u __token__ -p PYPI_API_TOKEN` instead.

**Source Distribution**
* Follow the same process in TestPyPI to produce the source distribution.
* Use ``twine upload --verbose dist/* --repository-url https://upload.pypi.org/legacy/`` instead to upload to the official PyPI.
* Test with ``pip install --use-deprecated=legacy-resolver --no-binary onnx onnx``

## After PyPI Release

**Release summary**
* Create release summary in github with the right tag and upload the release summary along with .tar.gz and .zip (these compressed files will be auto-generated after publishing the release summary).

**Announce**
* Announce in slack, for instance, `onnx-general` channel.
* Notify ONNX partners like converter team and runtime team.
* Create a news by updating `js/news.json` to announce ONNX release under [onnx/onnx.github.io](https://github.com/onnx/onnx.github.io) repo. For instance: https://github.com/onnx/onnx.github.io/pull/83.

**Update conda-forge package with the new ONNX version**
* Conda builds of ONNX are done via conda-forge, which runs infrastructure for building packages and uploading them to conda-forge. If it does not happen automatically, you need to submit a PR to https://github.com/conda-forge/onnx-feedstock (see https://github.com/conda-forge/onnx-feedstock/pull/1/files or https://github.com/conda-forge/onnx-feedstock/pull/50/files for example PRs) You will need to have uploaded to PyPI already, and update the version number and tarball hash of the PyPI uploaded tarball.

**Merge into main branch**
* After everything above is done, merge the release branch into the main branch to make it consistent. This step is needed only when there are urgent changes that are made directly into the release branch. The main branch does not have these needed changes. In all other circumstances, the merge PR shall show as empty so nothing needs to be merged.

**Remove old onnx-weekly packages on PyPI**
* Once ONNX has been released on PyPI, remove all previous versions of [onnx-weekly package](https://pypi.org/project/onnx-weekly/#history) on PyPI to save space.
* Steps: Login and go [here](https://pypi.org/manage/project/onnx-weekly/releases/) -> Choose target package -> Options -> Delete.

**Bump opset version for ai.onnx**
* Bump opset version for ai.onnx domain in `onnx/defs/operator_sets.h` and `onnx/defs/schema.h` for use by future operator additions and changes. For example, this [demo PR](https://github.com/onnx/onnx/pull/4134/files).

**Update IR TBD date if there is an IR bump in the release**
* Update the latest IR TBD date in https://github.com/onnx/onnx/blob/main/onnx/onnx.in.proto and regenerate corresponding proto files in the main branch if there is an IR bump in the release.
