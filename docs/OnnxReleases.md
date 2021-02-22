<!--- SPDX-License-Identifier: Apache-2.0 -->

The ONNX project, going forward, will plan to release roughly on a two month cadence. We follow the [Semver](https://semver.org/) versioning approach and will make decisions as a community on a release by release basis on whether to do a major or minor release.

## Preparation

* Install Twine, a utility tool to interact with PyPI. Do  - ``pip install twine``
* Get hold of the username and password for the ‘onnx’ PyPI account. Release manager should get onnx pypi account credentials from steering committee or from previous release manager.
* Pick a release tag (v.1.X.X) for the new release through mutual consent – Slack channel for Releases (https://lfaifoundation.slack.com/archives/C018VGGJUGK)
* Prepare a change log for the release –
    * ``git log --pretty=format:"%h - %s" <tag of the previous release>...<new tag>``
    * And draft a new release statement - https://github.com/onnx/onnx/releases listing out the new features and bug fixes, and potential changes being introduced in the release.
* Before creating the release branch, increase `VERSION_NUMBER` in the main branch. The following files will be updated: [VERSION_NUMBER file](https://github.com/onnx/onnx/blob/master/VERSION_NUMBER) and
[version.h](../onnx/common/version.h)

* Please use a VERSION_NUMBER smaller than the target (release VERSION_NUMBER) and larger than the previous one to test TestPyPI before using the target VERSION_NUMBER. 

* Make sure that the IR version number and opset version numbers are up-to-date in
[ONNX proto files](../onnx/onnx.in.proto),
[Versioning.md](Versioning.md),
[schema.h](../onnx/defs/schema.h),
[helper.py](../onnx/helper.py) and [helper_test.py](../onnx/test/helper_test.py). Please note that this also needs to be happened in the main branch before creating the release branch.

* Create a release branch (please use rel-* as the branch name) from master. Checkout the release tag in a clean branch on your local repo. Make sure all tests pass on that branch.
* Create an issue in onnxruntime to update onnx commit in onnxruntime to the release branch commit and run all the CI and packaging pipelines.

## Upload to TestPyPI
**Wheels**
* In release branch update the version number in file [VERSION_NUMBER] to something like `1.x.0rc1` as release candidate for verification before finally using the targeted version number for this release.
* Windows
  * Use GitHub Action (`.github/workflows/release_win.yml`) under onnx repo to produce wheels for Windows.

* Linux
  * Use GitHub Action (`.github/workflows/release_linux_x86_64.yml`) and (`.github/workflows/release_linux_i686.yml`) under onnx repo to produce x64/i686 wheels for Linux.

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
* Do ``python setup.py sdist`` to generate the source distribution.
* Do ``twine upload dist/* https://test.pypi.org/legacy/ -u PYPI_USERNAME -p PYPI_PASSWORD`` to upload it to the test instance of PyPI.

## TestPyPI package verification
**Test ONNX itself**
* Test the PyPI package installation with different combinations of various Python versions, Protobuf versions and platforms.
  * Python versions : Applicable python versions for the release.
  * Protobuf versions : Latest protobuf version at the time of the release + protobuf version used for previous release
  * Utilize the following matrix to check:

    |   | 3.5 | 3.6 | 3.7 | 3.8 |
    -- | -- | -- | -- | -- |
    Linux |   |   |   |   |
    Windows |   |   |   |   |
    Mac |   |   |   |   |


* After installing the PyPI package, run `pytest` in the release branch.

**Partner Validation**

 * Test with onnxruntime package: To test the interaction with onnxruntime, use ONNX functions like `load`, `checker.check_model`, `shape_inference.infer_shapes`, `save` with onnxruntime functions like `InferenceSession` and `InferenceSession.run` on certain example ONNX model. For example, run the test script from ``.github/workflows/test_with_ort.py`` with installed onnxruntime package.

 * Test with ONNX converters: Create GitHub issues in converters repos to provide them the package links and have them test the TestPyPI packages.
   * https://github.com/pytorch/pytorch
   * https://github.com/onnx/onnx-tensorflow
   * https://github.com/onnx/tensorflow-onnx
   * https://github.com/onnx/sklearn-onnx
   * https://github.com/onnx/onnxmltools
   * https://github.com/onnx/keras-onnx
   * https://github.com/onnx/onnx-tensorrt
   * https://github.com/onnx/onnx-coreml


**Source distribution verification**
* Test the source distribution by doing ``pip install -i https://test.pypi.org/simple/ onnx`` in a new environment.

## Upload to official PyPI
**NOTE: Once the packages are uploaded to PyPI, you cannot overwrite it on the same PyPI instance. Please make sure everything is good on TestPyPI before uploading to PyPI**

**Wheels**
* Windows/Linux/Mac
  * Same as TestPyPI, use `twine upload --verbose *.whl --repository-url https://upload.pypi.org/legacy/ -u PYPI_USERNAME -p PYPI_PASSWORD` instead.

**Source Distribution**
* Follow the same process in TestPyPI to produce the source distribution.
* Use ``twine upload --verbose dist/* --repository-url https://upload.pypi.org/legacy/`` instead to upload to the official PyPI.
* Test with ``pip install --index-url https://upload.pypi.org/legacy/ onnx``

## After PyPI Release

**Release summary**
* Upload the source distribution, `.tar.gz` and `.zip`, in the release summary.
* Create release in github with the right tag and upload the release summary along with .tar.gz and .zip

**Announce**
* Announce in slack, for instance, `onnx-general` channel.
* Notify ONNX partners like converter team and runtime team.
* Create a news by updating `js/news.json` to announce ONNX release under [onnx/onnx.github.io](https://github.com/onnx/onnx.github.io) repo. For instance: https://github.com/onnx/onnx.github.io/pull/83.

**Update conda-forge package with the new ONNX version**
* Conda builds of ONNX are done via conda-forge, which runs infrastructure for building packages and uploading them to conda-forge. If it does not happen automatically, you need to submit a PR to https://github.com/conda-forge/onnx-feedstock (see https://github.com/conda-forge/onnx-feedstock/pull/1/files or https://github.com/conda-forge/onnx-feedstock/pull/50/files for example PRs) You will need to have uploaded to PyPI already, and update the version number and tarball hash of the PyPI uploaded tarball.

**Merge into main branch**
* After everything above is done, merge the release branch into the main branch to make it consistent.

## TODO list for next release
* Remove `onnx.optimizer` in ONNX 1.9
* Be aware of protobuf version gap issue (like building onnx with protobuf>=3.12 is not compatible with older protobuf)
* (Optional) Deprecate Python 3.5 and add Python 3.9.
* (Optional) Automatically upload created wheels for Windows
