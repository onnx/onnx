The ONNX project, going forward, will plan to release roughly on a two month cadence. We follow the [Semver](https://semver.org/) versioning approach and will make decisions as a community on a release by release basis on whether to do a major or minor release.

## Preparation

* Install Twine, a utility tool to interact with PyPI. Do  - ``pip install twine``
* Get hold of the username and password for the ‘onnx’ PyPI account. Release Manager should obtain this information from last Release Manager or certain private repo under onnx.
* Pick a release tag (v.1.X.X) for the new release through mutual consent – Slack channel for Releases (https://lfaifoundation.slack.com/archives/C018VGGJUGK)
* Prepare a change log for the release – 
    * ``git log --pretty=format:"%h - %s" <tag of the previous release>...<new tag>``
    * And draft a new release statement - https://github.com/onnx/onnx/releases listing out the new features and bug fixes, and potential changes being introduced in the release.
* Increase `VERSION_NUMBER` in the main branch before creating a release branch.
* Create a release branch (please use rel-* as the branch name) from master. Checkout the release tag in a clean branch on your local repo. Make sure all tests pass on that branch.
* Create an issue in onnxruntime to update onnx commit in onnxruntime to the release branch commit and run all the CI and packaging pipelines.
* In the release branch, make sure that the Release version number information is up-to-date in the following places:
[VERSION_NUMBER file](https://github.com/onnx/onnx/blob/master/VERSION_NUMBER) and
[version.h](../onnx/common/version.h)
* Make sure that the IR version number and opset version numbers are up-to-date in
[ONNX proto files](../onnx/onnx.in.proto),
[Versioning.md](Versioning.md), 
[schema.h](../onnx/defs/schema.h), 
[helper.py](../onnx/helper.py) and [helper_test.py](../onnx/helper_test.py).

## Upload to TestPyPI
**Wheels**
* In release branch update the version number in file [VERSION_NUMBER] to something like `1.x.0rc1` as release candidate for verification before finally using the targeted version number for this release.
* Windows
  * Use GitHub Action (`.github/workflows/release_win.yml`) under onnx repo to produce wheels for Windows.
  * After success, upload the produced wheels manually to TestPyPI: `twine upload --verbose *.whl --repository-url https://test.pypi.org/legacy/ -u PYPI_USERNAME -p PYPI_PASSWORD`.
* Linux and Mac
  * Use Travis CI from [onnx/wheel-builder](https://github.com/onnx/wheel-builder) repo to produce and automatiallcy upload wheels for Linux and Mac.
  * Update `BUILD_COMMIT` as the commit ID of latest release branch in `.travis.yml` and update `ONNX_NAMESPACE` as `ONNX_REL_1_X` in `config.sh `. For example: https://github.com/onnx/wheel-builder/pull/27.
  * Update `$PYPI_USERNAME` and `$PYPI_PASSWORD` as `- secure: ***` in `.travis.yml`. Create the encrypted variables for these variables by `travis encrypt` in your local machine.
  Reference: https://docs.travis-ci.com/user/environment-variables/#defining-encrypted-variables-in-travisyml
  * Only `pypi-test` branch will automatiallcy upload created wheels to TestPyPI.
  * Currently Python 3.5 on Mac cannot upload wheel successfully in Travis CI. In that case, you need to upload the created wheels to AWS S3 bucket, get the wheel from AWS and upload it manually (same as Windows). To upload to AWS, updade your `ARTIFACTS_KEY`, `ARTIFACTS_SECRET` and `ARTIFACTS_BUCKET` by `travis encrypt`. Reference: https://docs.travis-ci.com/user/uploading-artifacts/

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
* Test the source distribution by doing ``pip install --index-url https://test.pypi.org/legacy/ onnx`` in a new environment.

## Upload to official PyPI
**NOTE: Once the packages are uploaded to PyPI, you cannot overwrite it on the same PyPI instance. Please make sure everything is good on TestPyPI before uploading to PyPI**

**Wheels**
* Windows
  * Same as TestPyPI, use `twine upload --verbose *.whl --repository-url https://upload.pypi.org/legacy/ -u PYPI_USERNAME -p PYPI_PASSWORD` instead.
* Linux and Mac
  * Similar to TestPyPI. In wheel-builder repo, merge `pypi-test` branch to main branch and create a new Release with main branch and tag to trigger Travis CI. This will automatically upload PyPI packages after successful CI run.


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
* (Optional) Move Linux and Mac release pipelines in onnx/wheel-builder to GitHub Action in onnx repo
* (Optional) Deprecate Python 3.5. It has been officially deprecated by Python and some problems exist in Travis CI for Mac.
* (Optional) Automatically upload created wheels for Windows
