The ONNX project, going forward, will plan to release roughly on a two month cadence. We follow the [Semver](https://semver.org/) versioning approach and will make decisions as a community on a release by release basis on whether to do a major or minor release.

Next expected release: May 15th 

Release Checklist: 

* Install Twine, a utility tool to interact with pypi. Do  - ``pip install twine``
* Get hold of the username and password for the ‘onnx’ pypi account.
* Pick a release tag for the new release through mutual consent – Gitter Room for Releases (https://gitter.im/onnx/Releases)
* Prepare a change log for the release – 
    * ``git log --pretty=format:"%h - %s" <tag of the previous release>...<new tag>``
    * And draft a new release statement - https://github.com/onnx/onnx/releases listing out the new features and bug fixes, and potential changes being introduced in the release.
* Checkout the release tag in a clean branch on your local repo. Make sure all tests pass on that branch.
* Make sure the VERSION_NUMBER file has up-to-date version - https://github.com/onnx/onnx/blob/master/VERSION_NUMBER
* Make sure all operators version are even number, if opset version is odd number then increment it and update all corresponding OPs.
* Make sure all the git submodules are updated
    * ``git submodule update –init``
* Make sure the git checkout is clean –
    * Do ``git clean -nxd`` and make sure that none of the auto-generated header files *like* the following are not present.
        * onnx/onnx-operators.pb.cc
        * onnx/onnx-operator.pb.h
        * onnx/onnx.pb.cc
        * onnx/onnx.pb.h
    * If they are present run ``git clean -ixd`` and remove those files from your local branch
* Do ``python setup.py sdist`` to generate the source distribution.
* Put the following content into ``~/.pypirc`` file:
```
[distutils]
index-servers =
  pypi
  pypitest
 
[pypi]
username=<username>
password=<password>
 
[pypitest]
repository=https://test.pypi.org/legacy/
username=<username>
password=<password>
```
* Do ``twine upload dist/* -r pypitest`` to upload it to the test instance of pypi.
* After uploading to pypitest, you can test the source distribution by doing ``pip install –index-url https://test.pypi.org/simple/ onnx`` in a new environment. Test this installation with different environments and versions of protobuf binaries.
* *NOTE - Once a distribution is uploaded to pypi, you cannot overwrite it on the same pypi instance.*
* Once completely verified do, ``twine upload dist/*``  to upload to the official pypi.
* *Conda - *
    * Conda builds of ONNX are done via conda-forge, which runs infrastructure for building packages and uploading them to conda-forge. You need to submit a PR to https://github.com/conda-forge/onnx-feedstock (see https://github.com/conda-forge/onnx-feedstock/pull/1/files for an example PR.) You will need to have uploaded to PyPi already, and update the version number and tarball hash of the PyPi uploaded tarball.
* The release branch needs to be marked as a protected branch to preserve it. The branch corresponding to each release must be preserved. (You may need to ask an admin to do this)


 


