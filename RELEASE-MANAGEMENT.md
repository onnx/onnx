# ONNX release management

This describes the process by which versions of ONNX are officially released to the public.

Releases
--------

Releases are versioned according to [docs/Versioning.md](docs/Versioning.md). This describes IR and operator versioning policies, as well as propose how models themselves should be versioned.

On a regular basis, new versions of ONNX are published, representing the aggregate of changes in the IR and operator sets. Such releases use semantic versioning to describe the progression of the standard.

The GitHub repo for ONNX provides release branches where the project is stabilized as per the process described here. Release notes are used to communicate the stability and status of a release. The master branch will be used to continue work for subsequent releases.

Major, minor and patch releases will have branch names and version numbers reflecting the nature of the change as per semantic versioning definitions.

Workflow
--------

The following workflow describes the steps taken to release an update of ONNX,
and can be undertaken regardless of whether a major, minor or patch release is
to be produced.

- The trigger for the workflow will typically be a time-based trigger based on
  elapsed time (say every three months), but for the first release, we will
  start the process in early November 2017.

- The release manager will announce the intent of the process (to produce a
  major, minor or patch update) and the overall timeline. A release branch is
  created with the name rel-major#.minor#(.patch#), and any version
  references in build scripts or version checks are updated.

- The release manager announces the initial commit for testing. The first
  period lasts a week; any regressions found should be fixed, typically via
  the master branch. Incomplete features should be done or excised during this
  period. A distribution can be made available with an -RC1 suffix.

- The release manager announces a second round of testing (unless it's only a
  patch update with no regressions found). Only critical bugs are fixed at
  this point, or those introduced by patches from the first week. A third
  weeek may be introduced at the release manager's discretion if significant
  fixes need to be taken. Distributions with -RCn suffixes can be made
  available if convenient.

- Release notes are updated with final changes, and a file with sources is
  provided along with a release on the GitHub project page.

Testing
-------

The release process really consists of communicating, testing to establish a
known state of the project, and distributing files. This section deals with
the second task.

At the very least, the tests that are part of the /test folder should be run
under a variety of configurations. Issues fixed should ensure coverage in this
suite to avoid regressions.

Send a Pull Request for updates to this section to include a configuration you
can help test if you care about one that's missing.

The community is encouraged to perform additional testing during the test
periods. Bugs and issues should be filed in the ONNX GitHub repo.

Distribution
------------

The distribution of files follows the basic procedures described in
[Creating Releases](https://help.github.com/articles/creating-releases/).

