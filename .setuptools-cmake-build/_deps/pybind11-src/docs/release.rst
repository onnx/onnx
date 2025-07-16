On version numbers
^^^^^^^^^^^^^^^^^^

The two version numbers (C++ and Python) must match when combined (checked when
you build the PyPI package), and must be a valid `PEP 440
<https://www.python.org/dev/peps/pep-0440>`_ version when combined.

For example:

.. code-block:: C++

    #define PYBIND11_VERSION_MAJOR X
    #define PYBIND11_VERSION_MINOR Y
    #define PYBIND11_VERSION_PATCH Z.dev1

For beta, ``PYBIND11_VERSION_PATCH`` should be ``Z.b1``. RC's can be ``Z.rc1``.
Always include the dot (even though PEP 440 allows it to be dropped). For a
final release, this must be a simple integer. There is also
``PYBIND11_VERSION_HEX`` just below that needs to be updated.


To release a new version of pybind11:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you don't have nox, you should either use ``pipx run nox`` instead, or use
``pipx install nox`` or ``brew install nox`` (Unix).

- Update the version number

  - Update ``PYBIND11_VERSION_MAJOR`` etc. in
    ``include/pybind11/detail/common.h``. PATCH should be a simple integer.

  - Update ``PYBIND11_VERSION_HEX`` just below as well.

  - Update ``pybind11/_version.py`` (match above).

  - Run ``nox -s tests_packaging`` to ensure this was done correctly.

- Ensure that all the information in ``setup.cfg`` is up-to-date, like
    supported Python versions.

- Add release date in ``docs/changelog.rst`` and integrate the output of
  ``nox -s make_changelog``.

  - Note that the ``nox -s make_changelog`` command inspects
    `needs changelog <https://github.com/pybind/pybind11/pulls?q=is%3Apr+is%3Aclosed+label%3A%22needs+changelog%22>`_.

  - Manually clear the ``needs changelog`` labels using the GitHub web
    interface (very easy: start by clicking the link above).

- ``git add`` and ``git commit``, ``git push``. **Ensure CI passes**. (If it
    fails due to a known flake issue, either ignore or restart CI.)

- Add a release branch if this is a new MINOR version, or update the existing
  release branch if it is a patch version

  - New branch: ``git checkout -b vX.Y``, ``git push -u origin vX.Y``

  - Update branch: ``git checkout vX.Y``, ``git merge <release branch>``, ``git push``

- Update tags (optional; if you skip this, the GitHub release makes a
  non-annotated tag for you)

  - ``git tag -a vX.Y.Z -m 'vX.Y.Z release'``

  - ``grep ^__version__ pybind11/_version.py``

    - Last-minute consistency check: same as tag?

  - ``git push --tags``

- Update stable

  - ``git checkout stable``

  - ``git merge -X theirs vX.Y.Z``

  - ``git diff vX.Y.Z``

  - Carefully review and reconcile any diffs. There should be none.

  - ``git push``

- Make a GitHub release (this shows up in the UI, sends new release
  notifications to users watching releases, and also uploads PyPI packages).
  (Note: if you do not use an existing tag, this creates a new lightweight tag
  for you, so you could skip the above step.)

  - GUI method: Under `releases <https://github.com/pybind/pybind11/releases>`_
    click "Draft a new release" on the far right, fill in the tag name
    (if you didn't tag above, it will be made here), fill in a release name
    like "Version X.Y.Z", and copy-and-paste the markdown-formatted (!) changelog
    into the description. You can use ``cat docs/changelog.rst | pandoc -f rst -t gfm``,
    then manually remove line breaks and strip links to PRs and issues,
    e.g. to a bare ``#1234``, without the surrounding ``<...>_`` hyperlink markup.
    Check "pre-release" if this is a beta/RC.

  - CLI method: with ``gh`` installed, run ``gh release create vX.Y.Z -t "Version X.Y.Z"``
    If this is a pre-release, add ``-p``.

- Get back to work

  - Make sure you are on master, not somewhere else: ``git checkout master``

  - Update version macros in ``include/pybind11/detail/common.h`` (set PATCH to
    ``0.dev1`` and increment MINOR).

  - Update ``pybind11/_version.py`` to match.

  - Run ``nox -s tests_packaging`` to ensure this was done correctly.

  - If the release was a new MINOR version, add a new ``IN DEVELOPMENT``
    section in ``docs/changelog.rst``.

  - ``git add``, ``git commit``, ``git push``

If a version branch is updated, remember to set PATCH to ``1.dev1``.

If you'd like to bump homebrew, run:

.. code-block:: console

    brew bump-formula-pr --url https://github.com/pybind/pybind11/archive/vX.Y.Z.tar.gz

Conda-forge should automatically make a PR in a few hours, and automatically
merge it if there are no issues.


Manual packaging
^^^^^^^^^^^^^^^^

If you need to manually upload releases, you can download the releases from
the job artifacts and upload them with twine. You can also make the files
locally (not recommended in general, as your local directory is more likely
to be "dirty" and SDists love picking up random unrelated/hidden files);
this is the procedure:

.. code-block:: bash

    nox -s build
    twine upload dist/*

This makes SDists and wheels, and the final line uploads them.
