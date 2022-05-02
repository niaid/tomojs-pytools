###########
Development
###########

The docs related to development and testing are contained in this section.

*************
Prerequisites
*************

Please read before cloning.

Github.com
==========

The repository is located in the NIAID Github.com enterprise organization. Having a github.com account which is a member
of the NIAID organization is required.

Git LFS
=======

Git `Large File Storage <https://git-lfs.github.com>`_ (LFS) is used to store larger files in the repository such as
test images, trained models, and other data ( i.e. not text based code ). Before the repository is cloned, git lfs must
be installed on the system and set up on the users account. The `tool's documentation <https://git-lfs.github.com>`_
provides details on installation, set up, and usage that is not duplicated here. Once set up the git usage is usually
transparent with operation such as cloning, adding files, and changing branches.

The ".gitattributes" configuration file automatically places files in the directories "test/data" and "my_pkg/data" to
be stored in Git LFS.

*****************
Development Setup
*****************

Create a Python virtual environment for this project and install the development requirements:

.. code:: bash

  python -m venv venv-pkg
  source venv-pkg/bin/activate
  python -m pip install -r requirements-dev.txt


Only when needed (when?) the current development package can be installed in edit mode:

.. code:: bash

  pip install -e .


*******
Testing
*******

Docs about unit and integration tests (running them, writing them) are contained here.

The test driver and recommended testing framework is `pytest <https://docs.pytest.org]>`_. The pytest driver
automatically discovers procedures prefixed with `test_` to be run. The testing framework has many features including
assertions and fixtures, along with numerous examples to assist with getting started.



Running Unit Tests
=========================

Unit tests are for testing independent components and method separately. They can be run with the following command
line:

.. code:: bash

    python -m pytest test/unit


Running Integration Tests
=========================

The integration tests are for testing the package as a whole as intended to be used. They may require access to
additional services, or data. These can be run as followed:


.. code:: bash

    $ python -m pytest test/integration



Test Configuration
==================

In the root directory of the project, the working directory when `python -m pytest` is run, a "pytest.ini" file can be
created to set default options. For example to print all logging messages during test the following can be used:

::

 [pytest]
 log_cli = 1
 log_cli_level = DEBUG

************
Contributing
************

The repository is configured to work with a branchy workflow, where PRs are made before merging into the master branch.
Github Actions automatically run the linting and test suite, when a PR is made. The tests must pass before merging into
master.

Linting
=======

The linting process uses both `Black <https://black.readthedocs.io/en/stable/>`_  and
`flake8 <https://flake8.pycqa.org/en/latest/>`_ to ensure uncompromising code formatting and some programmatic problems.
The Black must be used to auto format new code before committing. As the continuous integration enforces black style, it
case safely be run on the whole repository without changing old code:

.. code:: bash

    python -m black .


Black is installed as part for the development requirements.

As part of the linting process the secret scanner `tufflehog3 <https://github.com/feeltheajf/truffleHog3>`_ is also
used.

********************
Sphinx Documentation
********************

`Sphinx <https://www.sphinx-doc.org/>`_ documentation as automatically rendered and pushed the the gh-pages branch. The
API is documented in Sphinx from the the Python docstring automatically for the public module methods and select private
methods.


********
Releases
********

The release of packages are automatic and triggered by pushing a git tag to the repository. The tags must be prefixed
with `v` followed by the version to trigger the release actions. Examples of tags are: "v0.1", "v1.0a1", "v1.0rc2",
"v1.0.1". A git tag can be create and push as follows:

.. code:: bash

    git tag "v0.1" -m "my_pkg release 0.1"
    git push origin "v0.1"

Versioning
==========

`Semantic Versions <https://semver.org>`_ practices should be used as guidelines for when major, minor, and patch
version number should change. The `PEP 440 -- Version Identification and Dependency Specification <https://www.python.org/dev/peps/pep-0440/>`_
should also be followed when cheating a tag, but without a "post" or "dev" suffix. The version of that package is
automatically determined by with the use of `setuptools_scm <https://github.com/pypa/setuptools_scm/>`_ introspection
of the git repositories tags. Only the git tags are used to determine the version, there is no need to hard code the
version anywhere in the code!