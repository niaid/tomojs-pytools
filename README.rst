This package contains python based command line imaging utilities for the Hedwig project.

The Sphinx documentation can be found here:
https://niaid.github.io/tomojs-pytools


Installation
------------

This pytools is a proper Python package, and can be installed from published wheels, tarballs or from the git repository.

The wheels are packaged as part of the Github Actions and can be manually downloaded for the action interface. They are also published
to the bcbb-pypi hosted by NIAID actifactory. This extra repository can be configured with the
`--extra-index <https://pip.pypa.io/en/stable/cli/pip_install/>`_ option on the pip install command line. However, the argument needs
to include the username and password. The specifics can be found the the NIAID artifactory.

Use `pip` to install the tomojs_pytools package:

`python3 -m pip install -extra-index https://USERNAME:PASSWORD@artifactory.niaid.nih.gov/artifactory/api/pypi/bcbb-pypi/simple tomojs_pytools`

The requirements are specified conventionally in requirements.txt and
setup.py, so they will be enforced at installation time. Additionally,
the installation process installs the scripts as executables.

Alternatively, the package can be installed from the git repository:

`python3 -m pip https://github.com/niaid/tomojs-pytools.git@v0.6`
