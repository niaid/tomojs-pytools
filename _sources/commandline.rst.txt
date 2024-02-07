Command Line Interface
======================

The Pytools packages contain a command line executables for various tasks. They can be invoke directly as an executable:

.. code-block :: bash

   zarr_rechunk --help

Or the preferred way using the `python` executable to execute the module entry point:

.. code-block :: bash

    python -m zarr_rechunk --help

With either method of invoking the command line interface, the following sections describes the sub-commands available
and the command line options available.

.. click:: pytools.zarr_rechunk:main
    :prog: zarr_rechunk

.. click:: pytools.zarr_info:main
    :prog: zarr_info

