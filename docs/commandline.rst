Command Line Interface
======================

The Pytools packages contain a command line executables for varous tasks. They can be invoke directly as an executable:

.. code-block :: bash

    mrc_visual_min_max --help

Or the preferred way using the `python` executable to execute the module entry point:

.. code-block :: bash

    python -m mrc_visual_min_max --help

With either method of invoking the command line interface, the following sections descripts the sub-command available
and the command line options available.

.. click:: pytools.ng.mrc2nifti:main
   :prog: mrc2nifti

.. click:: pytools.ng.mrc2ngpc:main
   :prog: mrc2npgc

.. click:: pytools.ng.build_histogram:main
   :prog: mrc_visual_min_max
