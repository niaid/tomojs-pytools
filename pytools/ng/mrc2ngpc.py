#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0.txt
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import click
import subprocess
from pytools import __version__
import tempfile
import os.path
import sys
import logging
from pathlib import Path


@click.command()
@click.argument("input_image", type=click.Path(exists=True, dir_okay=False, resolve_path=True))
@click.argument("output_precompute", type=click.Path(exists=False, dir_okay=True, resolve_path=True))
@click.option(
    "--flat/--no-flat",
    default=True,
    show_default=True,
    help="Produce all chunks for a resolution in the same directory.",
)
@click.option("--gzip/--no-gzip", default=False, show_default=True, help="Compress each chunk.")
@click.version_option(__version__)
def main(input_image, output_precompute, flat, gzip):
    """Reads the INPUT_IMAGE as an MRC formatted file and creates a Neuroglancer precompute pyramid in
    OUTPUT_PRECOMPUTE.

    The OUTPUT_PRECOMPUTE path must not exist.

    The conversion is done in two steps first to a NIFTI file, then to precompute format. A temporary directory and file
    next to the OUTPUT_PRECOMPUTE path is used.

    Additionally, in the same directory as OUTPUT_PRECOMPUTE, a mrc2ngpc-output.json file is created with additional
    meta-data.
    """

    logger = logging.getLogger()

    if os.path.exists(output_precompute):
        raise IOError(f"Output '{output_precompute}' already exists!")

    output_path = os.path.dirname(output_precompute)

    with tempfile.TemporaryDirectory(prefix=".", dir=output_path) as temp_path:

        nifti_extension = ".nii"
        nifti_filename = os.path.join(temp_path, os.path.basename(output_precompute) + nifti_extension)

        logger.debug(f"Temporary NIFTI file: {nifti_filename}")

        # Execute task as sub-processing binding stderr/stdout with automatic python exception is process failure.
        check_process = True

        # Task 1: Convert mrc to NIFTI with SimpleITK script
        py_code_main = "import sys; from pytools.ng.mrc2nifti import main; sys.exit(main())"
        cmd = [sys.executable, "-c", py_code_main, input_image, nifti_filename]

        logger.info("Executing conversion of MRC to NIFTI..")
        logger.debug(f"Executing: {cmd}")
        subprocess.run(cmd, check=check_process)

        # Task 2: Convert NIFTI to ng precompute pyramid with neruroglancer-scripts script

        cmd_opts = ["--downscaling-method=average"]
        if flat:
            cmd_opts.append("--flat")
        if not gzip:
            cmd_opts.append("--no-gzip")

        cmd = ["volume-to-precomputed-pyramid"] + cmd_opts + [nifti_filename, output_precompute]

        logger.info("Executing conversion of NIFTI to Neruoglancer precompute pyramid...")
        logger.debug(f"Executing: {cmd}")
        subprocess.run(cmd, check=check_process)

        # Task 3: Process NIFTI file to create visualization min/max

        json_output = os.path.join(output_path, "mrc2ngpc-output.json")
        cmd_opts = ["--mad", "5", "--output-json", str(json_output)]
        py_code_main = "import sys; from pytools.ng.build_histogram import main; sys.exit(main())"
        cmd = [sys.executable, "-c", py_code_main, nifti_filename, *cmd_opts]

        logger.info("Computing visualization min max...")
        logger.debug(f"Executing: {cmd}")
        subprocess.run(cmd, check=check_process)


if __name__ == "__main__":
    main()
