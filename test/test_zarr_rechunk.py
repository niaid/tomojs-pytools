from click.testing import CliRunner
import pytools.zarr_rechunk
import pytest

args = ["--help", "--version"]


@pytest.mark.parametrize("cli_args", args)
def test_mrc2nifti_main_help(cli_args):
    runner = CliRunner()
    result = runner.invoke(pytools.zarr_rechunk.main, cli_args.split())
    assert not result.exception
