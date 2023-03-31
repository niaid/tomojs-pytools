from click.testing import CliRunner
import pytools.zarr_converter
import pytest
import SimpleITK as sitk
import json
from pathlib import Path

args = ["--help", "--version"]


@pytest.mark.parametrize("cli_args", args)
def test_zarr_converter_main_help(cli_args):
    runner = CliRunner()
    result = runner.invoke(pytools.zarr_converter.main, cli_args.split())
    assert not result.exception


@pytest.mark.parametrize(
    "image_mrc",
    [sitk.sitkUInt8, sitk.sitkInt16, sitk.sitkUInt16, sitk.sitkFloat32],
    indirect=["image_mrc"],
)
def test_zarr_converter1(image_mrc):
    runner = CliRunner()
    zarr_path = "output.zarr"
    with runner.isolated_filesystem():
        result = runner.invoke(pytools.zarr_converter.main, [image_mrc, zarr_path])
        assert not result.exception

        with open(Path(zarr_path) / ".zattrs") as fp:
            zattrs = json.load(fp)
            assert "multiscales" in zattrs
            assert "datasets" in zattrs["multiscales"][0]
