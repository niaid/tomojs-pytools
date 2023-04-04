from click.testing import CliRunner
import pytools.zarr_build_multiscales
import pytest
import zarr
import shutil
from pathlib import Path

args = ["--help", "--version"]


@pytest.mark.parametrize("cli_args", args)
def test_zarr_build_multiscales_main_help(cli_args):
    runner = CliRunner()
    result = runner.invoke(pytools.zarr_build_multiscales.main, cli_args.split())
    assert not result.exception


def test_zarr_build_multiscales_main(image_ome_ngff):
    runner = CliRunner()
    with runner.isolated_filesystem():
        local_path = image_ome_ngff.name
        shutil.copytree(image_ome_ngff, local_path)

        print(Path(local_path).absolute())
        result = runner.invoke(
            pytools.zarr_build_multiscales.main,
            [
                str(local_path),
            ],
        )
        assert not result.exception

        store = zarr.DirectoryStore(local_path)
        group = zarr.group(store=store)
        print(list(group.array_keys()))
        print(dict(group.attrs))
