from click.testing import CliRunner
import shutil
from pathlib import Path
import pytest
import pytools.zarr_rechunk
from pytools.HedwigZarrImages import HedwigZarrImages

args = ["--help", "--version"]


@pytest.mark.parametrize("cli_args", args)
def test_zarr_rechunk_main_help(cli_args):
    runner = CliRunner()
    result = runner.invoke(pytools.zarr_rechunk.main, cli_args.split())
    assert not result.exception


@pytest.mark.parametrize("chunk_size", [512, 64])
def test_zarr_rechunk_main(image_ome_ngff, chunk_size):
    runner = CliRunner()
    with runner.isolated_filesystem():
        local_path = image_ome_ngff.name
        shutil.copytree(image_ome_ngff, local_path)

        result = runner.invoke(pytools.zarr_rechunk.main, args=[str(local_path), "--chunk-size", chunk_size])
        assert not result.exception

        zi = HedwigZarrImages(Path(local_path).absolute(), read_only=False)
        for k, hzi in zi.series():
            for level in range(len(hzi._ome_ngff_multiscales()["datasets"])):
                arr = hzi._ome_ngff_multiscale_get_array(level)
                # check for expected chunking
                for s, c, d in zip(arr.shape, arr.chunks, hzi._ome_ngff_multiscale_dims()):
                    if d == "T":
                        assert s == 1
                    elif d == "C" or s < chunk_size:
                        assert s == c
                    else:
                        assert c == chunk_size
