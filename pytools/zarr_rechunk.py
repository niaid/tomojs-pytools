import click
import logging
from pathlib import Path
from pytools import __version__
from pytools.HedwigZarrImages import HedwigZarrImages
from numcodecs import Blosc


@click.command()
@click.argument("input_zarr", type=click.Path(exists=True, dir_okay=True, readable=True, path_type=Path))
@click.option(
    "--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False)
)
@click.option(
    "--chunk-size",
    default=64,
    show_default=True,
    type=click.IntRange(min=1),
    help="The size of zarr chunks stored in spatial dimensions.",
)
@click.option(
    "--recompress",
    is_flag=True,
    show_default=True,
    default=False,
    help="Use the preferred compressor when recompressing.",
)
@click.version_option(__version__)
def main(input_zarr, log_level, chunk_size, recompress):
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.getLevelName(log_level))

    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)
    z = HedwigZarrImages(input_zarr, read_only=False)

    for k in z.get_series_keys():
        z[k].rechunk(chunk_size, compressor=compressor if recompress else None)


if __name__ == "__main__":
    main()
