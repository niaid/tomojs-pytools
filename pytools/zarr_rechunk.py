import zarr
import click
import logging
from pathlib import Path
from pytools import __version__

logger = logging.getLogger(__name__)


def _chunk_logic_dim(drequest: int, dshape: int) -> int:
    if dshape > drequest > 0:
        return drequest
    return dshape


def rechunk_group(group: zarr.Group, chunk_size: int):
    logger.info(f'Processing group: "{group.name}"...')
    logger.debug(group)

    for group_name, child_group in group.groups():
        if group_name != "OME":
            rechunk_group(child_group, chunk_size)

    # grok through the OME-NGFF meta-dat, for each image scale (dataset/array) with axes information
    # https://ngff.openmicroscopy.org/latest/#multiscale-md
    if "multiscales" in group.attrs:
        for image in group.attrs["multiscales"]:
            chunk_request = tuple(chunk_size if a["type"] == "space" else -1 for a in image["axes"])

            for dataset in image["datasets"]:
                arr = group[dataset["path"]]
                logger.info(f'Processing array: "{arr.name}"...')
                logger.debug(arr.info)

                chunks = tuple(_chunk_logic_dim(r, s) for r, s in zip(chunk_request, arr.shape))
                if arr.chunks == chunks:
                    logger.info("Chunks already requested size")
                    continue

                group[dataset["path"]] = zarr.array(arr, chunks=chunks)
                logger.debug(group[dataset["path"]].info)


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
@click.version_option(__version__)
def main(input_zarr, log_level, chunk_size):
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.getLevelName(log_level))

    store = zarr.DirectoryStore(input_zarr)
    root = zarr.group(store=store)

    rechunk_group(root, chunk_size)


if __name__ == "__main__":
    main()
