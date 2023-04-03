import zarr
import click
import logging
from pathlib import Path
from pytools import __version__
from pytools.utils.zarr import build_pyramid
from math import log2, ceil
import sys

logger = logging.getLogger(__name__)


@click.command()
@click.argument("input_zarr", type=click.Path(exists=True, dir_okay=True, readable=True, path_type=Path))
@click.option(
    "--chunk-size",
    default=None,
    show_default=True,
    type=click.IntRange(min=1),
    help="The size of zarr chunks stored in spatial dimensions. Defaults to the chunk size of the first array.",
)
@click.option(
    "--resolutions",
    "max_resolution",
    default=None,
    show_default=True,
    type=click.IntRange(min=1),
    help="Override the maximum number of resolutions. By default resolutions stop when smaller that the chunk shape.",
)
@click.option(
    "--overwrite/--no-overwrite", default=False, show_default=True, help="Overwrite output file if it exists."
)
@click.option(
    "--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False)
)
@click.version_option(__version__)
def main(input_zarr, overwrite, chunk_size, max_resolution, log_level):
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.getLevelName(log_level))

    store = zarr.DirectoryStore(input_zarr)
    group = zarr.group(store=store)
    logger.debug(group.info)

    if "multiscales" not in group.attrs:
        logger.error(f"Missing OME-NGFF multiscales meta data in zarr group: {input_zarr}")
        sys.exit(-1)

    for image_meta in group.attrs["multiscales"]:
        axes = image_meta["axes"]

        for i, dataset in enumerate(image_meta["datasets"]):
            if i == 0:
                base_path = dataset["path"]
                base_spacing = dataset["coordinateTransformations"][0]["scale"]
                logger.info(f"First scale path: {base_path}")
            else:
                if dataset["path"] in group:
                    if overwrite:
                        logger.debug(f"Deleting existing array: \"{dataset['path']}\"")
                        del group[dataset["path"]]
                    else:
                        logger.error(f"Multi-scale array already exists: {dataset['path']}")
                else:
                    logger.warning(f"multi-scales meta-data referred to non-existing zarr path: {dataset['path']}")

        # Shrink all spatial dimensions
        shrink_dim = [d for d, ax in enumerate(axes) if ax["type"].lower() == "space"][::-1]

        if chunk_size is None:
            chunk_dim = None
        else:
            chunk_dim = [chunk_size if ax["type"].lower() == "space" else 1 for ax in axes]

        multiscale_components = [
            "0",
        ]

        if base_path != multiscale_components[0]:
            group.store.rename(base_path, multiscale_components[0])

        if max_resolution is None:
            shape = group[base_path].shape
            _chunks = group[base_path].chunks if chunk_size is None else chunk_dim

            # the number of time the image can be reduces by 2 in shrink_dim until smaller than chunks
            max_resolution = 1
            for d, (s, c) in enumerate(zip(shape, _chunks)):
                if d in shrink_dim:
                    max_resolution = max(max_resolution, ceil(log2(s) - log2(c - 1)))

        multiscales = []
        data_item = {
            "path": multiscale_components[0],
            "coordinateTransformations": [{"type": "scale", "scale": base_spacing}],
        }
        # OME NGFF 3.3 "coordinateTransformations"
        multiscales.append(data_item)

        for level in range(1, max_resolution):
            multiscale_components.append(f"{level}")
            data_item = {
                "path": f"{level}",
                "coordinateTransformations": [
                    {
                        "type": "scale",
                        "scale": [
                            2**level * s if ax["type"].lower() == "space" else s for ax, s in zip(axes, base_spacing)
                        ],
                    }
                ],
            }
            multiscales.append(data_item)

        build_pyramid(input_zarr, multiscale_components, shrink=shrink_dim, chunks=chunk_dim, overwrite=overwrite)

        # Note: this does not save to the zarr attributes.
        image_meta["datasets"] = multiscales

    # Need to assign the list back to the zarr attributes dictionary to write to disk.
    group.attrs["multiscales"] = group.attrs["multiscales"]
