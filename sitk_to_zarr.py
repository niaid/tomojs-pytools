#!/usr/bin/env python

import numpy as np

import dask.array as da
from ome_zarr.writer import write_multiscales_metadata
import zarr
import SimpleITK as sitk
import click
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


def from_sitk(filename, chunks=None):
    """
    Reads the filename into a dask array with map_block.

    Under the hood SimpleITK is used to stream read the chunk of the array if supported.

    ITK support full streaming includes MHA, MRC, NRRD and NIFTI file formats.

    """
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(filename))
    reader.ReadImageInformation()
    img_shape = reader.GetSize()[::-1]

    # default to loading the whole image from file
    if chunks is None:
        chunks = (-1,) * reader.GetDimension()

    is_multi_component = reader.GetNumberOfComponents() != 1

    if is_multi_component:
        img_shape = img_shape + (reader.GetNumberOfComponents(),)

        if len(chunks) < len(img_shape):
            chunks = chunks + (-1,)

    logger.debug(f"chunks: {chunks} {reader.GetDimension()} {reader.GetNumberOfComponents()}")
    z = da.zeros(shape=img_shape, dtype=sitk.extra._get_numpy_dtype(reader), chunks=chunks)

    def func(z, block_info=None):
        _reader = sitk.ImageFileReader()
        _reader.SetFileName(str(filename))
        logger.debug(f"block_info: {block_info}")
        if block_info is not None:
            if is_multi_component:
                size = block_info[None]["chunk-shape"][-2::-1]
                index = [al[0] for al in block_info[None]["array-location"][-2::-1]]
            else:
                size = block_info[None]["chunk-shape"][::-1]
                index = [al[0] for al in block_info[None]["array-location"][::-1]]

            _reader.SetExtractIndex(index)
            _reader.SetExtractSize(size)
            sitk_img = _reader.Execute()
            return sitk.GetArrayFromImage(sitk_img)
        return z

    da_img = da.map_blocks(func, z, meta=z, name="from-sitk")
    return da_img


def bin_shrink(img, shrink_dim=None):
    """Reduces image by a factor of 2 in xyz and performs averaging.

    :param img: an array-like object of 3 dimensions
    :param shrink_dim: an iterable for dimensions to perform operation on. Recommended to order from the fastest axis to
     slowest for performance.
    """
    input_type = img.dtype
    img = img.astype(np.float32)
    # When odd, set end index to -1 to drop odd entry
    stop_index = [-1 if s % 2 and s != 1 else None for s in img.shape]

    if shrink_dim is None:
        shrink_dim = range(img.ndim)[::-1]

    for i in shrink_dim:
        if img.shape[i] <= 1:
            continue
        idx1 = tuple(slice(0, stop_index[i], 2) if j == i else slice(None) for j in range(img.ndim))
        idx2 = tuple(slice(1, stop_index[i], 2) if j == i else slice(None) for j in range(img.ndim))

        img = (img[idx1] + img[idx2]) / 2

    return img.astype(input_type)


@click.command()
@click.argument("input_image", type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path))
@click.argument("output_image", type=click.Path(exists=False, dir_okay=True, writable=True, path_type=Path))
@click.option(
    "--alpha/--no-alpha", "alpha", default=True, show_default=True, help="When disabled removes the 4th channel."
)
@click.option(
    "--overwrite/--no-overwrite", default=False, show_default=True, help="Overwrite output file if it exists."
)
@click.option(
    "--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False)
)
@click.option("--chunk-size", default=64, type=click.IntRange(min=1))
def main(input_image, output_image, alpha, overwrite, chunk_size, log_level):

    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.getLevelName(log_level))

    compression_level = 9

    compressor = zarr.Zlib(level=compression_level)

    zarr_kwargs = {"compressor": compressor, "overwrite": overwrite, "dimension_separator": "/"}

    has_channels = False
    reader_type = input_image.suffix.lstrip(".")
    if reader_type in ["png", "mrc"]:
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(input_image))
        reader.ReadImageInformation()

        has_channels = reader.GetNumberOfComponents() > 1
        base_spacing = list(reader.GetSpacing())

        a = from_sitk(input_image)

        if reader.GetNumberOfComponents() > 1:
            # make order zyxc->czyx
            a = da.moveaxis(a, -1, 0)
            base_spacing.insert(0, 1.0)

    if not alpha and a.shape[0] == 4:
        a = a[0:3, ...]

    # OME NGFF 3.1 "axes"
    # https://ngff.openmicroscopy.org/latest/#axes-md
    axes = [
        {"name": "z", "type": "space", "unit": "nanometer"},
        {"name": "y", "type": "space", "unit": "nanometer"},
        {"name": "x", "type": "space", "unit": "nanometer"},
    ]

    if has_channels:
        axes.insert(0, {"name": "c", "type": "channel"})

    # OME NGFF 3.4 "multiscales"
    # https://ngff.openmicroscopy.org/latest/#multiscale-md
    multiscales = []

    data_item = {"path": "base", "coordinateTransformations": [{"type": "scale", "scale": base_spacing}]}
    # OME NGFF 3.3 "coordiateTransformations"
    multiscales.append(data_item)

    chunk_dims = list(a.shape)
    shrink_dims = []
    for idx, ax in enumerate(axes):
        if ax["type"].lower() == "space":
            if a.shape[idx] != 1:
                chunk_dims[idx] = chunk_size
                shrink_dims.insert(0, idx)
    logger.debug(f"chunk size: {chunk_dims}")
    logger.debug(f"shrink_dims: {shrink_dims}")

    logger.info(f"Writing level {data_item['path']} at {a.shape} {data_item}...")
    da.to_zarr(a.rechunk(chunks=chunk_dims), output_image, component=data_item["path"], **zarr_kwargs)
    a_b = da.from_zarr(output_image, component=data_item["path"])

    level = 1
    while any([a_b.shape[d] > chunk_dims[d] for d in range(a_b.ndim)]):
        a_b = bin_shrink(a_b, shrink_dims).rechunk(chunk_dims)
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
        logger.info(f"Writing level {data_item['path']}  at {a_b.shape} {data_item}...")

        da.to_zarr(a_b, output_image, component=data_item["path"], **zarr_kwargs)
        a_b = da.from_zarr(output_image, component=data_item["path"])
        level += 1

    g = zarr.group(output_image)
    write_multiscales_metadata(g, datasets=multiscales, axes=axes)


if __name__ == "__main__":
    main()
