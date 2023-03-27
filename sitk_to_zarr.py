#!/usr/bin/env python

import numpy as np

import dask.array as da
from ome_zarr.writer import write_multiscales_metadata
import zarr
import SimpleITK as sitk
import click


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

    print(f"chunks: {chunks} {reader.GetDimension()} {reader.GetNumberOfComponents()}")
    z = da.zeros(shape=img_shape, dtype=sitk.extra._get_numpy_dtype(reader), chunks=chunks)

    def func(z, block_info=None):
        _reader = sitk.ImageFileReader()
        _reader.SetFileName(str(filename))
        print(f"block_info: {block_info}")
        if block_info is not None:
            if is_multi_component:
                size = block_info[None]["chunk-shape"][-2::-1]
                index = [al[0] for al in block_info[None]["array-location"][-2::-1]]
            else:
                size = block_info[None]["chunk-shape"][::-1]
                index = [al[0] for al in block_info[None]["array-location"][::-1]]
            print(f"index: {index} size: {size}")
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
@click.argument("input_image", type=click.Path(exists=True, dir_okay=False, readable=True, resolve_path=True))
@click.argument("output_image", type=click.Path(exists=False, dir_okay=True, writable=True, resolve_path=True))
@click.option(
    "--alpha/--no-alpha", "alpha", default=True, show_default=True, help="When disabled removes the 4th channel."
)
def main(input_image, output_image, alpha):

    chunk_size = 512
    overwrite = True
    compression_level = 9

    compressor = zarr.Zlib(level=compression_level)

    zarr_kwargs = {"compressor": compressor, "overwrite": overwrite, "dimension_separator": "/"}

    reader = sitk.ImageFileReader()
    reader.SetFileName(input_image)
    reader.ReadImageInformation()

    # OME NGFF 3.1 "axes"
    # https://ngff.openmicroscopy.org/latest/#axes-md
    axes = [
        {"name": "z", "type": "space", "unit": "nanometer"},
        {"name": "y", "type": "space", "unit": "nanometer"},
        {"name": "x", "type": "space", "unit": "nanometer"},
    ]
    if reader.GetNumberOfComponents() > 1:
        axes.insert(0, {"name": "c^", "type": "channel"})

    # OME NGFF 3.4 "multiscales"
    # https://ngff.openmicroscopy.org/latest/#multiscale-md
    multiscales = []
    data_item = {"path": "base", "coordinateTransformations": [{"type": "scale", "scale": (1.0, 1.0, 1.0, 1.0)}]}
    # OME NGFF 3.3 "coordiateTransformations"
    multiscales.append(data_item)

    a = from_sitk(input_image)
    # make 3d
    if reader.GetDimension() == 2:
        a = a[None, ...]

    if reader.GetNumberOfComponents() > 1:
        if not alpha and a.shape[-1] == 4:
            a = a[..., 0:3]
        # make order zyxc->czyx
        a = da.moveaxis(a, -1, 0)

    _chunk_size = list(a.shape)
    shrink_dims = []
    for idx, ax in enumerate(axes):
        if ax["type"].lower() == "space":
            if a.shape[idx] != 1:
                _chunk_size[idx] = chunk_size
                shrink_dims.insert(0, idx)

    print(f"chunk size: {_chunk_size}")
    print(f"shrink_dims: {shrink_dims}")

    print(f"Writing level {data_item['path']} at {a.shape} {data_item}...")
    da.to_zarr(a.rechunk(chunks=_chunk_size), output_image, component=data_item["path"], **zarr_kwargs)
    a_b = da.from_zarr(output_image, component=data_item["path"])

    level = 1
    while any([a_b.shape[d] > _chunk_size[d] for d in range(a_b.ndim)]):
        a_b = bin_shrink(a_b, shrink_dims).rechunk(_chunk_size)
        data_item = {
            "path": f"{level}",
            "coordinateTransformations": [
                {
                    "type": "scale",
                    "scale": [2 ** level if ax["type"].lower() == "space" else 1.0 for ax in axes],
                }
            ],
        }
        multiscales.append(data_item)
        print(f"Writing level {data_item['path']}  at {a_b.shape} {data_item}...")

        da.to_zarr(a_b, output_image, component=data_item["path"], **zarr_kwargs)
        a_b = da.from_zarr(output_image, component=data_item["path"])
        level += 1

    g = zarr.group(output_image)
    write_multiscales_metadata(g, datasets=multiscales, axes=axes)


if __name__ == "__main__":
    main()
