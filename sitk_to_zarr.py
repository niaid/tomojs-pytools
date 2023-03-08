#!/usr/bin/env python

import argparse

import numpy as np

import dask.array as da
from dask.distributed import Client, LocalCluster
from ome_zarr.writer import write_multiscales_metadata
import zarr
import SimpleITK as sitk


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

    return img.astype(np.uint8)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input_filename", help="Input file stream readable by SimpleITK")
    parser.add_argument("output_filename", help="Output directory with .zarr extension")
    ns = parser.parse_args()

    output_filename = ns.output_filename
    input_filename = ns.input_filename  # "/Users/blowekamp/scratch/dask/GZH-002-uint8.nii"

    compressor = zarr.Zlib(level=9)

    zarr_kwargs = {"compressor": compressor, "overwrite": False, "dimension_separator": "/"}

    reader = sitk.ImageFileReader()
    reader.SetFileName(input_filename)
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

    # TODO: These parameters should not be hard coded values
    with LocalCluster(n_workers=8, memory_limit="8GiB", processes=True, threads_per_worker=1) as cluster:
        with Client(address=cluster) as client:

            a = from_sitk(input_filename)
            if reader.GetDimension() == 2:
                a = a[None, ...]
            if reader.GetNumberOfComponents() > 1:
                a = da.moveaxis(a, -1, 0)

            chunk_size = list(a.shape)
            chunk_size[-1] = 128
            chunk_size[-2] = 128
            if chunk_size[-3] > 128:
                chunk_size[-3] = 128
            print(f"chunk size: {chunk_size}")

            # Only shrink in ZYX
            shrink_dims = range(a.ndim - 1, a.ndim - 4, -1)

            print(f"Writing level {data_item['path']} at {a.shape} {data_item}...")
            da.to_zarr(a.rechunk(chunks=chunk_size), output_filename, component=data_item["path"], **zarr_kwargs)
            a_b = da.from_zarr(output_filename, component=data_item["path"])

            level = 1
            while all([d > 1 for d in a_b.shape[-2:]]):
                a_b = bin_shrink(a_b, shrink_dims).rechunk(chunk_size)
                data_item = {
                    "path": f"{level}",
                    "coordinateTransformations": [{"type": "scale", "scale": (1.0, 1.0, 2**level, 2**level)}],
                }
                multiscales.append(data_item)
                print(f"Writing level {data_item['path']}  at {a_b.shape} {data_item}...")

                da.to_zarr(a_b, output_filename, component=data_item["path"], **zarr_kwargs)
                a_b = da.from_zarr(output_filename, component=data_item["path"])
                level += 1

            g = zarr.group(output_filename)
            write_multiscales_metadata(g, datasets=multiscales, axes=axes)
