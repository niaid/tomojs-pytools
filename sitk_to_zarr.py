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
    z = da.zeros(shape=reader.GetSize()[::-1], dtype=sitk.extra._get_numpy_dtype(reader), chunks=chunks)

    def func(z, block_info=None):
        _reader = sitk.ImageFileReader()
        _reader.SetFileName(str(filename))
        if block_info is not None:
            size = block_info[None]["chunk-shape"][::-1]
            index = [al[0] for al in block_info[None]["array-location"][::-1]]
            _reader.SetExtractIndex(index)
            _reader.SetExtractSize(size)
            sitk_img = _reader.Execute()
            return sitk.GetArrayFromImage(sitk_img)
        return z

    da_img = da.map_blocks(func, z, meta=z, name="from-sitk")
    return da_img


def bin_shrink(img):
    """Reduces image by a factor of 2 in xyz and performs averaging.

    :param img: an array-like object of 3 dimensions
    """
    img = img.astype(np.float32)
    odd = [-1 if s % 2 else None for s in img.shape]
    img = img[:, :, 0 : odd[2] : 2] + img[:, :, 1::2]
    img = (
        img[0 : odd[0] : 2, 0 : odd[1] : 2, :]
        + img[1 : odd[0] : 2, 0 : odd[1] : 2, :]
        + img[0 : odd[0] : 2, 1 : odd[1] : 2, :]
        + img[1 : odd[0] : 2, 1 : odd[1] : 2, :]
    ) / 8
    # img = img[:,0::2,:]+img[:,1::2,:]
    # img = img[:,:,0::2]+img[:,:,1::2]
    return img.astype(np.uint8)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input_filename", help="Input file stream readable by SimpleITK")
    parser.add_argument("output_filename", help="Output directory with .zarr extension")
    ns = parser.parse_args()

    output_filename = ns.output_filename
    input_filename = ns.input_filename  # "/Users/blowekamp/scratch/dask/GZH-002-uint8.nii"

    compressor = zarr.Zlib(level=1)

    zarr_kwargs = {"compressor": compressor, "overwrite": False, "dimension_separator": "/"}

    with LocalCluster(n_workers=6, memory_limit="8GiB", processes=True, threads_per_worker=1) as cluster:
        with Client(address=cluster) as client:
            a = from_sitk(input_filename, chunks=(1, -1, -1))

            # OME NGFF 3.1 "axes"
            # https://ngff.openmicroscopy.org/latest/#axes-md
            axes = [
                {"name": "z", "type": "space", "unit": "nanometer"},
                {"name": "y", "type": "space", "unit": "nanometer"},
                {"name": "x", "type": "space", "unit": "nanometer"},
            ]

            # OME NGFF 3.4 "multiscales"
            # https://ngff.openmicroscopy.org/latest/#multiscale-md
            multiscales = []
            data_item = {"path": "base", "coordinateTransformations": [{"type": "scale", "scale": (1.0, 1.0, 1.0)}]}
            # OME NGFF 3.3 "coordiateTransformations"
            multiscales.append(data_item)

            print(f"Writing level {data_item['path']} at {a.shape} {data_item}...")
            da.to_zarr(a, output_filename, component=data_item["path"], **zarr_kwargs)
            a_b = da.from_zarr(output_filename, component=data_item["path"])

            l = 1
            while all([d > 1 for d in a_b.shape]):
                a_b = bin_shrink(a_b).rechunk((128, 128, 128))
                data_item = {"path": f"{l}", "coordinateTransformations": [{"type": "scale", "scale": (2**l,) * 3}]}
                multiscales.append(data_item)
                print(f"Writing level {data_item['path']}  at {a_b.shape} {data_item}...")

                da.to_zarr(a_b, output_filename, component=data_item["path"], **zarr_kwargs)
                a_b = da.from_zarr(output_filename, component=data_item["path"])
                l += 1

            g = zarr.group(output_filename)
            write_multiscales_metadata(g, datasets=multiscales, axes=axes)
