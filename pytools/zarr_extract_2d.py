#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0.txt
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

from pathlib import Path
from typing import Union, List
import logging
import SimpleITK as sitk
import zarr
import dask.array as da

logger = logging.getLogger(__name__)


def _ngff_get_cononical_axes_names(zarr_group: zarr.hierarchy.Group) -> List[str]:
    """Returns the canonical axes names of the OME-NGFF pyramid structured ZARR array.

    The names are of the form "t", "c", "x", "y", "z". The order of the names is the same as the order of the axes
     in the ZARR array.

    The name for the time dimension is explicitly "t". The name for the channel dimension is explicitly "c".
    The names for the spatial dimensions are preserved from the metadata and must be one of "x", "y", "z".

    :param zarr_group: The OME-NGFF pyramid structured ZARR array.
    :return: The canonical axes names of the OME-NGFF pyramid structured ZARR array.
    """

    axes = zarr_group.attrs["multiscales"][0]["axes"]
    axes_names = []
    for ax in axes:
        if ax["type"].lower() == "time":
            axes_names.append("t")
        elif ax["type"].lower() == "channel":
            axes_names.append("c")
        else:
            name = ax["name"].lower()
            if name not in ["x", "y", "z"]:
                raise ValueError(f"Unexpected axis name: {name}")
            axes_names.append(ax["name"].lower())

    return axes_names


def _ngff_get_max_size(zarr_group: zarr.hierarchy.Group) -> List[int]:
    """Returns the maximum size of the OME-NGFF pyramid structured ZARR array.

    :param zarr_group: The OME-NGFF pyramid structured ZARR array.
    :return: The maximum size of the OME-NGFF pyramid structured ZARR array.
    """

    max_size = [0] * len(zarr_group.attrs["multiscales"][0]["axes"])

    for dataset in zarr_group.attrs["multiscales"][0]["datasets"]:
        level_path = dataset["path"]

        arr = zarr_group[level_path]
        size = arr.shape

        max_size = [max(ms, s) for ms, s in zip(max_size, size)]

    return max_size


def _ngff_get_array_from_size(zarr_group: zarr.hierarchy.Group, target_size: List[int]) -> zarr.Array:
    """Returns the smallest array in the OME-NGFF pyramid structured ZARR array that is larger than the target size.

    :param zarr_group: The OME-NGFF pyramid structured ZARR array.
    :param target_size: The target size of the array to return.
    :return: The smallest array in the OME-NGFF pyramid structured ZARR array that is larger than the target size.
    """

    for dataset in reversed(zarr_group.attrs["multiscales"][0]["datasets"]):
        level_path = dataset["path"]

        arr = zarr_group[level_path]

        if any([s > t for s, t in zip(arr.shape, target_size) if t > 0]):
            return zarr_group[level_path]

    logger.warning(
        f"Could not find an array in the OME-NGFF pyramid structured ZARR array that is larger"
        f" than the target size: {target_size}"
    )
    return zarr_group[zarr_group["multiscales"][0]["datasets"]["path"]]


def zarr_extract_2d(
    input_zarr: Union[Path, str],
    target_size_x: int,
    target_size_y: int,
    *,
    size_factor: float = 1.5,
    output_filename: Union[Path, str, None] = None,
) -> Union[sitk.Image, None]:
    """Extracts a 2D image from an OME-NGFF pyramid structured with ZARR array that is 2D-like.

    The OME-NGFF pyramid structured ZARR array is assumed to have the following structure:
        - The axes spacial dimensions must be labeled as "X", "Y", and optionally "Z".
        - If a "Z" space dimension exists then it must be of size 1.
        - If a time dimension exists then it must be if of size 1.
        - If a channel dimension exists all channels are extracted.

    The extracted subvolume will be resized to the target size while maintaining the aspect ratio.

    The resized extracted subvolume with be the sample pixel type as the OME-NGFF pyramid structured ZARR array.

    If output_filename is not None then the extracted subvolume will be written to this file. The output ITK ImageIO
    used to write the file may place additional contains on the image which can be written. Such as JPEG only supporting
    uint8 pixel types and 1, 3, or 4 components.

    :param input_zarr: The path to an OME-NGFF structured ZARR array.
    :param target_size_x: The target size of the extracted subvolume in the x dimension.
    :param target_size_y: The target size of the extracted subvolume in the y dimension.
    :param size_factor: The size of the subvolume to extract will be increased by this factor so that the
        extracted subvolume can have antialiasing applied to it.
    :param output_filename: If not None then the extracted subvolume will be written to this file.
    :return: The extracted subvolume as a SimpleITK image.

    """

    input_zarr = Path(input_zarr)

    store = zarr.DirectoryStore(input_zarr)
    group = zarr.open_group(store=store, mode="r")
    logger.debug(group.info)

    if "multiscales" not in group.attrs:
        raise ValueError(f"Missing OME-NGFF multiscales meta data in zarr group: {input_zarr}")
    if len(group.attrs["multiscales"]) != 1:
        raise ValueError(f"Unexpected OME-NGFF multiscales meta data in zarr group: {input_zarr}")

    image_meta = group.attrs["multiscales"][0]
    axes = image_meta["axes"]

    max_size_per_dim = _ngff_get_max_size(group)

    zarr_source_size = [0] * len(axes)
    for d, ax in enumerate(axes):
        if ax["type"].lower() == "space":
            if ax["name"].lower() == "x":
                zarr_source_size[d] = target_size_x * size_factor
            elif ax["name"].lower() == "y":
                zarr_source_size[d] = target_size_y * size_factor
            elif ax["name"].lower() == "z":
                if max_size_per_dim[d] > 1:
                    raise ValueError(f"Z dimension has more than one element: {max_size_per_dim[d]}")
            else:
                raise ValueError(f"Unknown space axis: {ax['name']}")
        elif ax["type"].lower() == "time" and max_size_per_dim[d] > 1:
            raise ValueError(f"Time dimension has more than one element: {max_size_per_dim[d]}")

    logger.debug(f"zarr_source_size: {zarr_source_size}")

    arr = _ngff_get_array_from_size(group, zarr_source_size)
    logger.debug(arr.info)

    arr = da.from_zarr(arr.astype(arr.dtype.newbyteorder("=")))

    source_axes_name = _ngff_get_cononical_axes_names(group)

    target_axes_name = [n for n in "tzyxc" if n in source_axes_name]
    target_axes = [source_axes_name.index(n) for n in target_axes_name]
    logger.debug(f"source_axes: {source_axes_name} target_axes: {target_axes_name}")
    arr = da.transpose(arr, target_axes)

    squeeze_axes = "tz"
    for a in squeeze_axes:
        if a in target_axes_name:
            idx = target_axes_name.index(a)
            del target_axes_name[idx]
            arr = da.squeeze(arr, axis=idx)
    logger.debug(f"source_axes: {source_axes_name} target_axes: {target_axes_name}")

    img = sitk.GetImageFromArray(arr.compute(), isVector="c" in target_axes_name)

    logger.debug(img)

    logger.debug(f"resizing image of: {img.GetSize()} -> {(target_size_x, target_size_y)}")
    img = sitk.utilities.resize(img, (target_size_x, target_size_y), interpolator=sitk.sitkLinear, fill=False)

    if output_filename is not None:
        output_filename = Path(output_filename)
        logger.info(f"Writing image to: {output_filename}")
        sitk.WriteImage(img, str(output_filename))

    return img
