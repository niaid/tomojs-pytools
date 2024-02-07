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

from pathlib import Path

import SimpleITK as sitk
import zarr
from typing import Tuple, Dict, List, Optional
from pytools.utils import OMEInfo
import logging
import math
import re
import dask
from pytools.utils.histogram import DaskHistogramHelper, histogram_robust_stats, histogram_stats, weighted_quantile

logger = logging.getLogger(__name__)


class HedwigZarrImage:
    """
    Represents a OME-NGFF Zarr pyramidal image. The members provide information useful for the Hedwig imaging pipelines.
    """

    def __init__(self, zarr_grp: zarr.Group, _ome_info: OMEInfo, _ome_idx: int):
        self.zarr_group = zarr_grp
        self.ome_info = _ome_info
        self.ome_idx = _ome_idx

        assert "multiscales" in self.zarr_group.attrs

    @property
    def path(self) -> Path:
        """
        Returns full path to the ZARR group suitable for Neuroglancer.
        """
        return Path(self.zarr_group.store.path) / self.zarr_group.path

    @property
    def dims(self) -> str:
        """
        The Hedwig dimension of the image XY, XYC, XYZCT etc.

        Collapses ZCT dimensions if of size 1.

        Note: this is the reverse order of axis for numpy/zarr/dask.
        """
        dims = [d for s, d in zip(self.shape, self._ome_ngff_multiscale_dims()) if d in "XY" or s > 1]
        return "".join(dims[::-1])

    @property
    def shape(self) -> Tuple[int]:
        """The size of the dimensions of the full resolution image.

        This is in numpy/zarr/dask order.
        """
        return self._ome_ngff_multiscale_get_array(0).shape

    @property
    def spacing(self) -> Tuple[float]:
        """The size of the dimensions of the full resolution image.

        This is in numpy/zarr/dask order.
        """

        return self._ome_ngff_multiscales(idx=0)["datasets"][0]["coordinateTransformations"][0]["scale"]

    def rechunk(self, chunk_size: int, compressor=None, *, in_memory=False) -> None:
        """
        Change the chunk size of each ZARR array inplace in the pyramid.

        The chunk_size is applied to all spacial dimension, and other dimension (CT) are the full size.

        The ImageZarrImage need write access to the ZARR.

        :param chunk_size: The size as an integer to resize the chunk sizes.
        :param compressor: The output arrays will be written with the provided compressor, if None then the compressor
         of the input arrays will be used.
        :param in_memory: If true the entire arrays will be loaded into memory uncompressed, before writing to the
        rechunked size, otherwise the arrays will be written directly to the rechunked size. The former is faster but
        requires enough memory to hold the arrays.
        """

        logger.info(f'Processing group: "{self.zarr_group.name}"...')
        logger.debug(self.zarr_group)

        # grok through the OME-NGFF meta-dat, for each image scale (dataset/array) with axes information
        # https://ngff.openmicroscopy.org/latest/#multiscale-md
        image = self._ome_ngff_multiscales()

        chunk_request = tuple(chunk_size if a["type"] == "space" else -1 for a in image["axes"])

        for dataset in image["datasets"]:
            arr = self.zarr_group[dataset["path"]]
            arr_name = arr.name
            logger.info(f'Processing array: "{arr.name}"...')
            logger.debug(arr.info)

            chunks = tuple(self._chunk_logic_dim(r, s) for r, s in zip(chunk_request, arr.shape))
            if arr.chunks == chunks:
                logger.info("Chunks already requested size")
                continue

            temp_arr = arr
            if in_memory:
                logger.info(f'Loading array: "{arr.name}" into memory...')
                # optionally load the entire array uncompressed into memory
                memory_group = zarr.group(store=zarr.MemoryStore(), overwrite=True)
                zarr.copy(temp_arr, memory_group, name="temp", compressor=None)
                temp_arr = memory_group["temp"]

                logger.info(f'Rechunking array: "{arr.name} to disk"...')

            # copy array to a temp zarr array on file
            zarr.copy(
                temp_arr,
                self.zarr_group,
                name=arr_name + ".temp",
                chunks=chunks,
                compressor=arr.compressor if compressor is None else compressor,
                dimension_separator=arr._dimension_separator,
                filters=arr.filters,
                overwrite=False,
            )

            logger.debug(self.zarr_group[dataset["path"] + ".temp"].info)
            logger.debug(f"replace: {self.zarr_group[dataset['path'] + '.temp'].name} -> {arr_name}")
            del self.zarr_group[dataset["path"]]
            self.zarr_group.store.rename(self.zarr_group[dataset["path"] + ".temp"].name, arr_name)

    def extract_2d(
        self, target_size_x: int, target_size_y: int, *, size_factor: float = 1.5, auto_uint8: bool = False
    ) -> sitk.Image:
        """Extracts a 2D SimpleITK Image from an OME-NGFF pyramid structured with ZARR array that is 2D-like.

        The OME-NGFF pyramid structured ZARR array is assumed to have the following structure:
            - The axes spacial dimensions must be labeled as "X", "Y", and optionally "Z".
            - If a "Z" space dimension exists then it must be of size 1.
            - If a time dimension exists then it must be if of size 1.
            - If a channel dimension exists all channels are extracted.

        The extracted subvolume will be resized to the target size while maintaining the aspect ratio.

        The resized extracted subvolume with be the same pixel type as the OME-NGFF pyramid structured ZARR array.

        :param target_size_x: The target size of the extracted subvolume in the x dimension.
        :param target_size_y: The target size of the extracted subvolume in the y dimension.
        :param size_factor: The size of the subvolume to extract will be increased by this factor so that the
            extracted subvolume can have antialiasing applied to it.
        :param auto_uint8: If True the output image will be automatically linearly shifted and scaled to fit into uint8
            component pixel types.
        :return: The extracted subvolume as a SimpleITK image.

        """

        source_axes_names = self._ome_ngff_multiscale_dims()
        assert source_axes_names == "TCZYX"
        if self.shape[0] > 1:
            raise ValueError(f"Time dimension has more than one element: {self.shape[0]}")
        is_vector = self.shape[1] > 1
        if self.shape[2] > 1:
            raise ValueError(f"Z dimension has more than one element: {self.shape[2]}")

        target_axes_names = [n for n in "tzyxc" if n in source_axes_names]
        logger.debug(f"source_axes: {source_axes_names} target_axes: {target_axes_names}")

        zarr_request_size = [1, 0, 1, target_size_y * size_factor, target_size_x * size_factor]

        z_arr, spacing_tczyx = self._ome_ngff_get_array_from_size(zarr_request_size)

        logger.debug(z_arr.info)

        d_arr = dask.array.from_zarr(z_arr)
        d_arr = d_arr.astype(d_arr.dtype.newbyteorder("="))
        if is_vector:
            d_arr = dask.array.squeeze(d_arr, axis=(0, 2))  # Output CYX
            d_arr = dask.array.moveaxis(d_arr, 0, -1)  # transpose to YXC
        else:
            d_arr = dask.array.squeeze(d_arr, axis=(0, 1, 2))

        img = sitk.GetImageFromArray(d_arr.compute(), isVector=is_vector)
        img.SetSpacing((spacing_tczyx[4], spacing_tczyx[3]))

        logger.debug(img)

        logger.debug(f"resizing image of: {img.GetSize()} -> {(target_size_x, target_size_y)}")
        img = sitk.utilities.resize(img, (target_size_x, target_size_y), interpolator=sitk.sitkLinear, fill=False)

        if auto_uint8:
            if is_vector:
                img.ToScalarImage(True)

            min, max = sitk.MinimumMaximum(img)

            logger.debug(f"Adjusting output pixel intensity range from {min, max} -> {(0, 255)}.")

            img = sitk.ShiftScale(img, -min, 255.0 / (max - min), sitk.sitkUInt8)

            if is_vector:
                img.ToVectorImage(True)

        return img

    @property
    def shader_type(
        self,
    ) -> str:
        """
        Produces the shader type one of: RGB, Grayscale, or MultiChannel.
        """
        if self.ome_info and self.ome_info.maybe_rgb(self.ome_idx):
            return "RGB"
        if self._ome_ngff_multiscale_dims()[1] == "C" and self.shape[1] == 1:
            return "Grayscale"
        return "MultiChannel"

    def _neuroglancer_shader_parameters_multichannel(
        self,
        *,
        mad_scale=3,
        middle_quantile: Optional[Tuple[float, float]] = None,
        zero_black_quantiles=True,
        upper_quantile=0.9999,
    ) -> dict:
        """
        Produces the "shaderParameters" portion of the metadata for Neuroglancer when the shader type is MultiChannel.

        The output window parameters are used for the visible histogram range. The computation improves the robustness
         to outliers and the background pixel values with the zero_black_quantiles option and the upper_quantile value.

        :param mad_scale: The scale factor for the robust median absolute deviation (MAD) about the median to produce
            the minimum and maximum range that is used to select the visible pixel intensities.
        :param middle_quantile: If not None then the range is computed from the  provided quantiles of the image data.
            The middle_quantile is a tuple of two floats that are between 0.0 and 1.0. The first value is the lower
             quantile and the second value is the upper quantile.
        :param zero_black_quantiles: If True then the zero values are removed from the histogram before computing the
            quantiles.
        :param upper_quantile: The upper quantile to use for the "window", which is the extent of the visible histogram.
        :return: The dictionary of the shader parameters suitable for JSON serialization.

        """

        if middle_quantile:
            assert len(middle_quantile) == 2
            assert 0.0 <= middle_quantile[0] <= 1.0
            assert 0.0 <= middle_quantile[1] <= 1.0
            assert middle_quantile[0] < middle_quantile[1]

        assert self._ome_ngff_multiscale_dims()[1] == "C"

        if len(list(self.ome_info.channel_names(self.ome_idx))) != self.shape[1]:
            raise RuntimeError(
                f"Mismatch of number of Channels! Array has {self.shape[1]} but there"
                f"are {len(list(self.ome_info.channel_names(self.ome_idx)))} names:"
                f"{list(self.ome_info.channel_names(self.ome_idx))}"
            )

        color_sequence = ["red", "green", "blue", "cyan", "yellow", "magenta"]

        if self.shape[1] > len(color_sequence):
            raise RuntimeError(
                f"Too many channels! The array has {self.shape[1]} channels but"
                f" only {len(color_sequence)} is supported!"
            )

        json_channel_array = []

        for c, c_name in enumerate(self.ome_info.channel_names(self.ome_idx)):
            logger.debug(f"Processing channel: {c_name}")

            # replace non-alpha numeric with an underscore
            name = re.sub(r"[^a-zA-Z0-9]+", "_", c_name.lower())

            stats = self._image_statistics(
                quantiles=[*middle_quantile, upper_quantile] if middle_quantile else [upper_quantile],
                channel=c,
                zero_black_quantiles=zero_black_quantiles,
            )
            if middle_quantile:
                range = (stats["quantiles"][middle_quantile[0]], stats["quantiles"][middle_quantile[1]])
            else:
                range = (stats["median"] - stats["mad"] * mad_scale, stats["median"] + stats["mad"] * mad_scale)

            range = (max(range[0], stats["min"]), min(range[1], stats["max"]))

            json_channel_array.append(
                {
                    "range": [math.floor(range[0]), math.ceil(range[1])],
                    "window": [math.floor(stats["min"]), math.ceil(stats["quantiles"][upper_quantile])],
                    "name": name,
                    "color": color_sequence[c],
                    "channel": c,
                    "clamp": False,
                    "enabled": True,
                }
            )

        return {"brightness": 0.0, "contrast": 0.0, "channelArray": json_channel_array}

    def neuroglancer_shader_parameters(
        self, *, mad_scale=3, middle_quantile: Optional[Tuple[float, float]] = None
    ) -> dict:
        """
        Produces the "shaderParameters" portion of the metadata for Neuroglancer.

        Determines which shader type to use to render the image. The shader type is one of: RGB, Grayscale, or
        MultiChannel. The shader parameters are computed from the full resolution Zarr image. Dask is used for parallel
        reading and statistics computation. The global scheduler is used for all operations which can be changed with
        standard Dask configurations.

        For the MultiChannel shader type the default algorithm for the range is to compute the robust median absolute
        deviation (MAD) about the median to produce the minimum and maximum range. If the middle_quantile is not None
        then the range is computed from the provided quantiles of the image data.

        :param mad_scale: The scale factor for the robust median absolute deviation (MAD) about the median to produce
            the minimum and maximum range that is used to select the visible pixel intensities.
        :param middle_quantile: If not None then the range is computed from the  provided quantiles of the image data.
         The middle_quantile is a tuple of two floats that are between 0.0 and 1.0. The first value is the lower
         quantile and the second value is the upper quantile. The range is computed from the lower and upper quantiles.
        :return: The dictionary of the shader parameters suitable for JSON serialization
        """

        if self.ome_info is None:
            return {}

        _shader_type = self.shader_type
        if _shader_type == "RGB":
            return {}
        if _shader_type == "Grayscale":
            stats = self._image_statistics(channel=None)
            range = (stats["median"] - stats["mad"] * mad_scale, stats["median"] + stats["mad"] * mad_scale)
            range = (max(range[0], stats["min"]), min(range[1], stats["max"]))
            return {
                "range": [math.floor(range[0]), math.ceil(range[1])],
                "window": [math.floor(stats["min"]), math.ceil(stats["max"])],
            }

        if _shader_type == "MultiChannel":
            return self._neuroglancer_shader_parameters_multichannel(
                mad_scale=mad_scale, middle_quantile=middle_quantile
            )

        raise RuntimeError(f'Unknown shader type: "{self.shader_type}"')

    def _ome_ngff_multiscales(self, idx=0):
        """Get OME NGFF multiscale metadata"""
        return self.zarr_group.attrs["multiscales"][idx]

    def _ome_ngff_multiscale_get_array(self, level, idx=0) -> zarr.Array:
        """Get array at multiscale level"""
        return self.zarr_group[self._ome_ngff_multiscales(idx)["datasets"][level]["path"]]

    def _ome_ngff_multiscale_dims(self):
        """ZARR order dimension name in uppercase

        Expected to be "TCZYX".
        """
        dims = ""
        for ax in self._ome_ngff_multiscales()["axes"]:
            dims += ax["name"].upper()
        return dims

    def _ome_ngff_get_array_from_size(self, target_size: List[int]) -> Tuple[zarr.Array, List[float]]:
        """Returns the smallest array in the OME-NGFF pyramid structured ZARR array that is larger than the target sizes
        in all dimensions.

        :param target_size: The target size of the array to return.
        :return: The smallest array in the OME-NGFF pyramid structured ZARR array that is larger than the target size.
        """

        for dataset in reversed(self._ome_ngff_multiscales(idx=0)["datasets"]):
            spacing = dataset["coordinateTransformations"][0]["scale"]
            level_path = dataset["path"]

            arr = self.zarr_group[level_path]

            if any([s > t for s, t in zip(arr.shape, target_size) if t > 0]):
                return self.zarr_group[level_path], spacing

        logger.warning(
            f"Could not find an array in the OME-NGFF pyramid structured ZARR array that is larger"
            f" than the target size: {target_size}"
        )
        return self.zarr_group[level_path], spacing

    @staticmethod
    def _chunk_logic_dim(drequest: int, dshape: int) -> int:
        if dshape > drequest > 0:
            return drequest
        return dshape

    def _image_statistics(self, quantiles=None, channel=None, *, zero_black_quantiles=False) -> Dict[str, List[int]]:
        """Processes the full resolution Zarr image. Dask is used for parallel reading and statistics computation. The
         global scheduler is used for all operations which can be changed with standard Dask configurations.

        :param channel: The index of the channel to perform calculation on
        :param quantiles: values of quantiles to return in option "quantiles" element.

        :returns: The resulting dictionary will contain the following data elements:
            "min",
            "max",
            "median",
            "mad",
            "mean",
            "var",
            "sigma",
            "quantiles" (if quantiles is not None)

        """

        logger.debug(f"path: {self._ome_ngff_multiscale_get_array(0).path}")

        # extract channel
        assert self._ome_ngff_multiscale_dims()[1] == "C"
        if channel is not None:
            logger.info(f"Extracting channel {channel}..")
            da = dask.array.from_zarr(self._ome_ngff_multiscale_get_array(0))
            da = da[:, channel, ...]
        else:
            da = dask.array.from_zarr(self._ome_ngff_multiscale_get_array(0))

        histo = DaskHistogramHelper(da)

        logger.debug(f"dask.config.global_config: {dask.config.global_config}")

        logger.info(f'Building histogram for "{self.path}"...')
        h, bins = histo.compute_histogram(histogram_bin_edges=None, density=False)

        mids = 0.5 * (bins[1:] + bins[:-1])

        logger.info("Computing statistics...")
        stats = histogram_robust_stats(h, bins)
        stats.update(histogram_stats(h, bins))
        stats["min"], stats["max"] = weighted_quantile(mids, quantiles=[0.0, 1.0], sample_weight=h, values_sorted=True)
        if quantiles:
            if zero_black_quantiles:
                h[0] = 0

            quantile_value = weighted_quantile(mids, quantiles=quantiles, sample_weight=h, values_sorted=True)
            stats["quantiles"] = {q: v for q, v in zip(quantiles, quantile_value)}
        logger.debug(f"stats: {stats}")

        return stats
