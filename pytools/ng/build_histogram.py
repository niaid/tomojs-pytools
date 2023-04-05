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
import math
import click
import numpy as np
import logging
import SimpleITK as sitk
import zarr

from pytools.utils import MutuallyExclusiveOption
from pytools import __version__
from math import floor, ceil
from pathlib import Path
import dask.array
from typing import Union, Tuple, Any
from abc import ABC, abstractmethod


logger = logging.getLogger(__name__)


def weighted_quantile(values, quantiles, sample_weight=None, values_sorted=False, old_style=False):
    """Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
    initial array
    :param old_style: if True, will correct output to be consistent
    with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)

    # Remove zero weighted samples
    non_zero_mask = sample_weight != 0
    sample_weight = sample_weight[non_zero_mask]
    values = values[non_zero_mask]

    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), "quantiles should be in [0, 1]"

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)

    return np.interp(quantiles, weighted_quantiles, values)


class HistogramBase(ABC):
    @property
    @abstractmethod
    def dtype(self):
        pass

    @abstractmethod
    def compute_min_max(self) -> Tuple[Any, Any]:
        pass

    def compute_histogram_bin_edges(self, number_of_bins=1024) -> np.ndarray:
        if np.issubdtype(self.dtype, np.integer) and 2 ** np.iinfo(self.dtype).bits < number_of_bins:
            return np.arange(np.iinfo(self.dtype).min - 0.5, np.iinfo(self.dtype).max + 1.5)

        imin, imax = self.compute_min_max()

        if np.issubdtype(self.dtype, np.inexact):
            if imax - imin < np.finfo(imin).eps * number_of_bins:
                logger.warning("Computed difference between minimum and maximum is below tolerances.")
                imax = imin + np.finfo(imin).eps * number_of_bins / 2
                imin = imin - np.finfo(imin).eps * number_of_bins / 2
        elif np.issubdtype(self.dtype, np.integer) and imax - imin < number_of_bins:
            return np.arange(imin - 0.5, imax + 1.5)

        step = (imax - imin) / number_of_bins

        logger.debug(f"Computed minimum: {imin} maximum: {imax} step: {step}")
        histogram_bin_edges = np.arange(imin, imax + step, step)
        return histogram_bin_edges


class sitkHistogramHelper(HistogramBase):
    """
    Read image slice by slice, and build a histogram. The image file must be readable by SimpleITK.
    The SimpleITK is expected to support streaming the file format.

    The np.histogram function is run on each image slice with the provided histogram_bin_edges, and
    accumulated for the results.

    :param  filename: The path to the image file to read. MRC file type is recommend.
    :param histogram_bin_edges: A monotonically increasing array of min edges. The resulting
      histogram or weights will have n-1 elements. If None, then it will be automatically computed for integers, and
      an np.bincount may be used as an optimization.
    :param extract_axis: The image dimension which is sliced during image reading.
    :param density: If true the sum of the results is 1.0, otherwise it is the count of values in each bin.
    :param extract_step: The number of slices to read at one time.
    """

    def __init__(self, filename, extract_axis=2, extract_step=1):
        self.reader = sitk.ImageFileReader()
        self.reader.SetFileName(str(filename))
        self.reader.ReadImageInformation()

        logger.info(f'Reading "{self.reader.GetFileName()}" image information...')

        logger.info(f"\tPixel Type: {sitk.GetPixelIDValueAsString(self.reader.GetPixelIDValue())}")
        logger.info(f"\tPixel Type: {sitk.GetPixelIDValueAsString(self.reader.GetPixelIDValue())}")
        logger.info(f"\tSize: {self.reader.GetSize()}")
        logger.info(f"\tSpacing: {self.reader.GetSpacing()}")
        logger.info(f"\tOrigin:  {self.reader.GetOrigin()}")

        self.extract_axis = extract_axis
        self.extract_step = extract_step

    def compute_min_max(self):
        img = self.reader.Execute()

        min_max_filter = sitk.MinimumMaximumImageFilter()
        min_max_filter.Execute(img)
        return min_max_filter.GetMinimum(), min_max_filter.GetMaximum()

    @property
    def dtype(self):
        return sitk.extra._get_numpy_dtype(self.reader)

    def compute_histogram(self, histogram_bin_edges=None, density=False):
        use_bincount = False
        if histogram_bin_edges is None:
            if np.issubdtype(self.dtype, np.integer) and np.iinfo(self.dtype).bits <= 16:
                histogram_bin_edges = self.compute_histogram_bin_edges(
                    number_of_bins=2 ** np.iinfo(self.dtype).bits + 1
                )
                if self.dtype() in (np.uint8, np.uint16):
                    use_bincount = True
            else:
                histogram_bin_edges = self.compute_histogram_bin_edges()

        h = np.zeros(len(histogram_bin_edges) - 1, dtype=np.int64)

        extract_index = [0] * self.reader.GetDimension()

        size = self.reader.GetSize()
        extract_size = list(size)
        extract_size[self.extract_axis] = 0
        self.reader.SetExtractSize(extract_size)

        for i in range(0, self.reader.GetSize()[self.extract_axis], self.extract_step):
            extract_index[self.extract_axis] = i
            self.reader.SetExtractIndex(extract_index)
            logger.debug(f"extract_index: {extract_index}")

            extract_size[self.extract_axis] = min(i + self.extract_step, size[self.extract_axis]) - i
            self.reader.SetExtractSize(extract_size)
            img = self.reader.Execute()

            # accumulate histogram counts
            if use_bincount:
                h += np.bincount(sitk.GetArrayViewFromImage(img).ravel(), minlength=len(h))
            else:
                h += np.histogram(sitk.GetArrayViewFromImage(img).ravel(), bins=histogram_bin_edges, density=False)[0]

        if density:
            h /= np.sum(h)

        return h, histogram_bin_edges


class zarrHisogramHelper(HistogramBase):
    def __init__(self, filename):
        za = zarr.open_array(filename, mode="r")
        self._arr = dask.array.from_zarr(za)
        logging.debug(za.info)
        if not self._arr.dtype.isnative:
            logging.info("ZARR array needs converting to native byteorder.")
            self._arr = self._arr.astype(self._arr.dtype.newbyteorder("="))

    def compute_min_max(self):
        return self._arr.min(), self._arr.max()

    @property
    def dtype(self):
        return self._arr.dtype

    def compute_histogram(self, histogram_bin_edges=None, density=False):
        if histogram_bin_edges is None:
            if np.issubdtype(self.dtype, np.integer) and np.iinfo(self.dtype).bits <= 16:
                histogram_bin_edges = self.compute_histogram_bin_edges(
                    number_of_bins=2 ** np.iinfo(self.dtype).bits + 1
                )

                if np.dtype(self.dtype) in (np.uint8, np.uint16):
                    return (
                        dask.array.bincount(self._arr.ravel(), minlength=len(histogram_bin_edges) - 1).compute(),
                        histogram_bin_edges,
                    )

            else:
                histogram_bin_edges = self.compute_histogram_bin_edges()

        h, bins = dask.array.histogram(self._arr.ravel(), bins=histogram_bin_edges, density=density)
        return h.compute(), bins


def stream_build_histogram(
    filename: Union[Path, str], histogram_bin_edges=None, extract_axis=2, density=False, extract_step=1
):
    """
    Read image slice by slice, and build a histogram. The image file must be readable by SimpleITK.
    The SimpleITK is expected to support streaming the file format.
    The np.histogram function is run on each image slice with the provided histogram_bin_edges, and
    accumulated for the results.
    :param  filename: The path to the image file to read. MRC file type is recommend.
    :param histogram_bin_edges: A monotonically increasing array of min edges. The resulting
      histogram or weights will have n-1 elements. If None, then it will be automatically computed for integers, and
      an np.bincount may be used as an optimization.
    :param extract_axis: The image dimension which is sliced during image reading.
    :param density: If true the sum of the results is 1.0, otherwise it is the count of values in each bin.
    :param extract_step: The number of slices to read at one time.
    """

    input_image = Path(filename)

    if input_image.is_dir() and (input_image / ".zarray").exists():
        histo = zarrHisogramHelper(input_image)

    else:
        histo = sitkHistogramHelper(filename, extract_axis=extract_axis, extract_step=extract_step)

    return histo.compute_histogram(histogram_bin_edges=histogram_bin_edges, density=density)


def histogram_robust_stats(hist, bin_edges):
    """
    Computes the "median" and "mad" (Median Absolute Deviation).

    :param hist: The histogram weights ( density or count ).
    :param bin_edges: The edges of the bins. This array should be one greater than the hist.
    """
    assert len(hist) + 1 == len(bin_edges)

    mids = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    median = weighted_quantile(mids, values_sorted=True, sample_weight=hist, quantiles=0.5)

    mad = weighted_quantile(np.abs(mids - median), values_sorted=False, sample_weight=hist, quantiles=0.5)

    return {"median": median, "mad": mad}


def histogram_stats(hist, bin_edges):
    """
    Computes the "mean", "var" (variance), and "sigma" (standard deviation) from the provided histogram.

    :param hist: The histogram weights ( density or count ).
    :param bin_edges: The edges of the bins. This array should be one greater than the hist.
    """

    assert len(hist) + 1 == len(bin_edges)

    results = {}

    mids = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    results["mean"] = np.average(mids, weights=hist)
    results["var"] = np.average((mids - results["mean"]) ** 2, weights=hist)
    results["sigma"] = math.sqrt(results["var"])

    return results


@click.command()
@click.argument("input_image", type=click.Path(exists=True, dir_okay=True, path_type=Path))
@click.option(
    "--mad",
    "mad_scale",
    type=float,
    cls=MutuallyExclusiveOption,
    mutually_exclusive=["sigma", "percentile-crop"],
    help="Use INPUT_IMAGE's robust median absolute deviation (MAD) scale by option's value about the median for "
    "minimum and maximum range. ",
)
@click.option(
    "--sigma",
    "sigma_scale",
    type=float,
    cls=MutuallyExclusiveOption,
    mutually_exclusive=["mad", "percentile-crop"],
    help="Use INPUT_IMAGE's standard deviation (sigma) scale by option's value about the mean for minimum and "
    "maximum range. ",
)
@click.option(
    "--percentile",
    type=click.FloatRange(0.0, 100),
    cls=MutuallyExclusiveOption,
    mutually_exclusive=["sigma", "mad"],
    help="Use INPUT_IMAGE's middle percentile (option's value) of data for minimum and maximum range.",
)
@click.option(
    "--clamp/--no-clamp",
    default=False,
    help="Clamps minimum and maximum range to existing intensity values (floor and limit).",
)
@click.option(
    "--output-json",
    type=click.Path(exists=False, dir_okay=False, resolve_path=True),
    help='The output filename produced in JSON format with "neuroglancerPrecomputedMin", '
    '"neuroglancerPrecomputedMax", "neuroglancerPrecomputedFloor" and "neuroglancerPrecomputedLimit" data '
    "elements of a double numeric value.",
)
@click.option(
    "--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False)
)
@click.version_option(__version__)
def main(input_image: Path, mad_scale, sigma_scale, percentile, clamp, output_json, log_level):
    """
    Reads the INPUT_IMAGE to compute an estimated minimum and maximum range to be used for visualization of the
    data set. The image is required to have an integer pixel type.

    The optional OUTPUT_JSON filename will be created with the following data elements with integer values as strings:
        "neuroglancerPrecomputedMin"
        "neuroglancerPrecomputedMax"
        "neuroglancerPrecomputedFloor"
        "neuroglancerPrecomputedLimit"
    """

    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.getLevelName(log_level))

    if input_image.is_dir() and (input_image / ".zarray").exists():
        histo = zarrHisogramHelper(input_image)

    else:
        histo = sitkHistogramHelper(input_image, extract_axis=2, extract_step=1)

    logger.info(f'Building histogram for "{input_image}"...')
    h, bins = histo.compute_histogram(histogram_bin_edges=None, density=False)

    mids = 0.5 * (bins[1:] + bins[:-1])

    logger.info("Computing statistics...")
    if mad_scale:
        stats = histogram_robust_stats(h, bins)
        logger.debug(f"stats: {stats}")
        min_max = (stats["median"] - stats["mad"] * mad_scale, stats["median"] + stats["mad"] * mad_scale)
    elif sigma_scale:
        stats = histogram_stats(h, bins)
        logger.debug(f"stats: {stats}")
        min_max = (stats["mean"] - stats["sigma"] * sigma_scale, stats["mean"] + stats["sigma"] * sigma_scale)
    elif percentile:
        lower_quantile = (0.5 * (100 - percentile)) / 100.0
        upper_quantile = percentile / 100.0 + lower_quantile
        logger.debug(f"quantiles: {lower_quantile} {upper_quantile}")

        # cs = np.cumsum(h)
        # min_max = (np.searchsorted(cs, cs[-1] * (percentile_crop * .005)),
        #           np.searchsorted(cs, cs[-1] * (1.0 - percentile_crop * .005)))
        # min_max = (mids[min_max[0]], mids[min_max[1]])
        min_max = weighted_quantile(
            mids, quantiles=[lower_quantile, upper_quantile], sample_weight=h, values_sorted=True
        )
    else:
        raise RuntimeError("Missing expected argument")

    floor_limit = weighted_quantile(mids, quantiles=[0.0, 1.0], sample_weight=h, values_sorted=True)

    if clamp:
        min_max = (max(min_max[0], floor_limit[0]), min(min_max[1], floor_limit[1]))

    output = {
        "neuroglancerPrecomputedMin": str(floor(min_max[0])),
        "neuroglancerPrecomputedMax": str(ceil(min_max[1])),
        "neuroglancerPrecomputedFloor": str(floor(floor_limit[0])),
        "neuroglancerPrecomputedLimit": str(ceil(floor_limit[1])),
    }

    logger.debug(f"output: {output}")
    if output_json:
        import json

        with open(output_json, "w") as fp:
            json.dump(output, fp)
    else:
        print(output)

    return output


if __name__ == "__main__":
    main()
