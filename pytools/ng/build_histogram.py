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
from pytools.utils import MutuallyExclusiveOption
from pytools import __version__
from math import floor, ceil


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


def stream_build_histogram(filename: str, histogram_bin_edges, extract_axis=1, density=False):
    """
    Read image slice by slice, and build a histogram. The image file must be readable by SimpleITK.
    The SimpleITK is expected to support streaming the file format.

    The np.histogram function is run on each image slice with the provided histogram_bin_edges, and
    accumulated for the results.

    :param  filename: The path to the image file to read. MRC file type is recommend.
    :param histogram_bin_edges: A monotonically increasing array of min edges. The resulting
      histogram or weights will have n-1 elements.
    :param extract_axis: The image dimension which is sliced during image reading.
    :param density: If true the sum of the results is 1.0, otherwise it is the count of values in each bin.
    """
    reader = sitk.ImageFileReader()
    reader.SetFileName(filename)
    reader.ReadImageInformation()

    extract_index = [0] * reader.GetDimension()

    extract_size = list(reader.GetSize())
    extract_size[extract_axis] = 0
    reader.SetExtractSize(extract_size)

    h = np.zeros(len(histogram_bin_edges) - 1, dtype=np.int64)

    for i in range(reader.GetSize()[extract_axis]):
        extract_index[extract_axis] = i
        reader.SetExtractIndex(extract_index)
        img = reader.Execute()

        # accumulate histogram counts
        h += np.histogram(sitk.GetArrayViewFromImage(img).flatten(), bins=histogram_bin_edges, density=False)[0]

    if density:
        h /= np.sum(h)

    return h, np.array(histogram_bin_edges)


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
@click.argument("input_image", type=click.Path(exists=True, dir_okay=False, resolve_path=True))
@click.option(
    "--mad",
    type=float,
    cls=MutuallyExclusiveOption,
    mutually_exclusive=["sigma", "percentile-crop"],
    help="Use INPUT_IMAGE's robust median absolute deviation (MAD) scale by option's value about the median for "
    "minimum and maximum range. ",
)
@click.option(
    "--sigma",
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
@click.version_option(__version__)
def main(input_image, mad, sigma, percentile, clamp, output_json):
    """
    Reads the INPUT_IMAGE to compute an estimated minimum and maximum range to be used for visualization of the
    data set. The image is required to have an integer pixel type.

    The optional OUTPUT_JSON filename will be created with the following data elements with integer values as strings:
        "neuroglancerPrecomputedMin"
        "neuroglancerPrecomputedMax"
        "neuroglancerPrecomputedFloor"
        "neuroglancerPrecomputedLimit"
    """

    logger = logging.getLogger()
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.DEBUG)

    reader = sitk.ImageFileReader()
    reader.SetFileName(input_image)

    logger.info(f'Reading "{reader.GetFileName()}" image information...')
    reader.ReadImageInformation()

    logger.info(f"\tPixel Type: {sitk.GetPixelIDValueAsString(reader.GetPixelIDValue())}")
    logger.info(f"\tSize: {reader.GetSize()}")
    logger.info(f"\tSpacing: {reader.GetSpacing()}")
    logger.info(f"\tOrigin:  {reader.GetOrigin()}")

    pixel_type = reader.GetPixelID()

    sitk_to_np = {
        sitk.sitkUInt8: np.uint8,
        sitk.sitkUInt16: np.uint16,
        sitk.sitkUInt32: np.uint32,
        sitk.sitkUInt64: np.uint64,
        sitk.sitkInt8: np.int8,
        sitk.sitkInt16: np.int16,
    }

    img_dtype = sitk_to_np[pixel_type]

    bin_edges = np.arange(np.iinfo(img_dtype).min - 0.5, np.iinfo(img_dtype).max + 1.5)

    logger.info(f'Building histogram for "{reader.GetFileName()}"...')
    h, bins = stream_build_histogram(input_image, list(bin_edges))
    mids = 0.5 * (bins[1:] + bins[:-1])

    logger.info("Computing statistics...")
    if mad:
        stats = histogram_robust_stats(h, bins)
        logger.debug(f"stats: {stats}")
        min_max = (stats["median"] - stats["mad"] * mad, stats["median"] + stats["mad"] * mad)
    elif sigma:
        stats = histogram_stats(h, bins)
        logger.debug(f"stats: {stats}")
        min_max = (stats["mean"] - stats["sigma"] * sigma, stats["mean"] + stats["sigma"] * sigma)
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
