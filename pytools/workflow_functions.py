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
from typing import Dict, Union
import logging
import math
from SimpleITK.utilities.dask import from_sitk
from pytools.utils.histogram import DaskHistogramHelper, ZARRHistogramHelper, histogram_robust_stats, weighted_quantile
import SimpleITK as sitk
import dask.array


logger = logging.getLogger(__name__)


def visual_min_max(
    input_image: Union[Path, str],
    mad_scale: float,
    clamp: bool = True,
) -> Dict[str, int]:
    """Reads a path to an input_image file or a directory of zarr array to estimate minimum and maximum ranges to be
    used for visualization of the data set in Neuroglancer.

    Dask is used for parallel reading and statistics computation. The global scheduler is used for all operations which
    can be changed with standard Dask configurations.

    :param input_image: If an image file then SimpleITK is used to perform the IO. SimpleITK support formats such as
        mrc, mii, png, tiff etc.A zarr array is detected by a directory containing a ".zarray" file. For an OME-NGFF
        structured  ZARR a subdirectory such as "0" commonly contains the full resolution ZARR array. Such a case would
        be specified by "dirname.zarr/0".

    :param mad_scale: The scale factor for the robust median absolute deviation (MAD) about the median to produce the
        "minimum and maximum range."

    :param clamp: If True then the minimum and maximum range will be clamped to the computed floor and limit values.


    :returns: The resulting dictionary will contain the following data elements with integer values as strings:
       - "neuroglancerPrecomputedMin"
       - "neuroglancerPrecomputedMax"
       - "neuroglancerPrecomputedFloor"
       - "neuroglancerPrecomputedLimit"
    """

    input_image = Path(input_image)

    if input_image.is_dir() and (input_image / ".zarray").exists():
        histo = ZARRHistogramHelper(input_image)
    elif input_image.suffix in (".nii", ".mha", ".mrc", ".rec"):
        logger.info("Loading chunk with SimpleITK and dask...")
        sitk_da = from_sitk(input_image, chunks=(1, -1, -1))
        histo = DaskHistogramHelper(sitk_da)
    else:
        logger.info("Loading whole image with SimpleITK...")
        img = sitk.ReadImage(input_image)
        histo = DaskHistogramHelper(dask.array.from_array(sitk.GetArrayViewFromImage(img), chunks=(1, -1, -1)))

    logger.debug(f"dask.config.global_config: {dask.config.global_config}")

    logger.info(f'Building histogram for "{input_image}"...')
    h, bins = histo.compute_histogram(histogram_bin_edges=None, density=False)

    mids = 0.5 * (bins[1:] + bins[:-1])

    logger.info("Computing statistics...")
    stats = histogram_robust_stats(h, bins)
    logger.debug(f"stats: {stats}")
    min_max = (stats["median"] - stats["mad"] * mad_scale, stats["median"] + stats["mad"] * mad_scale)

    floor_limit = weighted_quantile(mids, quantiles=[0.0, 1.0], sample_weight=h, values_sorted=True)

    if clamp:
        logger.debug(f"clamping min_max: {min_max} to floor_limit: {floor_limit}")
        min_max = (max(min_max[0], floor_limit[0]), min(min_max[1], floor_limit[1]))

    output = {
        "neuroglancerPrecomputedMin": str(math.floor(min_max[0])),
        "neuroglancerPrecomputedMax": str(math.ceil(min_max[1])),
        "neuroglancerPrecomputedFloor": str(math.floor(floor_limit[0])),
        "neuroglancerPrecomputedLimit": str(math.ceil(floor_limit[1])),
    }

    return output
