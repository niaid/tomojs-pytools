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
import pytools.utils.histogram as pytools_hist
import pytest
from pytest import approx
import SimpleITK as sitk
import numpy as np
from pathlib import Path

import pytools.workflow_functions


def test_weighted_quantile():
    data = [2]

    h, b = np.histogram(data, bins=np.arange(-0.5, 10.5))
    v = 0.5 * (b[1:] + b[:-1])

    assert pytools_hist.weighted_quantile(v, [0], sample_weight=h) == approx(2)
    assert pytools_hist.weighted_quantile(v, [0.5], sample_weight=h) == approx(2)
    assert pytools_hist.weighted_quantile(v, [1], sample_weight=h) == approx(2)

    data = [2, 6]
    h, b = np.histogram(data, bins=np.arange(-0.5, 10.5))
    v = 0.5 * (b[1:] + b[:-1])

    assert pytools_hist.weighted_quantile(v, [0], sample_weight=h) == approx(2)
    assert pytools_hist.weighted_quantile(v, [0.5], sample_weight=h) == approx(4)
    assert pytools_hist.weighted_quantile(v, [1], sample_weight=h) == approx(6)


def test_histogram_stats():
    data = [3]
    h, b = np.histogram(data, bins=np.arange(-0.5, 10.5))

    stats = pytools_hist.histogram_stats(h, b)
    assert "mean" in stats and "var" in stats
    assert stats["mean"] == approx(np.mean(data))
    assert stats["var"] == approx(np.var(data))
    assert stats["sigma"] == approx(np.std(data))

    data = [1, 3, 4, 2, 3, 8, 9, 2, 1, 4, 1]
    h, b = np.histogram(data, bins=np.arange(-0.5, 10.5))

    stats = pytools_hist.histogram_stats(h, b)
    assert "mean" in stats and "var" in stats
    assert stats["mean"] == approx(np.mean(data))
    assert stats["var"] == approx(np.var(data))
    assert stats["sigma"] == approx(np.std(data))


def test_histogram_robust_stats():
    data = [3]
    h, b = np.histogram(data, bins=np.arange(-0.5, 10.5))

    stats = pytools_hist.histogram_robust_stats(h, b)
    assert stats["median"] == approx(np.median(data))
    assert stats["mad"] == approx(np.median(np.abs(data - stats["median"])))

    data = [1, 3, 4, 2, 3, 8, 9, 2, 1, 4, 1]
    h, b = np.histogram(data, bins=np.arange(-0.5, 10.5))

    stats = pytools_hist.histogram_robust_stats(h, b)
    assert stats["median"] == approx(np.median(data), rel=0.5)
    assert stats["mad"] == approx(np.median(np.abs(data - stats["median"])), rel=0.25)


@pytest.mark.parametrize(
    "image_mrc,expected_min, expected_max, expected_floor, expected_limit",
    [
        (sitk.sitkUInt8, 0, 0, 0, 0),
        (sitk.sitkUInt16, 0, 0, 0, 0),
        ("uint16_uniform", 8191, 57344, 0, 65535),
        ("float32_uniform", 0, 1, 0, 1),
        ("uint8_bimodal", -64, 319, 0, 255),
    ],
    indirect=["image_mrc"],
)
def test_build_histogram_main(image_mrc, expected_min, expected_max, expected_floor, expected_limit):
    res = pytools.visual_min_max(Path(image_mrc), mad_scale=1.5)

    assert "neuroglancerPrecomputedMin" in res
    assert "neuroglancerPrecomputedMax" in res
    assert "neuroglancerPrecomputedFloor" in res
    assert "neuroglancerPrecomputedLimit" in res
    assert float(res["neuroglancerPrecomputedMin"]) == expected_min
    assert float(res["neuroglancerPrecomputedMax"]) == expected_max
    assert float(res["neuroglancerPrecomputedFloor"]) == expected_floor
    assert float(res["neuroglancerPrecomputedLimit"]) == expected_limit
    assert type(res["neuroglancerPrecomputedMin"]) == str
    assert type(res["neuroglancerPrecomputedMax"]) == str
    assert type(res["neuroglancerPrecomputedFloor"]) == str
    assert type(res["neuroglancerPrecomputedLimit"]) == str


def test_build_histogram_zarr_main(image_ome_ngff):
    res = pytools.visual_min_max(Path(image_ome_ngff) / "0", mad_scale=1.5)

    assert "neuroglancerPrecomputedMin" in res
    assert "neuroglancerPrecomputedMax" in res
    assert "neuroglancerPrecomputedFloor" in res
    assert "neuroglancerPrecomputedLimit" in res
    assert type(res["neuroglancerPrecomputedMin"]) == str
    assert type(res["neuroglancerPrecomputedMax"]) == str
    assert type(res["neuroglancerPrecomputedFloor"]) == str
    assert type(res["neuroglancerPrecomputedLimit"]) == str
