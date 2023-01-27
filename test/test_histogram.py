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
import pytools.ng.build_histogram as pytools_hist
import pytest
from pytest import approx
import pytools.ng.build_histogram
import SimpleITK as sitk
import numpy as np
from click.testing import CliRunner
import json


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
    "image_mrc",
    [
        sitk.sitkUInt8,
        sitk.sitkInt16,
        sitk.sitkUInt16,
    ],
    indirect=["image_mrc"],
)
def test_stream_build_histogram(image_mrc):

    bin_edges1 = np.arange(np.iinfo(np.int16).min - 0.5, np.iinfo(np.uint16).max + 1.5)

    img = sitk.ReadImage(image_mrc)

    test_args = [
        {},
        {"extract_axis": 0},
        {"histogram_bin_edges": bin_edges1},
        {"histogram_bin_edges": bin_edges1, "extract_axis": 0},
        {"histogram_bin_edges": bin_edges1, "extract_axis": 1},
        {"histogram_bin_edges": bin_edges1, "extract_axis": 2},
        {"extract_axis": 0, "extract_step": 2},
        {"extract_axis": 1, "extract_step": 3},
        {"extract_axis": 2, "extract_step": 5},
        {"histogram_bin_edges": bin_edges1, "extract_axis": 0, "extract_step": 99},
        {"histogram_bin_edges": bin_edges1, "extract_axis": 1, "extract_step": 45},
        {"histogram_bin_edges": bin_edges1, "extract_axis": 2, "extract_step": 32},
    ]
    for args in test_args:
        h, b = pytools_hist.stream_build_histogram(image_mrc, **args)

        assert np.sum(h) == img.GetNumberOfPixels(), f"with args '{args}'"
        assert np.sum(h * (b[1:] + b[:-1])) == np.sum(sitk.GetArrayViewFromImage(img)), f"with args '{args}'"


args = ["--help", "--version"]


@pytest.mark.parametrize("cli_args", args)
def test_histogram_mai_help(cli_args):
    runner = CliRunner()
    result = runner.invoke(pytools.ng.build_histogram.main, cli_args.split())
    assert not result.exception


@pytest.mark.parametrize(
    "image_mrc,expected_min, expected_max, expected_floor, expected_limit, clamp",
    [
        (sitk.sitkUInt8, 0, 0, 0, 0, False),
        (sitk.sitkInt16, 0, 0, 0, 0, True),
        (sitk.sitkUInt16, 0, 0, 0, 0, False),
        ("uint16_uniform", 8191, 57344, 0, 65535, True),
        ("uint16_uniform", 8191, 57344, 0, 65535, False),
        ("uint8_bimodal", 0, 255, 0, 255, True),
        ("uint8_bimodal", -64, 319, 0, 255, False),
        (sitk.sitkFloat32, 0, 1, 0, 1, True),
    ],
    indirect=["image_mrc"],
)
def test_build_histogram_main(image_mrc, expected_min, expected_max, expected_floor, expected_limit, clamp):
    runner = CliRunner()
    output_filename = "out.json"
    args = [image_mrc, "--mad", "1.5", "--output-json", output_filename]
    if clamp:
        args.append("--clamp")
    print(args)
    with runner.isolated_filesystem():
        result = runner.invoke(pytools.ng.build_histogram.main, args=args)
        assert not result.exception
        with open(output_filename) as fp:
            res = json.load(fp)

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
