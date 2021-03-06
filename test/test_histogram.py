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
from fixtures import image_mrc
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

    bin_edges = np.arange(np.iinfo(np.int16).min - 0.5, np.iinfo(np.uint16).max + 1.5)

    h, b = pytools_hist.stream_build_histogram(image_mrc, bin_edges)

    img = sitk.ReadImage(image_mrc)
    assert np.sum(h) == img.GetNumberOfPixels()
    assert np.sum(h * (b[1:] + b[:-1])) == np.sum(sitk.GetArrayViewFromImage(img))


args = ["--help", "--version"]


@pytest.mark.parametrize("cli_args", args)
def test_histogram_mai_help(cli_args):
    runner = CliRunner()
    result = runner.invoke(pytools.ng.build_histogram.main, cli_args.split())
    assert not result.exception


@pytest.mark.parametrize(
    "image_mrc,expected_min, expected_max",
    [(sitk.sitkUInt8, 0, 0), (sitk.sitkInt16, 0, 0), (sitk.sitkUInt16, 0, 0)],
    indirect=["image_mrc"],
)
def test_build_histogram_main(image_mrc, expected_min, expected_max):
    runner = CliRunner()
    output_filename = "out.json"
    with runner.isolated_filesystem():
        result = runner.invoke(
            pytools.ng.build_histogram.main, [image_mrc, "--mad", "5", "--output-json", output_filename]
        )
        assert not result.exception
        with open(output_filename) as fp:
            res = json.load(fp)

    assert "min" in res
    assert "max" in res
    assert res["min"] == expected_min
    assert res["max"] == expected_max
