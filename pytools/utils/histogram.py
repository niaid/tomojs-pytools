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

import numpy as np
import logging
import zarr
import dask.array
from typing import Tuple, Any
from abc import ABC, abstractmethod
import math

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

    def compute_histogram_bin_edges(self, number_of_bins=1024) -> np.array:
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

    @abstractmethod
    def compute_histogram(self, histogram_bin_edges=None, density=False) -> Tuple[np.array, np.array]:
        pass


class DaskHistogramHelper(HistogramBase):
    def __init__(self, arr: dask.array):
        self._arr = arr
        if not self._arr.dtype.isnative:
            logger.info("ZARR array needs converting to native byteorder.")
            self._arr = self._arr.astype(self._arr.dtype.newbyteorder("="))

    def compute_min_max(self):
        return self._arr.min(), self._arr.max()

    @property
    def dtype(self):
        return self._arr.dtype

    def compute_histogram(self, histogram_bin_edges=None, density=False) -> Tuple[np.array, np.array]:
        if histogram_bin_edges is None:
            if np.issubdtype(self.dtype, np.integer) and np.iinfo(self.dtype).bits <= 16:
                histogram_bin_edges = self.compute_histogram_bin_edges(
                    number_of_bins=2 ** np.iinfo(self.dtype).bits + 1
                )

                if np.dtype(self.dtype) in (np.uint8, np.uint16):
                    # optimize chunks for ravel operations.
                    new_chunk = (None,) * (self._arr.ndim - 1) + (-1,)
                    arr = self._arr.rechunk(new_chunk).ravel()
                    return (
                        dask.array.bincount(arr, minlength=len(histogram_bin_edges) - 1).compute(),
                        histogram_bin_edges,
                    )

            else:
                histogram_bin_edges = self.compute_histogram_bin_edges()

        h, bins = dask.array.histogram(self._arr, bins=histogram_bin_edges, density=density)
        return h.compute(), bins


class ZARRHistogramHelper(DaskHistogramHelper):
    def __init__(self, filename):
        za = zarr.open_array(filename, mode="r")
        super().__init__(dask.array.from_zarr(za))
        logging.debug(za.info)


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
