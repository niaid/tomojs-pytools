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

from pytools.HedwigZarrImages import HedwigZarrImages
import pytest
import shutil


@pytest.fixture(
    scope="function",
    params=["2013-1220-dA30_5-BSC-1_22_full_rec.zarr", "OM_P1_S1_ScanOnly_1k.zarr", "OM_P1_S1_ScanOnly_1k_tiff.zarr"],
)
def temp_zarr_path(request, tmp_path_factory, data_path):
    """Copies ZARRs to temporary directory for modification."""

    zarr_name = request.param
    tmp_zarr = tmp_path_factory.mktemp("Z") / zarr_name
    shutil.copytree(data_path / zarr_name, tmp_zarr)

    return tmp_zarr


@pytest.mark.parametrize(
    "zarr_name, image_ext, array_shape, dims, shader_type, ngff_dims, shader_params",
    [
        (
            "2013-1220-dA30_5-BSC-1_22_full_rec.zarr",
            "mrc",
            (1, 1, 512, 87, 512),
            "XYZ",
            "Grayscale",
            "TCZYX",
            {"window": [-2664, 1677], "range": [263, 413]},
        ),
        ("OM_P1_S1_ScanOnly_1k.zarr", "png", (1, 3, 1, 1024, 521), "XYC", "RGB", "TCZYX", {}),
        ("OM_P1_S1_ScanOnly_1k_tiff.zarr", "tiff", (1, 3, 1, 1024, 521), "XYC", "RGB", "TCZYX", {}),
    ],
)
def test_HedwigZarrImage_info(
    data_path, zarr_name, image_ext, array_shape, dims, shader_type, ngff_dims, shader_params
):
    zi = HedwigZarrImages(data_path / zarr_name)

    for k, z_img in zi.series():
        assert image_ext in k
        assert array_shape == z_img.shape
        assert dims == z_img.dims
        assert shader_type == z_img.shader_type
        assert ngff_dims == z_img._ome_ngff_multiscale_dims()
        assert shader_params == z_img.neuroglancer_shader_parameters()


@pytest.mark.parametrize(
    "target_chunk",
    [2048, 512, 64],
)
def test_HedwigZarrImage_rechunk(target_chunk, temp_zarr_path):
    """Test rechunking all array is a ZARR structure"""

    zi = HedwigZarrImages(temp_zarr_path, read_only=False)
    for k, hzi in zi.series():
        hzi.rechunk(target_chunk)

    for k, hzi in zi.series():
        for level in range(len(hzi._ome_ngff_multiscales()["datasets"])):
            arr = hzi._ome_ngff_multiscale_get_array(level)
            # check for expected chunking
            for s, c, d in zip(arr.shape, arr.chunks, hzi._ome_ngff_multiscale_dims()):
                if d == "T":
                    assert s == 1
                elif d == "C" or s < target_chunk:
                    assert s == c
                else:
                    assert c == target_chunk
