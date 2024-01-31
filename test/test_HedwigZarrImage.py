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
import pytest
from pytest import approx
import shutil
from pytools.HedwigZarrImages import HedwigZarrImages
from pytools.HedwigZarrImage import HedwigZarrImage


@pytest.mark.parametrize(
    "zarr_name, image_ext, array_shape, dims, shader_type, ngff_dims, spacing, shader_params",
    [
        (
            "OM_P1_S1_ScanOnly_1k.zarr",
            "png",
            (1, 3, 1, 1024, 521),
            "XYC",
            "MultiChannel",
            "TCZYX",
            [1.0, 1.0, 1.0, 1.0, 1.0],
            {},
        )
    ],
)
def test_HedwigZarrImage_info_bad_ome(
    data_path, zarr_name, image_ext, array_shape, dims, shader_type, ngff_dims, spacing, shader_params, tmp_path
):
    """
    Testcase assumes that OME directory can be missing.
    """
    # Remove OME directory from the .zarr file
    shutil.copytree(data_path / zarr_name, tmp_path / zarr_name)
    shutil.rmtree(tmp_path / zarr_name / "OME")

    zi = HedwigZarrImages(tmp_path / zarr_name)
    assert zi.ome_xml_path is None
    assert zi.ome_info is None
    # keys = list(zi.get_series_keys())

    for k, z_img in zi.series():
        assert array_shape == z_img.shape
        assert dims == z_img.dims
        assert shader_type == z_img.shader_type
        assert ngff_dims == z_img._ome_ngff_multiscale_dims()
        assert shader_params == z_img.neuroglancer_shader_parameters()
        for i in range(len(spacing)):
            assert spacing[i] == approx(z_img.spacing[i])


@pytest.mark.parametrize(
    "zarr_name, attrs",
    [
        (
            "KC_M3_S2_ReducedImageSubset2.zarr",
            [
                (
                    "czi",
                    (1, 2, 1, 3102, 206),
                    "XYC",
                    "MultiChannel",
                    "TCZYX",
                    [1.0, 1.0, 1.0, 0.3252445, 0.3252445],
                    {"channelArray": 2},
                ),
                ("label", (1, 3, 1, 758, 1588), "XYC", "RGB", "TCZYX", [1.0, 1.0, 1.0, 0.3252445, 0.3252445], {}),
                ("macro", (1, 3, 1, 685, 567), "XYC", "RGB", "TCZYX", [1.0, 1.0, 1.0, 0.3252445, 0.3252445], {}),
            ],
        )
    ],
)
def test_HedwigZarrImage_info_for_czi(data_path, zarr_name, attrs):
    """
    Tests ZarrImages attributes generated from valid .czi files
    """
    zi = HedwigZarrImages(data_path / zarr_name)
    assert zi.ome_xml_path is not None
    image_names = list(zi.get_series_keys())
    assert len(image_names) == 3
    assert all(image_names)

    for (k, z_img), attr in zip(zi.series(), attrs):
        image_ext, array_shape, dims, shader_type, ngff_dims, spacing, shader_params = attr
        # assert image_ext in k
        assert array_shape == z_img.shape
        assert dims == z_img.dims
        assert shader_type == z_img.shader_type
        assert ngff_dims == z_img._ome_ngff_multiscale_dims()
        for i in range(len(spacing)):
            assert spacing[i] == approx(z_img.spacing[i])

        for param_key in shader_params:
            shader_params[param_key] == len(z_img.neuroglancer_shader_parameters()[param_key])


@pytest.mark.parametrize(
    "zarr_name, key",
    [
        ("KC_M3_S2_ReducedImageSubset2.zarr", "Scene #0"),
    ],
)
def test_HedwigZarrImage_info_for_czi_quantiles(data_path, zarr_name, key):
    """
    Test that the quantiles are computed correctly for the neuroglancer shader parameters
    """
    zi = HedwigZarrImages(data_path / zarr_name)
    assert zi.ome_xml_path is not None
    image_names = list(zi.get_series_keys())
    assert len(image_names) == 3
    assert all(image_names)

    z_img = zi[key]
    assert z_img._ome_ngff_multiscale_dims() == "TCZYX"

    shader_params = z_img._neuroglancer_shader_parameters_multichannel(zero_black_quantiles=False)

    for idx, channel_params in enumerate(shader_params["channelArray"]):
        param_window_min, param_window_max = channel_params["window"]
        qvalues = np.quantile(z_img._ome_ngff_multiscale_get_array(0)[:, idx, ...], [0.0, 0.9999])

        assert param_window_min == approx(qvalues[0], abs=1)
        # not sure why this is not more accurate
        assert param_window_max == approx(qvalues[1], abs=20)

    shader_params = z_img.neuroglancer_shader_parameters(middle_quantile=(0.25, 0.75))

    for idx, channel_params in enumerate(shader_params["channelArray"]):
        param_range_min, param_range_max = channel_params["range"]
        qvalues = np.quantile(z_img._ome_ngff_multiscale_get_array(0)[:, idx, ...], [0.25, 0.75])
        assert param_range_min == approx(qvalues[0], abs=1)
        assert param_range_max == approx(qvalues[1], abs=1)


@pytest.mark.parametrize("targetx, targety", [(300, 300), (600, 600), (1024, 1024)])
def test_hedwigimage_extract_2d(data_path, targetx, targety):
    """
    Tests extract_2d image extraction method for HedwigZarrImage
    Asserts largest dimension matches with the target size of the dimension
    """
    zarr_name = "KC_M3_S2_ReducedImageSubset2.zarr"
    zi = HedwigZarrImages(data_path / zarr_name)
    for k, z_img in zi.series():
        sitk_img = z_img.extract_2d(targetx, targety)
        x, y = sitk_img.GetSize()
        shape = dict(zip(z_img._ome_ngff_multiscale_dims(), z_img.shape))
        actualx, actualy = shape["X"], shape["Y"]
        assert x == targetx if actualx > actualy else y == targety


@pytest.mark.parametrize(
    "zarr_name, key, expected_spacing",
    [
        ("KC_M3_S2_ReducedImageSubset2.zarr", "Scene #0", 3.3608607),
        ("KC_M3_S2_ReducedImageSubset2.zarr", "label image", 1.7216279),
        ("KC_M3_S2_ReducedImageSubset2.zarr", "macro image", 0.74264179),
        ("OM_P1_S1_ScanOnly_1k.zarr", "OM_P1_S1_ScanOnly_1k.png", 3.4133333),
    ],
)
def test_hedwigimage_extract_2d_spacing(data_path, zarr_name, key, expected_spacing):
    """
    Tests extract_2d image extraction method for HedwigZarrImage
    Asserts largest dimension matches with the target size of the dimension
    """
    targetx, targety = 300, 300
    zi = HedwigZarrImages(data_path / zarr_name)
    z_img = zi[key]

    sitk_img = z_img.extract_2d(targetx, targety)
    x, y = sitk_img.GetSize()
    shape = dict(zip(z_img._ome_ngff_multiscale_dims(), z_img.shape))
    actualx, actualy = shape["X"], shape["Y"]
    assert sitk_img.GetDimension() == 2
    assert x == targetx if actualx > actualy else y == targety

    img_spacing = sitk_img.GetSpacing()
    assert img_spacing[0] == approx(img_spacing[1], rel=1e-12)
    assert img_spacing[0] == approx(expected_spacing, rel=1e-7)


def test_hedwigimage_extract_2d_invalid_shapes(data_path, monkeypatch):
    """
    Asserts extract_2d does not work with zarr image having dimensions:
        time > 1
        z-axis > 1
    """
    zarr_name = "KC_M3_S2_ReducedImageSubset2.zarr"
    zi = HedwigZarrImages(data_path / zarr_name)

    @property
    def _mock_shape(obj):
        return (2, 3, 1, 100, 100)

    monkeypatch.setattr(HedwigZarrImage, "shape", _mock_shape)

    z_img = zi[list(zi.get_series_keys())[0]]
    assert z_img.shape[0] == 2
    with pytest.raises(ValueError) as execinfo:
        z_img.extract_2d(300, 300)
    assert "Time dimension" in str(execinfo.value)

    @property
    def _mock_shape(obj):
        return (1, 3, 3, 100, 100)

    monkeypatch.setattr(HedwigZarrImage, "shape", _mock_shape)

    z_img = zi[list(zi.get_series_keys())[0]]
    assert z_img.shape[2] == 3
    with pytest.raises(ValueError) as execinfo:
        z_img.extract_2d(300, 300)
    assert "Z dimension" in str(execinfo.value)
