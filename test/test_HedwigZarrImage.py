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

import pytest
import shutil
from pytools.HedwigZarrImages import HedwigZarrImages


@pytest.mark.parametrize(
    "zarr_name, image_ext, array_shape, dims, shader_type, ngff_dims, shader_params",
    [("OM_P1_S1_ScanOnly_1k.zarr", "png", (1, 3, 1, 1024, 521), "XYC", "MultiChannel", "TCZYX", {})],
)
def test_HedwigZarrImage_info(
    data_path, zarr_name, image_ext, array_shape, dims, shader_type, ngff_dims, shader_params, tmp_path
):
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
