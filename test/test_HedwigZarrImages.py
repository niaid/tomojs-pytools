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

import json


@pytest.mark.parametrize(
    "zarr_name",
    ["2013-1220-dA30_5-BSC-1_22_full_rec.zarr", "OM_P1_S1_ScanOnly_1k.zarr"],
)
def test_HedwigZarrImage_info(data_path, zarr_name):
    zi = HedwigZarrImages(data_path / zarr_name)
    keys = list(zi.get_series_keys())
    print(f"zarr groups: {keys}")

    print(zi.ome_xml_path)

    for k, z_img in zi.series():
        print(f'image name: "{k}"')
        print(f"\tarray shape: {z_img.shape}")
        print(f"\tzarr path: {z_img.path}")
        print(f"\tdims: {z_img.dims}")
        print(f"\tshader type: {z_img.shader_type}")
        print(f"\tNGFF dims: {z_img._ome_ngff_multiscale_dims()}")
        print(f"\tshader params: {json.dumps(z_img.neuroglancer_shader_parameters())}")
