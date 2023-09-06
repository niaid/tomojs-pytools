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


def test_ome_info(data_path):
    """
    Tests attributes of valid zarr image OMEInfo object
    """
    zarr_name = "KC_M3_S2_ReducedImageSubset2.zarr"
    zi = HedwigZarrImages(data_path / zarr_name)
    assert zi.ome_xml_path is not None
    assert zi.ome_info is not None
    ome = zi.ome_info

    assert ome.number_of_images() == 3
    # assert all(list(ome.image_names()))
    assert len(list(ome.channel_names(0))) == 2
    assert len(list(ome.channel_names(1))) == 0
    assert len(list(ome.channel_names(2))) == 0
