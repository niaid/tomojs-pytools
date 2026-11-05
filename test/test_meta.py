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

from pytools.meta import is_int16, is_16bit, read_ome_info
import pytest
import SimpleITK as sitk


@pytest.mark.parametrize(
    "image_mrc,expected_result",
    [
        (sitk.sitkUInt8, False),
        (sitk.sitkInt16, True),
        (sitk.sitkUInt16, False),
        (sitk.sitkFloat32, False),
    ],
    indirect=["image_mrc"],
)
def test_is_int16_mrc(image_mrc, expected_result):
    assert is_int16(image_mrc) == expected_result


@pytest.mark.parametrize(
    "image_tiff,expected_result",
    [
        (sitk.sitkUInt8, False),
        (sitk.sitkInt8, False),
        (sitk.sitkInt16, True),
        (sitk.sitkUInt16, False),
        (sitk.sitkFloat32, False),
    ],
    indirect=["image_tiff"],
)
def test_is_int16_tif(image_tiff, expected_result):
    assert is_int16(image_tiff) == expected_result


@pytest.mark.parametrize(
    "image_tiff,expected_result",
    [
        (sitk.sitkUInt8, False),
        (sitk.sitkInt8, False),
        (sitk.sitkInt16, True),
        (sitk.sitkUInt16, True),
        (sitk.sitkFloat32, False),
    ],
    indirect=["image_tiff"],
)
def test_is_16bit_tif(image_tiff, expected_result):
    assert is_16bit(image_tiff) == expected_result


def test_read_ome_info(data_path):
    ome_info = read_ome_info(data_path / "ome_jpeg2000.ome.tif")
    assert ome_info.number_of_images() == 1
    assert tuple(ome_info.image_names()) == ("test",)


def test_read_ome_info_not_ome(tmp_path):
    fn = tmp_path / "not_ome.tif"
    sitk.WriteImage(sitk.Image([10, 9], sitk.sitkUInt16), str(fn))

    with pytest.raises(ValueError):
        read_ome_info(fn)
