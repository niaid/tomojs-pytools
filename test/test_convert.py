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

from pytools.convert import file_to_uint8
from pytools.meta import _make_image_file_reader_with_info as make_reader
import pytest
import SimpleITK as sitk


@pytest.mark.parametrize(
    "image_tiff",
    [
        sitk.sitkUInt8,
        sitk.sitkInt8,
        sitk.sitkInt16,
        sitk.sitkUInt16,
        sitk.sitkFloat32,
    ],
    indirect=["image_tiff"],
)
def test_file_to_uint8_tif(image_tiff, tmp_path):
    out_file = tmp_path / "out.tiff"

    file_to_uint8(image_tiff, out_file_path=out_file)

    reader = make_reader(out_file)

    assert reader.GetPixelID() == sitk.sitkUInt8


@pytest.mark.parametrize(
    "min_value,max_value",
    [
        (0, 0),
        (-1, 1),
        (0, 255),
        (-32768, 0),
        (0, 32767),
        (-32768, 32767),
        # these cases cause overflow issue due to rounding
        # (-32768, -32767)
        # (32766, 32767),
    ],
)
def test_file_to_uint8_values(min_value, max_value, tmp_path):
    """
    Test extra ranges of min, max values.
    """
    in_file = tmp_path / "in.tiff"
    out_file = tmp_path / "out.tiff"

    img = sitk.Image([3, 1], sitk.sitkInt16)

    img[0, 0] = min_value
    img[1, 0] = max_value
    print(img[0, 0], img[1, 0])

    sitk.WriteImage(img, in_file)

    file_to_uint8(in_file, out_file)

    rimg = sitk.ReadImage(out_file)

    print(
        rimg[0, 0],
        rimg[1, 0],
    )
    assert rimg[0, 0] == 0
    if min_value != max_value:
        assert rimg[1, 0] == 255
    assert rimg.GetPixelID() == sitk.sitkUInt8


def test_file_to_uint8_round(tmp_path):
    in_file = tmp_path / "in.tiff"
    out_file = tmp_path / "out.tiff"

    img = sitk.Image([257, 1], sitk.sitkInt16)

    for i in range(257):
        img[i, 0] = i

    sitk.WriteImage(img, in_file)

    file_to_uint8(in_file, out_file)

    rimg = sitk.ReadImage(out_file)

    for i in range(257):
        print(i * 255 / 256)
        assert rimg[i, 0] == i - (i > 0)
