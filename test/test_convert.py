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
def test_is_int16_tif(image_tiff, tmp_path):
    out_file = tmp_path / "out.tiff"

    file_to_uint8(image_tiff, out_file_path=out_file)

    reader = make_reader(out_file)

    assert reader.GetPixelID() == sitk.sitkUInt8
