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
import SimpleITK as sitk
import numpy as np


@pytest.fixture(
    scope="session",
    params=[sitk.sitkUInt8, sitk.sitkInt16, sitk.sitkUInt16, sitk.sitkFloat32, "uint16_uniform", "uint8_bimodal"],
)
def image_mrc(request, tmp_path_factory):
    if isinstance(request.param, str) and request.param == "uint16_uniform":

        print(f"Calling image_mrc with {request.param}")
        fn = f"image_mrc_{request.param.replace(' ', '_')}.mrc"

        a = np.linspace(0, 2**16 - 1, num=2**16, dtype="uint16").reshape(16, 64, 64)
        img = sitk.GetImageFromArray(a)
        img.SetSpacing([1.23, 1.23, 4.96])

    elif isinstance(request.param, str) and request.param == "uint8_bimodal":

        print(f"Calling image_mrc with {request.param}")
        fn = f"image_mrc_{request.param.replace(' ', '_')}.mrc"

        a = np.zeros([16, 16, 16], np.uint8)
        a[len(a) // 2 :] = 255
        img = sitk.GetImageFromArray(a)
        img.SetSpacing([12.3, 12.3, 56.7])
    else:
        pixel_type = request.param
        print(f"Calling image_mrc with {sitk.GetPixelIDValueAsString(pixel_type)}")
        fn = f"image_mrc_{sitk.GetPixelIDValueAsString(pixel_type).replace(' ', '_')}.mrc"

        size = [10, 9, 8]
        if pixel_type == sitk.sitkFloat32:

            a = np.linspace(0.0, 1.0, num=np.prod(size), dtype=np.float32).reshape(*size[::-1])
            img = sitk.GetImageFromArray(a)
        else:
            # image of just zeros
            img = sitk.Image([10, 9, 8], pixel_type)
        img.SetSpacing([1.1, 1.2, 1.3])

    fn = tmp_path_factory.mktemp("data").joinpath(fn)
    sitk.WriteImage(img, str(fn))
    return str(fn)


@pytest.fixture(
    scope="session",
    params=[sitk.sitkInt8, sitk.sitkUInt8, sitk.sitkInt16, sitk.sitkUInt16, sitk.sitkFloat32],
)
def image_tiff(request, tmp_path_factory):

    pixel_type = request.param
    print(f"Calling image_tiff with {sitk.GetPixelIDValueAsString(pixel_type)}")
    fn = f"image_tiff_{sitk.GetPixelIDValueAsString(pixel_type).replace(' ', '_')}.tiff"
    img = sitk.Image([10, 9, 8], pixel_type)

    fn = tmp_path_factory.mktemp("data").joinpath(fn)
    sitk.WriteImage(img, str(fn))
    return str(fn)
