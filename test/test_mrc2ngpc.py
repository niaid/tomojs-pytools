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

from click.testing import CliRunner
from pytools.ng.mrc2nifti import main
import pytest
import SimpleITK as sitk
import numpy as np


@pytest.fixture(
    scope="session",
    params=[sitk.sitkUInt8, sitk.sitkInt16, sitk.sitkUInt16, sitk.sitkFloat32],
)
def image_mrc(request, tmp_path_factory):

    pixel_type = request.param
    print(f"Calling image_mrc with {sitk.GetPixelIDValueAsString(pixel_type)}")
    fn = f"image_mrc_{sitk.GetPixelIDValueAsString(pixel_type).replace(' ', '_')}.mrc"
    img = sitk.Image([10, 9, 8], pixel_type)
    fn = tmp_path_factory.mktemp("data").joinpath(fn)
    sitk.WriteImage(img, str(fn))
    return str(fn)


args = ["--help", "--version"]


@pytest.mark.parametrize("cli_args", args)
def test_mrc2nifti_main_help(cli_args):
    runner = CliRunner()
    result = runner.invoke(main, cli_args.split())
    assert not result.exception


@pytest.mark.parametrize(
    "image_mrc,expected_pixel_type",
    [
        (sitk.sitkUInt8, sitk.sitkUInt8),
        (sitk.sitkInt16, sitk.sitkUInt16),
        (sitk.sitkUInt16, sitk.sitkUInt16),
        (sitk.sitkFloat32, sitk.sitkFloat32),
    ],
    indirect=["image_mrc"],
)
def test_mrc2nifti(image_mrc, expected_pixel_type):
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(main, [image_mrc, "out.nii"])
        assert not result.exception
        img = sitk.ReadImage("out.nii")

    assert img.GetPixelID() == expected_pixel_type
    assert img.GetSize() == (10, 9, 8)
    np.testing.assert_allclose(img.GetSpacing(), (1e-7, 1e-7, 1e-7))
