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
from pytools.utils.OMEInfo import OMEInfo
import pytest
from pytools.data import ROILabel, ROIRectangle, OMEROIModel


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


@pytest.mark.parametrize("xml_file", ["IA_P2_S1.ome.xml"])
def test_ome_annotations_info(data_path, xml_file):
    """
    Tests attributes of valid zarr image OMEInfo object
    """
    with open(data_path / xml_file, "r") as fp:
        ome_info = OMEInfo(fp.read())

    assert ome_info.number_of_images() == 1

    # check the number of ROI models in the OMEInfo object
    assert len(list(ome_info.roi(0))) == 12

    for i, ome_roi_model in enumerate(ome_info.roi(0)):
        # check ome_roi_model is of type OMEROIModel
        assert isinstance(ome_roi_model, OMEROIModel)

        assert ome_roi_model.id == f"ROI:{i}"
        assert len(ome_roi_model.union) == 2

        # check the first ROI in the union is of type ROILabel
        assert isinstance(ome_roi_model.union[0], ROILabel)
        assert ome_roi_model.union[0].text == f"{i+1:03d}"

        # check the second ROI in the union is of type ROIRectangle
        assert isinstance(ome_roi_model.union[1], ROIRectangle)


@pytest.mark.parametrize("xml_file", ["dapi_one_channel.xml"])
def test_ome_fluorescence(data_path, xml_file):
    """
    Tests attributes of valid zarr image OMEInfo object
    """
    with open(data_path / xml_file, "r") as fp:
        ome_info = OMEInfo(fp.read())

    assert ome_info.number_of_images() == 6

    for idx in range(ome_info.number_of_images() - 2):
        assert ome_info.maybe_flourescence(idx) is True
        assert ome_info.maybe_rgb(idx) is False
        assert tuple(ome_info.channel_names(idx)) == ("DAPI",)

    # image 5 and 6 should be RGB not fluorescence
    assert ome_info.maybe_flourescence(4) is False
    assert ome_info.maybe_rgb(4) is True
    assert ome_info.maybe_flourescence(5) is False
    assert ome_info.maybe_rgb(5) is True

    # Check that the last two images are named "label image" and "macro image"
    assert tuple(ome_info.image_names()) == (
        "ScanRegion0",
        "ScanRegion1",
        "ScanRegion2",
        "ScanRegion3",
        "label image",
        "macro image",
    )


@pytest.mark.parametrize("xml_file", ["he_rgb.xml"])
def test_ome_rgb(data_path, xml_file):
    """
    Tests attributes of valid zarr image OMEInfo object
    """
    with open(data_path / xml_file, "r") as fp:
        ome_info = OMEInfo(fp.read())

    assert ome_info.number_of_images() == 5

    for idx in range(ome_info.number_of_images() - 2):
        assert ome_info.maybe_flourescence(idx) is False
        assert ome_info.maybe_rgb(idx) is True
        assert tuple(ome_info.channel_names(idx)) == ("TL Brightfield", "TL Brightfield", "TL Brightfield")

    # image 5 and 6 should be RGB not fluorescence
    assert ome_info.maybe_flourescence(4) is False
    assert ome_info.maybe_rgb(3) is True
    assert ome_info.maybe_flourescence(4) is False
    assert ome_info.maybe_rgb(4) is True

    # Check that the last two images are named "label image" and "macro image"
    assert tuple(ome_info.image_names()) == ("ScanRegion0", "ScanRegion1", "ScanRegion2", "label image", "macro image")
