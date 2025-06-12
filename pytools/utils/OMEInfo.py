#
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

import xml.etree.ElementTree as ET
from typing import Iterable
import logging

from pytools.data import ROILabel, ROIRectangle, OMEROIModel

logging.getLogger(__name__)


class OMEInfo:
    _ome_ns = {"OME": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}

    def __init__(self, ome_xml_string):
        self._root_element = ET.fromstring(ome_xml_string)

    def _image_element(self, image_index) -> ET.Element:
        return self._root_element.findall("OME:Image", self._ome_ns)[image_index]

    def number_of_images(self) -> int:
        return len(self._root_element.findall("OME:Image", self._ome_ns))

    def image_names(self) -> Iterable[str]:
        for counter, e in enumerate(self._root_element.iterfind("OME:Image", self._ome_ns)):
            yield e.attrib["Name"] or f"Scene #{counter}"

    def channel_names(self, image_index) -> Iterable[str]:
        el = self._image_element(image_index).iterfind("OME:Pixels/OME:Channel", self._ome_ns)
        for e in el:
            if "Name" in e.attrib:
                yield e.attrib["Name"]

    def maybe_flourescence(self, image_index):
        """
        Checks if all the channels' "IlluminationType" attribute is "Epifluorescence".
        """

        px_element = self._image_element(image_index).find("OME:Pixels", self._ome_ns)
        channel_elements = px_element.findall("./OME:Channel", self._ome_ns)

        def _check_channel(channel_element: ET.Element):
            if channel_element.attrib["SamplesPerPixel"] != "1":
                return False
            if channel_element.attrib.get("IlluminationType") == "Epifluorescence":
                return True
            return False

        return all([_check_channel(ce) for ce in channel_elements])

    def maybe_rgb(self, image_index):
        px_element = self._image_element(image_index).find("OME:Pixels", self._ome_ns)
        channel_elements = px_element.findall("./OME:Channel", self._ome_ns)
        three_or_four = len(channel_elements) in [3, 4]

        def _check_channel(channel_element: ET.Element):
            # return true if IlluminationType="Transmitted"
            if channel_element.attrib["SamplesPerPixel"] != "1":
                return False

            exclude_list_attribs = ["EmissionWavelength", "IlluminationType", "Fluor"]
            no_rgb_exclude_attrib = all([False for x in exclude_list_attribs if x in channel_element.keys()])

            if channel_element.attrib.get("Fluor") == "TL Brightfield":
                return True
            if channel_element.attrib.get("IlluminationType") == "Transmitted":
                return True
            return no_rgb_exclude_attrib

        return three_or_four and all([_check_channel(ce) for ce in channel_elements])

    def dimension_order(self, image_index):
        return self._image_element(image_index).find("OME:Pixels", self._ome_ns).attrib["DimensionOrder"]

    def size(self, image_index):
        px_element = self._image_element(image_index).find("OME:Pixels", self._ome_ns)
        size = []
        for d in self.dimension_order(image_index):
            size.append(int(px_element.get(f"Size{d}", 1)))
        return size

    def spacing(self, image_index):
        px_element = self._image_element(image_index).find("OME:Pixels", self._ome_ns)
        spacing = []
        for d in self.dimension_order(image_index):
            attr = "PhysicalSize{}".format(d)
            spacing.append(float(px_element.get(attr, 1.0)))
        return spacing

    def units(self, image_index):
        px_element = self._image_element(image_index).find("OME:Pixels", self._ome_ns)
        default = "µm"
        units = []
        for d in self.dimension_order(image_index):
            attr = "PhysicalSize{}Unit".format(d)
            units.append(px_element.get(attr, default))
        return units

    def roi(self, image_index) -> Iterable[OMEROIModel]:
        """
        Extracts the ROI referenced by an image in the OME XML data.

        The OME-XML specifications for ROI models is here:
            https://docs.openmicroscopy.org/ome-model/5.6.3/developers/roi.html

        """

        for roiref_el in self._image_element(image_index).iterfind("OME:ROIRef", self._ome_ns):
            roi_id = roiref_el.attrib["ID"]
            for roi_el in self._root_element.iterfind(f".//OME:ROI[@ID='{roi_id}']", self._ome_ns):
                # The ROI element may have and ID and Name attribute may be useful context if the label is not
                # sufficient. This is not currently used in the implementation.
                roi_model = OMEROIModel(
                    id=roi_id, name=roi_el.attrib.get("Name", None), description=roi_el.attrib.get("Description", None)
                )
                # iterate over all child elements of the ROI/Unions
                for union_el in roi_el.findall(".//OME:Union", self._ome_ns):
                    for child_el in union_el:
                        if child_el.tag == f"{{{self._ome_ns['OME']}}}Rectangle":
                            roi_model.union.append(
                                ROIRectangle(
                                    x=float(child_el.attrib["X"]),
                                    y=float(child_el.attrib["Y"]),
                                    width=float(child_el.attrib["Width"]),
                                    height=float(child_el.attrib["Height"]),
                                )
                            )
                        elif child_el.tag == f"{{{self._ome_ns['OME']}}}Label":
                            roi_model.union.append(
                                ROILabel(
                                    x=float(child_el.attrib["X"]),
                                    y=float(child_el.attrib["Y"]),
                                    text=child_el.attrib["Text"],
                                )
                            )
                yield roi_model
