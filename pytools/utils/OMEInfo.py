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


class OMEInfo:
    _ome_ns = {"OME": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}

    def __init__(self, ome_xml_string):
        self._root_element = ET.fromstring(ome_xml_string)

    def _image_element(self, image_index) -> ET.Element:
        return self._root_element.findall("OME:Image", self._ome_ns)[image_index]

    def number_of_images(self) -> int:
        return len(self._root_element.findall("OME:Image", self._ome_ns))

    def image_names(self) -> Iterable[str]:
        for e in self._root_element.iterfind("OME:Image", self._ome_ns):
            yield e.attrib["Name"]

    def channel_names(self, image_index) -> Iterable[str]:
        el = self._image_element(image_index).iterfind("OME:Pixels/OME:Channel", self._ome_ns)
        for e in el:
            if "Name" in e.attrib:
                yield e.attrib["Name"]

    def maybe_rgb(self, image_index):
        px_element = self._image_element(image_index).find("OME:Pixels", self._ome_ns)
        channel_elements = px_element.findall("./OME:Channel", self._ome_ns)
        three_or_four = len(channel_elements) in [3, 4]
        is_interleaved = "Interleaved" in px_element.attrib and px_element.attrib["Interleaved"].lower() == "true"

        def _check_channel(channel_element: ET.Element):
            exclude_list_attribs = ["EmissionWavelength", "IlluminationType", "Flour"]
            no_rgb_exclude_attrib = all([False for x in exclude_list_attribs if x in channel_element.keys()])
            return no_rgb_exclude_attrib and channel_element.attrib["SamplesPerPixel"] == "1"

        return three_or_four and is_interleaved and all([_check_channel(ce) for ce in channel_elements])

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
