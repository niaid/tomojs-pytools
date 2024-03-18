from dataclasses import dataclass, field
from typing import Optional, List, Union, Tuple


@dataclass
class OMEDataObject:
    """
    Generic base object for OME data defined in the OME XML.
    """

    pass


@dataclass
class ROIRectangle(OMEDataObject):
    """
    Represents a rectangle ROI in the OME XML

    Coordinates are in pixels.
    """

    x: float
    y: float
    width: float
    height: float

    the_z: Optional[float] = None
    the_t: Optional[float] = None
    the_c: Optional[float] = None

    @property
    def point_a(self) -> Tuple[float, float]:
        """
        Returns the top left point of the rectangle.
        """
        return self.x, self.y

    @property
    def point_b(self) -> Tuple[float, float]:
        """
        Returns the bottom right point of the rectangle.
        """
        return self.x + self.width, self.y + self.height


@dataclass
class ROILabel(OMEDataObject):
    """
    A spacial text label.

    Coordinates are in pixels.
    """

    x: float
    y: float
    text: str


@dataclass
class OMEROIModel(OMEDataObject):
    """
    Represents the OME ROI model. Which contains a union/list of annotations.

    See https://docs.openmicroscopy.org/ome-model/5.6.3/developers/roi.html for more information.

    The OME.TIFF files generated in spacial-omics microscopy have the label followed by the rectangle, so
    that the label could be used as the description of the Neuroglancer annotation.
    """

    id: str
    name: Optional[str] = None
    description: Optional[str] = None
    union: List[Union[ROIRectangle, ROILabel]] = field(default_factory=list)
