from dataclasses import dataclass, field
from typing import Optional, List, Union


@dataclass
class OMEDataObject:
    """
    Generic base object for OME data defined in the OME XML
    """

    pass


@dataclass
class ROIRectangle(OMEDataObject):
    x: float
    y: float
    width: float
    height: float

    the_z: Optional[float] = None
    the_t: Optional[float] = None
    the_c: Optional[float] = None

    @property
    def point_a(self):
        return self.x, self.y

    @property
    def point_b(self):
        return self.x + self.width, self.y + self.height


@dataclass
class ROILabel(OMEDataObject):
    x: float
    y: float
    text: str


@dataclass
class OMEROIModel(OMEDataObject):
    """
    Represents the OME ROI model. Which contains a set of annotations.
    """

    id: str
    name: Optional[str] = None
    description: Optional[str] = None
    union: List[Union[ROIRectangle, ROILabel]] = field(default_factory=list)
