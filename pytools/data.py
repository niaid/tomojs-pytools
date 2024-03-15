from dataclasses import dataclass
from typing import Optional


@dataclass
class ROIRectangle:
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
class ROILabel:
    x: float
    y: float
    text: str
