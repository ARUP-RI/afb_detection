from functools import cache
from typing import Union, Optional, List
from typing_extensions import Annotated

from pydantic import BaseModel, Field
from shapely.geometry import box
from shapely import union_all

PositiveInt = Annotated[int, Field(gt=0)]


@cache
def scaling(level: int) -> int:
    return 2**level


class PixelVector(BaseModel):
    x: int
    y: int


class PixelBox(BaseModel):
    left: int
    top: int
    width: PositiveInt
    height: PositiveInt

    def __eq__(self, other):
        return all(
            [
                self.left == other.left,
                self.top == other.top,
                self.width == other.width,
                self.height == other.height,
            ]
        )

    @property
    def right(self):
        return self.left + self.width

    @property
    def bottom(self):
        return self.top + self.height

    @property
    def size(self) -> PixelVector:
        return PixelVector(x=self.width, y=self.height)

    def scale(self, factor: Union[int, float]) -> "PixelBox":
        return PixelBox(
            left=round(self.left * factor),
            top=round(self.top * factor),
            width=round(self.width * factor) or 1,  # width must be a positive integer
            height=round(self.height * factor)
            or 1,  # height must be a positive integer
        )

    def at_level(self, level) -> "PixelBox":
        return self.scale(1 / scaling(level))

    @classmethod
    def left_top_right_bottom(cls, left: int, top: int, right: int, bottom: int):
        return cls(left=left, top=top, width=right - left, height=bottom - top)

    @staticmethod
    def overlap(box1: "PixelBox", box2: "PixelBox") -> Optional["PixelBox"]:
        if any(
            (
                box1.right < box2.left,
                box2.right < box1.left,
                box1.bottom < box2.top,
                box2.bottom < box1.top,
            )
        ):
            return None

        return PixelBox.left_top_right_bottom(
            left=max(box1.left, box2.left),
            top=max(box1.top, box2.top),
            right=min(box1.right, box2.right),
            bottom=min(box1.bottom, box2.bottom),
        )


def compute_effective_area(pixel_boxes: List[PixelBox]):
    """
    Computes the effective area covered by a list of (possibly overlapping) PixelBoxes, i.e.,
    so that areas covered by > 1 PixelBox are only counted once.
    """
    if len(pixel_boxes) == 0:
        return 0
    # Convert each tile's coordinates to a Shapely Rectangle
    shapely_boxes = [box(pb.left, pb.top, pb.right, pb.bottom) for pb in pixel_boxes]
    return union_all(shapely_boxes).area
