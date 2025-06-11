from dataclasses import dataclass

from typing import List, Tuple

from afb.utils.geometry import PixelBox

# more/less just typed np.arrays, might want to back with a np.array
# and keep consistent with np api


# TODO: migrate these classes to slidearm? Might be a natural fit there,
# and boxen might be able to make use of them then as well?
@dataclass
class Vec2D:
    x: float
    y: float

    def __lt__(self, other: "Vec2D") -> bool:
        return self.x < other.x and self.y < other.y

    def __le__(self, other: "Vec2D") -> bool:
        return self.x <= other.x and self.y <= other.y

    def __list__(self) -> List[float]:
        return [self.x, self.y]

    def __add__(self, other: "Vec2D") -> "Vec2D":
        return Vec2D(self.x + other.x, self.y + other.y)


@dataclass
class XYXYBox:
    x1: Vec2D
    x2: Vec2D

    def __post_init__(self):
        if self.x1.x > self.x2.x or self.x1.y > self.x2.y:
            raise ValueError(
                f"Invalid bounding box, x1 should be the minimum point and x2 should be the maximum: {self}"
            )

    def __contains__(self, other: "XYXYBox") -> bool:
        """Check if this box is entirely contained within the other box"""
        return (
            self.x1.x <= other.x1.x
            and self.x1.y <= other.x1.y
            and self.x2.x >= other.x2.x
            and self.x2.y >= other.x2.y
        )

    def overlaps(self, other: "XYXYBox") -> bool:
        """Check if this box overlaps with another box"""
        return not (
            self.x1.x >= other.x2.x
            or self.x2.x <= other.x1.x
            or self.x1.y >= other.x2.y
            or self.x2.y <= other.x1.y
        )

    def __lt__(self, other: "XYXYBox") -> bool:
        """Check if this box is entirely contained within the other box"""
        return (
            self.x1.x > other.x1.x
            and self.x1.y > other.x1.y
            and self.x2.x < other.x2.x
            and self.x2.y < other.x2.y
        )

    def __le__(self, other: "XYXYBox") -> bool:
        """Check if this box is contained within or on the border to the other box"""
        return (
            self.x1.x >= other.x1.x
            and self.x1.y >= other.x1.y
            and self.x2.x <= other.x2.x
            and self.x2.y <= other.x2.y
        )

    def __gt__(self, other: "XYXYBox") -> bool:
        """Check if this box entirely contains the other box"""
        return (
            other.x1.x > self.x1.x
            and other.x1.y > self.x1.y
            and other.x2.x < self.x2.x
            and other.x2.y < self.x2.y
        )

    def __ge__(self, other: "XYXYBox") -> bool:
        """Check if this box entirely contains or is on the border to the other box"""
        return (
            other.x1.x >= self.x1.x
            and other.x1.y >= self.x1.y
            and other.x2.x <= self.x2.x
            and other.x2.y <= self.x2.y
        )

    def intersects(self, other: "XYXYBox") -> bool:
        """Check if this box intersects with another box"""
        return not (
            self.x1.x >= other.x2.x
            or self.x2.x <= other.x1.x
            or self.x1.y >= other.x2.y
            or self.x2.y <= other.x1.y
        )

    def from_points(x1, y1, x2, y2) -> "XYXYBox":
        # MM: I think this was a typo before?
        return XYXYBox(Vec2D(x1, y1), Vec2D(x2, y2))

    def from_center_box(
        center_x: float, center_y: float, width: float, height: float
    ) -> "XYXYBox":
        """Create a new XYXYBox from a center_x, center_y, width, height box"""
        return XYXYBox(
            Vec2D(
                center_x - width / 2,
                center_y - height / 2,
            ),
            Vec2D(center_x + width / 2, center_y + height / 2),
        )

    def to_center_box(self) -> Tuple[float, float, float, float]:
        """Convert this XYXYBox to a center_x, center_y, width, height box"""
        return (
            (self.x1.x + self.x2.x) / 2,
            (self.x1.y + self.x2.y) / 2,
            self.x2.x - self.x1.x,
            self.x2.y - self.x1.y,
        )

    def from_normalized_center_box(
        center_x: float,
        center_y: float,
        width: float,
        height: float,
        img_size: Tuple[int, int],
    ) -> "XYXYBox":
        """Create a new XYXYBox from a normalized center_x, center_y, width, height box"""
        return XYXYBox(
            Vec2D(
                round((center_x - width / 2) * img_size[0]),
                round((center_y - height / 2) * img_size[1]),
            ),
            Vec2D(
                round((center_x + width / 2) * img_size[0]),
                round((center_y + height / 2) * img_size[1]),
            ),
        )

    def to_normalized_center_box(
        self, img_size: Tuple[int, int]
    ) -> Tuple[float, float, float, float]:
        """Convert this XYXYBox to a normalized center_x, center_y, width, height box"""
        return (
            (self.x1.x + self.x2.x) / 2 / img_size[0],
            (self.x1.y + self.x2.y) / 2 / img_size[1],
            (self.x2.x - self.x1.x) / img_size[0],
            (self.x2.y - self.x1.y) / img_size[1],
        )

    def from_pixel_box(pb: PixelBox) -> "XYXYBox":
        return XYXYBox(
            Vec2D(pb.left, pb.top), Vec2D(pb.left + pb.width, pb.top + pb.height)
        )

    def to_list(self) -> List[float]:
        return [self.x1.x, self.x1.y, self.x2.x, self.x2.y]

    def offset(self, vec: Vec2D) -> "XYXYBox":
        """Shifts this XYXYBox by the supplied vector."""
        return XYXYBox(self.x1 + vec, self.x2 + vec)

    def area(self) -> int:
        """Calculate the area of this box"""
        return (self.x2.x - self.x1.x) * (self.x2.y - self.x1.y)

    def normalize(self, scalar: float) -> "XYXYBox":
        """Divide this XYXYBox by a scalar"""
        return XYXYBox(
            Vec2D(self.x1.x / scalar, self.x1.y / scalar),
            Vec2D(self.x2.x / scalar, self.x2.y / scalar),
        )
