from typing import NamedTuple


class Location(NamedTuple):
    x: int
    y: int

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
