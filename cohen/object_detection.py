"""
TODO add a wrapper module for other modules to easily call a function and
retrieve the results from object detection
"""
from typing import NamedTuple

class BoundingBox(NamedTuple):
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0

