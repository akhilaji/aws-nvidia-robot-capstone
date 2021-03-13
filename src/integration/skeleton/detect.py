"""

"""

import numpy as np

from typing import Any, List, NamedTuple, Tuple
from nptyping import NDArray

class BoundingBox(NamedTuple):
    """
    NamedTuple class definition for a bounding box inside of an image.

    Attributes:
        x (int): the x-coordinate position of the bounding box
        y (int): the y-coordinate position of the bounding box
        w (int): the width of the bounding box
        h (int); the height of the bounding box
    """
    x: int
    y: int
    w: int
    h: int

class ObjectDetection(NamedTuple):
    """
    NamedTuple class definition for detected object instances.

    Attributes:
        bbox (BoundingBox): NamedTuple class definition for bounding box
            information. Contains attributes x, y, w, h in that order.
        
        obj_class (str): The assigned object classification of this detection.

        id (Any): The assigned id of this detection.

        pt (NDArray[3, float]): 
    """
    bbox: BoundingBox
    obj_class: str
    id: Any
    pt: NDArray[3, float]

class ObjectDetector:
    """
    Abstract callable object to run object detection on an RGB image.
    Impelementation details to be specified by child.
    """

    def __call__(self, np_rgb_img: NDArray[(Any, Any, 3), np.uint8]) -> List[ObjectDetection]:
        """
        Runs object detection on a numpy 3-channel RGB image and formats the
        results as a list of ObjectDection containing the BoundingBox information
        and the detected object class.
        """
        pass

class ObjectTracker:
    """

    """

    def track(self, detections: List[ObjectDetection]) -> None:
        """

        """
        pass

class PointPicker:
    """

    """

    def pick(self, detections: List[ObjectDetection]) -> None:
        """

        """
        pass
