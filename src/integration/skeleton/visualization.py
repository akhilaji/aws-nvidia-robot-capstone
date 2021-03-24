from skeleton.detect import BoundingBox, ObjectDetection

import cv2
import matplotlib.pyplot as plt
import numpy as np

from typing import Any
from nptyping import NDArray

def intensity_image(depth_map: NDArray[(Any, Any), float]):
    plt.figure()
    plt.imshow(depth_map)
    plt.colorbar(label='Distance to Camera')
    plt.title('Depth image')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    plt.show()

# Visualization Tools Needed:
#   predicted position of bounding box from object tracker
#   

def draw_detection(
        img: NDArray[(Any, Any, 3), np.uint8],
        det: ObjectDetection
    ) -> NDArray[(Any, Any, 3), np.uint8]:
    # TODO graphically display information from an ObjectDetection instance by
    # using functions like cv2.putText, cv2.rectangle, cv2.line, etc.
    pass

def draw_edge(
        img: NDArray[(Any, Any, 3), np.uint8],
        node1: ObjectDetection,
        node2: ObjectDetection
    ) -> NDArray[(Any, Any, 3), np.uint8]:
    weight = node2.pt - node1.pt
    # TODO draw edge and weight information on an RGB image by
    # using functions like cv2.putText, cv2.rectangle, cv2.line, etc.
    pass
