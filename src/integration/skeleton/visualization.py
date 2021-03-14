from skeleton import detect

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

def draw_detection(rgb_img, detection: detect.ObjectDetection, color, font):
    bbox, obj_class = detection
    cv2.rectangle(rgb_img, (bbox.x, bbox.y), (bbox.x + bbox.w, bbox.y + bbox.h), color)
    pass
