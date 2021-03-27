from skeleton.detect import BoundingBox, ObjectDetection

from core import utils
from core.config import cfg

from cv2 import cv2
import matplotlib.pyplot as plt
import numpy as np

from typing import Any, Tuple
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
        det: ObjectDetection,
        color: NDArray[3, np.uint8],
        font_face: int,
        font_scale: float,
        thickness: int = 1,
        line_type: int = cv2.LINE_8,
    ) -> None:
    # TODO graphically display information from an ObjectDetection instance by
    # using functions like cv2.putText, cv2.rectangle, cv2.line, etc.
    text = 'id=%r, class=%s, prob=%r' % (det.id, (list(utils.read_class_names(cfg.YOLO.CLASSES).values()))[det.obj_class], det.prob)
    draw_bbox(img, det.bbox, color, thickness, line_type)
    draw_text(img, text, (det.bbox.x, det.bbox.y), font_face, font_scale, color, thickness, line_type)

def draw_edge(
        img: NDArray[(Any, Any, 3), np.uint8],
        node1: ObjectDetection,
        node2: ObjectDetection
    ) -> NDArray[(Any, Any, 3), np.uint8]:
    weight = node2.pt - node1.pt
    # TODO draw edge and weight information on an RGB image by
    # using functions like cv2.putText, cv2.rectangle, cv2.line, etc.
    pass

def draw_bbox(
        img: NDArray[(Any, Any, 3), np.uint8],
        bbox: BoundingBox,
        color: NDArray[3, np.uint8],
        thickness: int = 1,
        line_type: int = cv2.LINE_8,
    ) -> None:
    cv2.rectangle(
        img,
        (bbox.x, bbox.y),
        (bbox.w, bbox.h),
        color,
        thickness=thickness,
        lineType=line_type,
    )

def draw_text(
        img: NDArray[(Any, Any, 3), np.uint8],
        text: str,
        pos: Tuple[int, int],
        font_face: int,
        font_scale: float,
        color: NDArray[3, np.uint8],
        thickness: int = 1,
        line_type: int = cv2.LINE_8,
        bottom_left_orgin: bool = False,
    ) -> None:
    cv2.putText(
        img=img,
        text=text,
        org=pos,
        fontFace=font_face,
        fontScale=font_scale,
        color=color,
        thickness=thickness,
        lineType=line_type,
        bottomLeftOrigin=bottom_left_orgin,
    )
