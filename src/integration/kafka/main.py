import os
import sys
import datetime
import time
import cv2
import argparse
import open3d
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.config import cfg
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import depth
from skeleton import depth
from skeleton import graph
from skeleton import reconstruction
from skeleton import visualization

from skeleton.detect import BoundingBox
from skeleton.detect import ObjectDetection

flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_list('images', './data/images/kite.jpg', 'path to input image')
flags.DEFINE_string('output', './detections/', 'path to output folder')

# load model
model_path = "./yolov4-608"
saved_model_loaded = tf.saved_model.load(
    model_path, tags=[tag_constants.SERVING])


def get_depth(MidasEstimator, frame):
    print("depth function")
    depth_map = MidasEstimator(frame)
    return depth_map

def get_object(_argv, data, config, session, input_size, images):
    print("object detection function")
    # detection_properties
    iou = 0.45
    score = 0.25

    original_image = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

    image_data = data
    image_data = image_data / 255
    images_data = []
    for i in range(1):
        images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)

    infer = saved_model_loaded.signatures['serving_default']
    batch_data = tf.constant(images_data)
    pred_bbox = infer(batch_data)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=iou,
        score_threshold=score
    )

    # read in all class names from config
    class_names = utils.read_class_names(cfg.YOLO.CLASSES)

    # by default allow all classes in .names file
    allowed_classes = list(class_names.values())

    detections = []
    for i in range(valid_detections[0]):
        bbox = BoundingBox(boxes[0][i][0],
                           boxes[0][i][1],
                           boxes[0][i][2],
                           boxes[0][i][3])

        obj = ObjectDetection(id=-1,
                              bbox=bbox,
                              obj_class=classes[0][i],
                              prob=scores[0][i],
                              pt=[-1,-1,-1])

        detections.append(obj)

    return detections

def depth_visualization(depth_map, inv_depth_map, frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    visualization.intensity_image(depth_map)
    visualization.intensity_image(inv_depth_map)

    height, width, _ = img.shape
    print(width,height)
    intr = open3d.camera.PinholeCameraIntrinsic(width=width, height=height, cx=width//2, cy=height//2, fx=1920.0, fy=1080.0)
    pt_cloud = reconstruction.construct_point_cloud_from_color_and_depth(img, depth_map, intr)
    open3d.visualization.draw_geometries([pt_cloud])

def main(_argv):
    print("initializing services")
    # Video Settings
    # set 0 for camera
    video_src = 0
    resolution_width = 608
    resolution_height = 608
    dim = (resolution_width, resolution_height)

    video = cv2.VideoCapture(video_src)
    video.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    #tensorflow loading
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = 608
    images = FLAGS.images

    MidasEstimator = depth.construct_midas_large()
    
    while video.isOpened():
        frame_time = datetime.datetime.now()
        success, frame = video.read()
        if not success:
            break
        frame = cv2.resize(frame, dim)

        executor = ThreadPoolExecutor(max_workers=2)
        depth_value = executor.submit(get_depth(MidasEstimator,frame))
        objects = executor.submit(get_object(_argv, frame, config, session, input_size, images))

        #Calculate the depth map
        #depth_value = get_depth(MidasEstimator,frame)
        #depth_map = 1.0/depth_value
        #depth_visualization(depth_map, depth_value, frame)
        
        #objects = get_object(_argv, frame, config, session, input_size, images)

        #print(depth_value)
        #print(objects)


        print(frame_time)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
