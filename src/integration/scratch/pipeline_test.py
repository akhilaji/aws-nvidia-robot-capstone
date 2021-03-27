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
from skeleton import depth
from skeleton import detect
from skeleton import graph
from skeleton import reconstruction
from skeleton import visualization

from absl import app, flags, logging
from absl.flags import FLAGS


def get_depth(MidasEstimator, frame):
    print("depth function")
    depth_map = MidasEstimator(frame)
    
    return depth_map

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

    MidasEstimator = depth.construct_midas_large()
    Detector = detect.load_object_detector()
    
    
    while video.isOpened():
        frame_time = datetime.datetime.now()
        success, frame = video.read()
        if not success:
            break
        frame = cv2.resize(frame, dim)

        executor = ThreadPoolExecutor(max_workers=2)
        depth_value = executor.submit(get_depth(MidasEstimator,frame))
        objects = executor.submit(Detector(frame))

        #Calculate the depth map
        #depth_value = get_depth(MidasEstimator,frame)
        #depth_map = 1.0/depth_value
        #depth_visualization(depth_map, depth_value, frame)
        objects = Detector(frame)

        #print(depth_value)
        print(objects)
        print(frame_time)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
