from skeleton import depth
from skeleton import graph
from skeleton import reconstruction
from skeleton import visualization

import os
import sys

import argparse
import cv2
import open3d

def main(args: argparse.Namespace) -> None:

    filename = ''

    midas_model = depth.construct_midas_large()
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inv_depth_map = midas_model(img)
    depth_map = 1.0 / inv_depth_map

    visualization.intensity_image(depth_map)
    visualization.intensity_image(inv_depth_map)

    height, width, _ = img.shape
    print(width,height)
    intr = open3d.camera.PinholeCameraIntrinsic(width=width, height=height, cx=width//2, cy=height//2, fx=1920.0, fy=1080.0)
    pt_cloud = reconstruction.construct_point_cloud_from_color_and_depth(img, depth_map, intr)
    open3d.visualization.draw_geometries([pt_cloud])
    
    return None

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    
    main(arg_parser.parse_args())

