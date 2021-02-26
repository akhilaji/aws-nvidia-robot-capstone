import sys
sys.path.append('cohen')

import depth
import reconstruction
import cv2
import open3d


def image_to_ptc(img_path, output_path):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    width, height, no_color_channels = img.shape
    depth_map = depth.midas_large(img)
    intr = open3d.camera.PinholeCameraIntrinsic(width=width, height=height, fx=888.888, fy=500.0, cx=width//2, cy=height//2)
    pt_cloud = reconstruction.construct_point_cloud_from_color_and_depth(img, depth_map, intr)
    open3d.visualization.draw_geometries([pt_cloud])
    open3d.io.write_point_cloud(output_path, pt_cloud)

image_to_ptc("mott/image0.jpeg", "mott/source.ply")
image_to_ptc("mott/image1.jpeg", "mott/target.ply")