import open3d as o3d
import time
import numpy as np
import matplotlib.pyplot as plt

pt_cloud = o3d.io.read_point_cloud('mott/rapture.ply')  # load point cloud 

o3d.visualization.draw_geometries([pt_cloud])  

downpcd = pt_cloud.voxel_down_sample(voxel_size=0.05)  # down sample image

tic = time.time()
keypoints = o3d.geometry.keypoint.compute_iss_keypoints(downpcd)  # find key points
toc = 1000 * (time.time() - tic)
print("ISS Computation took {:.0f} [ms]".format(toc))

# key point visualization 
downpcd.paint_uniform_color([0.5, 0.5, 0.5])
keypoints.paint_uniform_color([1.0, 0.75, 0.0])
o3d.visualization.draw_geometries([downpcd, keypoints])  


keypoints.estimate_normals(  # create normal estimation of keypoints 
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# visualize keypoint normals 
# o3d.visualization.draw_geometries([keypoints],
#                                   zoom=0.3412,
#                                   front=[0.4257, -0.2125, -0.8795],
#                                   lookat=[2.6172, 2.0475, 1.532],
#                                   up=[-0.0694, -0.9768, 0.2024],
#                                   point_show_normal=True)


plane_model, inliers = downpcd.segment_plane(distance_threshold=0.01,  # perform RANSAC
                                         ransac_n=3,
                                         num_iterations=1000)
[a, b, c, d] = plane_model
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

inlier_cloud = downpcd.select_by_index(inliers)
inlier_cloud.paint_uniform_color([1.0, 0, 0])
outlier_cloud = downpcd.select_by_index(inliers, invert=True)
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                  zoom=0.8,
                                  front=[-0.4999, -0.1659, -0.8499],
                                  lookat=[2.1813, 2.0619, 2.0999],
                                  up=[0.1204, -0.9852, 0.1215])
