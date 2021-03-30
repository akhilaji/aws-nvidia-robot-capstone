"""
@article{Zhou2018,
   author  = {Qian-Yi Zhou and Jaesik Park and Vladlen Koltun},
   title   = {{Open3D}: {A} Modern Library for {3D} Data Processing},
   journal = {arXiv:1801.09847},
   year    = {2018},
}
"""

from skeleton.detect import ID, ObjectDetection
from skeleton.graph import Graph, all_simple_path_costs

import numpy as np
import open3d

from typing import Any, Dict, List
from nptyping import NDArray

class SceneReconstructor:
    """

    """

    def __init__(self, G: Graph, obj_class: Dict[ID, int]):
        self.G = G
        self.obj_class = obj_class

    def add_edge(self, src: ObjectDetection, dst: ObjectDetection) -> None:
        """

        """
        # TODO add filter
        weight = dst.pt - src.pt
        self.G[src.id][dst.id] = weight

    def add_frame(self, detections: List[ObjectDetection]) -> None:
        """

        """
        for i in range(len(detections)):
            self.obj_class[detections[i].id] = detections[i].obj_class
            for j in range(i + 1, len(detections)):
                self.add_edge(detections[i], detections[j])
                self.add_edge(detections[j], detections[i])

    def finalize(self):
        """

        """
        avg = lambda x: sum(x) / len(x)
        initial_cost = np.array([0, 0, 0], np.float32)
        node_itr = iter(self.G.E.keys())
        anchor = next(node_itr)
        pt_map = {node: avg(all_simple_path_costs(self.G, anchor, node, initial_cost)) for node in node_itr}
        pt_map[anchor] = initial_cost        

def np_to_o3d_rgb_image(
        np_rgb_image: NDArray[(Any, Any, 3), np.uint8]
    ) -> open3d.geometry.Image:
    return open3d.geometry.Image(np_rgb_image)

def np_to_o3d_depth_map(
        np_depth_map: NDArray[(Any, Any), np.float32]
    ) -> open3d.geometry.Image:
    return open3d.geometry.Image(np_depth_map.astype(np.float32))

def construct_point_cloud_from_color_and_depth(
        np_rgb_image: NDArray[(Any, Any, 3), np.uint8],
        np_depth_map: NDArray[(Any, Any), np.float32],
        camera_intrinsics: open3d.camera.PinholeCameraIntrinsic,
        depth_scale: float=1000.0, depth_trunc: float=1000.0,
        convert_rgb_to_intensity: bool=False,
        project_valid_depth_only: bool=True
    ) -> open3d.geometry.PointCloud:
    """

    """
    pt_cloud = open3d.geometry.PointCloud.create_from_rgbd_image(
        open3d.geometry.RGBDImage.create_from_color_and_depth(
            np_to_o3d_rgb_image(np_rgb_image),
            np_to_o3d_depth_map(np_depth_map),
            depth_scale=depth_scale,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=convert_rgb_to_intensity
        ),
        camera_intrinsics,
        project_valid_depth_only=project_valid_depth_only
    )
    return pt_cloud
