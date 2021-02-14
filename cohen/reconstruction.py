"""

"""

from typing import Any

import open3d
import numpy as np
from nptyping import NDArray

def project_point(
        u: int, v: int, d: np.float32,
        fx: np.float32, fy: np.float32,
        cx: int, cy: int
    ) -> NDArray[3, np.float32]:
    """
    
    """
    xz = (cx - u) / fx
    yz = (cy - v) / fy
    z = d / np.sqrt(1.0 + xz**2 + yz**2)
    return np.array([xz * z, yz * z, z], dtype=np.float32)

def np_to_o3d_rgb_image(
        np_rgb_image: NDArray[(Any, Any, 3), np.uint8],
    ) -> open3d.geometry.Image:
    return open3d.geometry.Image((np_rgb_image / 255.0).astype(np.float32))

def np_to_o3d_depth_map(
        np_depth_map: NDArray[(Any, Any), np.float32]
    ) -> open3d.geometry.Image:
    return open3d.geometry.Image(np_depth_map.astype(np.float32))

def construct_point_cloud_from_color_and_depth(
        np_rgb_image: NDArray[(Any, Any, 3), np.uint8],
        np_depth_map: NDArray[(Any, Any), np.float32],
        camera_intrinsics: open3d.camera.PinholeCameraIntrinsic,
        depth_scale: float=1000.0, depth_trunc: float=1000.0,
        project_valid_depth_only: bool=False
    ) -> open3d.geometry.PointCloud:
    """

    """
    o3d_rgb_image = np_to_o3d_rgb_image(np_rgb_image)
    o3d_depth_map = np_to_o3d_depth_map(np_depth_map)
    pt_cloud = open3d.geometry.PointCloud.create_from_rgbd_image(
        open3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_rgb_image,
            o3d_depth_map,
            depth_scale=depth_scale,
            depth_trunc=depth_trunc
        ),
        camera_intrinsics,
        project_valid_depth_only=project_valid_depth_only
    )
    o3d_rgb_array = np.asarray(o3d_rgb_image)
    pt_cloud.colors = open3d.utility.Vector3dVector(
        o3d_rgb_array.reshape(o3d_rgb_array.size // 3, 3).astype(np.float64)
    )
    return pt_cloud
