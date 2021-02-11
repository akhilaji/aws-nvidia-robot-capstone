"""

"""

import numpy
import open3d

def cv2_to_o3d_rgb_image(cv2_rgb_image):
    return open3d.geometry.Image((cv2_rgb_image / 255.0).astype(numpy.float32))

def midas_to_o3d_depth_image(midas_depth_map):
    return open3d.geometry.Image(midas_depth_map.astype(numpy.float32))

def from_color_and_depth(
    cv2_rgb_image: numpy.array,
    midas_depth_map: numpy.array,
    camera_intrinsics: open3d.camera.PinholeCameraIntrinsic,
    depth_scale=1000.0, depth_trunc=1000.0,
    project_valid_depth_only=False):
    """

    """
    o3d_rgb_image = cv2_to_o3d_rgb_image(cv2_rgb_image)
    o3d_depth_image = midas_to_o3d_depth_image(midas_depth_map)
    pt_cloud = open3d.geometry.PointCloud.create_from_rgbd_image(
        open3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_rgb_image,
            o3d_depth_image,
            depth_scale=depth_scale,
            depth_trunc=depth_trunc
        ),
        camera_intrinsics,
        project_valid_depth_only=project_valid_depth_only
    )
    o3d_rgb_array = numpy.asarray(o3d_rgb_image)
    pt_cloud.colors = open3d.utility.Vector3dVector(
        o3d_rgb_array.reshape(o3d_rgb_array.size // 3, 3).astype(numpy.float64)
    )
    return pt_cloud
