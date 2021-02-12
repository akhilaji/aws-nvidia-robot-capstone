"""
This module exists as a wrapper for obtaining depth information from RGB
images.

Requirements Installation:
pip install torch===1.7.1 torchvision===0.8.2 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

Example:
    testing_depth_script.py:
        import cv2
        import depth

        filename = 'test_file.jpg'
        rgb_img = cv2.imread(filename)
        depth_map = depth.midas_large(rgb_img)

Attributes:
    midas_device (torch.device): The torch.device to be used for applications of MiDaS.

    midas_transforms (???): The image transforms to be used for transforming
        images before giving them to the MiDaS model. Loaded from torch using
        torch.hub.load('intel-isl/MiDaS', 'transforms').
    
    midas_transforms_large (???): The image transforms for use with the default
        MiDaS model. Equivalent to midas_transforms.default_transform.
    
    midas_model_large (???): The default MiDaS model. Loaded from torch using
        torch.hub.load('intel-isl/MiDaS', 'MiDaS').
    
    midas_model_small (???): The small MiDaS model. Loaded from torch using
        torch.hub.load('intel-isl/MiDaS', 'MiDaS_small').
"""

import torch
import numpy as np
from nptyping import NDArray
from typing import Any

midas_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')

midas_transforms_large = midas_transforms.default_transform
midas_model_large = torch.hub.load('intel-isl/MiDaS', 'MiDaS')
midas_model_large.to(midas_device)
midas_model_large.eval()

midas_transforms_small = midas_transforms.small_transform
midas_model_small = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas_model_small.to(midas_device)
midas_model_small.eval()

def midas(
        rgb_img: NDArray[(Any, Any, 3), np.uint8],
        model, transforms
    ) -> NDArray[(Any, Any), np.float32]:
    """
    Gets a single frame estimated depth map by applying the provided image
    transforms and MiDaS model instance to an RGB image.

    Args:
        rgb_img (NDArray[(Any, Any, 3), np.uint8]): The RGB image to apply the
            MiDaS model to.
        
        model (???): The model loaded using torch.hub.load. Preloaded into
            this module so should be either midas_model_large or
            midas_model_small.

        transforms (???): The transforms as obtained by torch.hub.load.
            Already loaded into this module so use either of the global
            midas_transforms_large or midas_transforms_large declared
            in this module depending on which variant of the MiDaS model
            you want to use.
    
    Returns:
        NDArray[(Any, Any), np.float32]: The inverse depth map estimated by the
            MiDaS model for the input RGB image.
    """
    input_batch = transforms(rgb_img).to(midas_device)
    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=rgb_img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    return prediction.cpu().numpy()

def midas_large(
        rgb_img: NDArray[(Any, Any, 3), np.uint8]
    ) -> NDArray[(Any, Any), np.float32]:
    """
    Performs a single frame depth map estimation by feeding an RGB image to the
    default MiDaS model and returns the result.

    Args:
        rgb_img (NDArray[(Any, Any, 3), np.uint8]): The RGB image to feed to
            MiDaS.

    Returns:
        NDArray[(Any, Any), np.float32]: The inverse depth map estimated by
            MiDaS.
    """
    return midas(rgb_img, midas_model_large, midas_transforms_large)

def midas_small(
        rgb_img: NDArray[(Any, Any, 3), np.uint8]
    ) -> NDArray[(Any, Any), np.float32]:
    """
    Performs a single frame depth map estimation by feeding an RGB image to the
    small MiDaS model and returns the result.

    Args:
        rgb_img (NDArray[(Any, Any, 3), np.uint8]): The RGB image to feed to
            MiDaS.

    Returns:
        NDArray[(Any, Any), np.float32]: The inverse depth map estimated by
            MiDaS.
    """
    return midas(rgb_img, midas_model_small, midas_transforms_small)