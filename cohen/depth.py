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

def midas(img, model, transforms):
    """
    Gets a single frame estimated depth map by applying the provided image
    transforms and MiDaS model instance to an RGB image.

    Args:
        img (numpy.array): The RGB image to apply the model to. The RGB image
            should be a numpy.array as obtained by reading an image file with
            cv2.imread(filename).
        
        model (???): The model loaded using torch.hub.load. Preloaded into
            this module so should be either midas_model_large or
            midas_model_small.

        transforms (???): The transforms as obtained by torch.hub.load.
            Already loaded into this module so use either of the global
            midas_transforms_large or midas_transforms_large declared
            in this module depending on which variant of the MiDaS model
            you want to use.
    
    Returns:
        numpy.array: The depth map estimated by model as a numpy.array of
            floats.
    """
    input_batch = transforms(img).to(midas_device)
    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    return prediction.cpu().numpy()

def midas_large(img):
    """
    Gets a single frame depth map estimation by applying the large version of
    the MiDaS model to an RGB image.

    Args:
        img (numpy.array): An RGB image as obtained by reading an image with
            cv2.imread(filename).

    Returns:
        numpy.array: The depth map estimated by MiDaS as a numpy.array of
            floats.
    """
    return midas(img, midas_model_large, midas_transforms_large)

def midas_small(img):
    """
    Gets a single frame depth map estimation by applying the small version of
    the MiDaS model to an RGB image.

    Args:
        img (numpy.array): An RGB image as obtained by reading an image with
            cv2.imread(filename).

    Returns:
        numpy.array: The depth map estimated by MiDaS_small as a numpy.array of
            floats.
    """
    return midas(img, midas_model_small, midas_transforms_small)