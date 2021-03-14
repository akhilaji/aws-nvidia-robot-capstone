"""
https://github.com/intel-isl/MiDaS/
@article{Ranftl2020,
	author    = {Ren\'{e} Ranftl and Katrin Lasinger and David Hafner and Konrad Schindler and Vladlen Koltun},
	title     = {Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer},
	journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
	year      = {2020},
}
"""

import numpy as np
import torch

from typing import Any
from nptyping import NDArray

class DepthEstimator:
    """
    Callable wrapper for different depth estimation techniques.
    """

    def __call__(self,
            np_rgb_img: NDArray[(Any, Any, 3), np.uint8]
        ) -> NDArray[(Any, Any), np.float]:
        """
        Estimates a depth map for a numpy 3-channel RGB image.
        """
        pass

class MidasDepthEstimator(DepthEstimator):
    """
    Callable wrapper for depth estimation using MiDaS loaded from torch.hub.

    Attributes:
        model: The model to be used. Assumed to be already be moved to device.
            Designed to be obtained with torch.hub.

        transforms (callable): Image transformations to process images before
            being passed fed to the model.

        device (torch.device): The device to run the model on. Since the model
            is already assumed to be on this device before construction, is
            only used as the destination to forward input batches to.
    
    https://github.com/intel-isl/MiDaS/
    @article{Ranftl2020,
        author    = {Ren\'{e} Ranftl and Katrin Lasinger and David Hafner and Konrad Schindler and Vladlen Koltun},
        title     = {Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer},
        journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
        year      = {2020},
    }
    """
    def __init__(self, model, transforms, device) -> None:
        self.model = model
        self.transforms = transforms
        self.device = device

    def __call__(self, np_rgb_img: NDArray[(Any, Any, 3), np.uint8]) -> NDArray[(Any, Any), np.float]:
        """
        Forwards a numpy RGB image to MiDaS and returns the inverse depth map
        estimated by the model.
        """
        input_batch = self.transforms(np_rgb_img).to(self.device)
        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=np_rgb_img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        return prediction.cpu().numpy()

def construct_midas_large(device: torch.device = torch.device('cuda'), verbose: bool = False) -> MidasDepthEstimator:
    """
    Factory function for creating a new instance of the DepthEstimator wrapper
    class for MiDaS_small loaded from torch.hub.
    """
    transforms = torch.hub.load('intel-isl/MiDaS', 'transforms', verbose=verbose).default_transform
    model = torch.hub.load('intel-isl/MiDaS', 'MiDaS', verbose=verbose)
    model.to(device)
    model.eval()
    return MidasDepthEstimator(model, transforms, device)

def construct_midas_small(device: torch.device = torch.device('cuda'),  verbose: bool = False) -> MidasDepthEstimator:
    """
    Factory function for creating a new instance of the DepthEstimator wrapper
    class for MiDaS loaded from torch.hub.
    """
    transforms = torch.hub.load('intel-isl/MiDaS', 'transforms', verbose=verbose).small_transform
    model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', verbose=verbose)
    model.to(device)
    model.eval()
    return MidasDepthEstimator(model, transforms, device)
