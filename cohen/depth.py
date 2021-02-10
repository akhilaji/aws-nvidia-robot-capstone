# requirements
# pip install torch===1.7.1 torchvision===0.8.2 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

import cv2
import torch
import matplotlib.pyplot as plt

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
    return midas(img, midas_model_large, midas_transforms_large)

def midas_small(img):
    return midas(img, midas_model_small, midas_transforms_small)