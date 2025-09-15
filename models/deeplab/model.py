import torch
import torch.nn as nn
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights
)
from .dataset import NUM_CLASSES

def build_model(device=None):
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    model = deeplabv3_resnet50(weights=weights, progress=True)
    model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    return model.to(device) if device else model
