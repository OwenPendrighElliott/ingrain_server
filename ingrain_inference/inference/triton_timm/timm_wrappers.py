import torch
from torch import nn
from torchvision.transforms import Compose, ToTensor


class TimmClassifierWrapper(nn.Module):
    def __init__(self, visual: nn.Module, transforms: Compose):
        """
        A wrapper that encapsulates the image encoder part of the CLIP model.
        """
        super().__init__()

        self.visual = visual

        to_tensor_index = transforms.transforms.index(ToTensor)

        self.tensor_transforms = [
            t for t in transforms.transforms[to_tensor_index + 1 :]
        ]

    def forward(self, image):
        """
        Forward pass to encode image.
        """

        x = self.tensor_transforms(image)
        x = self.visual(x)
        return x
