import torch
from timm.data import MaybeToTensor, MaybePILToTensor
from torch import nn
from torchvision.transforms import Compose, ToTensor


class TimmClassifierWrapper(nn.Module):
    def __init__(self, visual: nn.Module, transforms: Compose):
        """
        A wrapper that encapsulates the image encoder part of the CLIP model.
        """
        super().__init__()

        self.visual = visual

        to_tensor_index = next(
            i
            for i, t in enumerate(transforms.transforms)
            if isinstance(t, (ToTensor, MaybeToTensor, MaybePILToTensor))
        )

        self.tensor_transforms = Compose(
            [t for t in transforms.transforms[to_tensor_index + 1 :]]
        )

    def forward(self, image):
        """
        Forward pass to encode image.
        """

        x = self.tensor_transforms(image)
        print("SHAPESHAPE:", x.shape)
        x = self.visual(x)
        return x
