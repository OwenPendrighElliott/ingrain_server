from PIL import Image
import numpy as np
import json

from typing import List


class ImagePreprocessor:
    def __init__(self, steps: list, return_numpy: bool = True):
        self.steps = steps
        self.return_numpy = return_numpy

    def __call__(self, image: Image.Image) -> Image.Image:
        for step in self.steps:
            image = step(image)

        if self.return_numpy:
            return np.array(image)
        return image


class ImageTransformBase:
    def __call__(self, image: Image.Image) -> Image.Image:
        raise NotImplementedError

    @classmethod
    def from_dict(data: dict):
        raise NotImplementedError


class ResizeImage(ImageTransformBase):
    def __init__(self, size: tuple | list, method: int = Image.Resampling.BILINEAR):
        if len(size) != 2:
            raise ValueError("Size must be a tuple of 2 integers")
        self.size = size
        self.method = method

    def __call__(self, image: Image.Image) -> Image.Image:
        return image.resize(self.size, self.method)

    @classmethod
    def from_dict(data: dict):
        if "size" not in data:
            raise ValueError("Size is required for the resize transform")

        return ResizeImage(data["size"], data.get("method", Image.Resampling.BILINEAR))


class ConvertToRGB:
    def __call__(self, image: Image.Image) -> Image.Image:
        return image.convert("RGB")

    @classmethod
    def from_dict(_: dict):
        return ConvertToRGB()


def image_transform_from_dict(data: List[dict]) -> ImagePreprocessor:
    transforms = []

    for t_dict in data:
        match t_dict["type"]:
            case "ResizeImage":
                transform = ResizeImage.from_dict(t_dict)
            case "ConvertToRGB":
                transform = ConvertToRGB.from_dict(t_dict)
            case _:
                raise ValueError(f"Unknown transform type: {t_dict['type']}")
        transforms.append(transform)

    return ImagePreprocessor(transforms)


def load_image_transform_config(preprocess_config_path: str) -> ImagePreprocessor:
    with open(preprocess_config_path, "r") as f:
        preprocess_config = json.load(f)
    return image_transform_from_dict(preprocess_config)
