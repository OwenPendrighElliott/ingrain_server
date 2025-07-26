from PIL import Image, ImageOps
import numpy as np
import json

from typing import List


class ImageTransformBase:
    def __call__(self, image: Image.Image) -> Image.Image:
        raise NotImplementedError


class ParameterisedImageTransformBase(ImageTransformBase):
    @classmethod
    def from_dict(cls, data: dict):
        raise NotImplementedError


class NonParameterisedImageTransformBase(ImageTransformBase):
    @classmethod
    def from_dict(cls):
        raise NotImplementedError


class ImagePreprocessor:
    def __init__(
        self,
        steps: list[ImageTransformBase],
    ):
        self.steps = steps

    def __call__(self, image: Image.Image) -> np.ndarray:
        for step in self.steps:
            image = step(image)

        image = np.array(image, dtype=np.float32)
        image = image / 255.0
        image = np.moveaxis(image, -1, 0)
        return np.array(image)


class ResizeImage(ParameterisedImageTransformBase):
    def __init__(self, size: tuple | list, method: int = 2):
        if len(size) != 2:
            raise ValueError("Size must be a tuple of 2 integers")
        self.size = size
        self.method = method

    def __call__(self, image: Image.Image) -> Image.Image:
        return image.resize(self.size, resample=self.method)

    @classmethod
    def from_dict(cls, data: dict):
        if "size" not in data:
            raise ValueError("Size is required for the resize transform")

        return ResizeImage(data["size"], data.get("method", Image.Resampling.BILINEAR))


class CenterCropImage(ParameterisedImageTransformBase):
    def __init__(self, size: tuple | list):
        if len(size) != 2:
            raise ValueError("Size must be a tuple of 2 integers")
        self.size = (int(size[0]), int(size[1]))

    def __call__(self, image: Image.Image) -> Image.Image:
        crop_height, crop_width = self.size

        image_width, image_height = image.size

        if crop_width > image_width or crop_height > image_height:
            pad_left = max((crop_width - image_width) // 2, 0)
            pad_top = max((crop_height - image_height) // 2, 0)
            pad_right = max((crop_width - image_width + 1) // 2, 0)
            pad_bottom = max((crop_height - image_height + 1) // 2, 0)
            image = ImageOps.expand(
                image, border=(pad_left, pad_top, pad_right, pad_bottom), fill=0
            )
            image_width, image_height = image.size
            if image_width == crop_width and image_height == crop_height:
                return image

        left = int(round((image_width - crop_width) / 2.0))
        top = int(round((image_height - crop_height) / 2.0))

        return image.crop((left, top, left + crop_width, top + crop_height))

    @classmethod
    def from_dict(cls, data: dict):
        if "size" not in data:
            raise ValueError("Size is required for the center crop transform")
        return CenterCropImage(data["size"])


class ConvertToRGB(NonParameterisedImageTransformBase):
    def __call__(self, image: Image.Image) -> Image.Image:
        return image.convert("RGB")

    @classmethod
    def from_dict(cls):
        return ConvertToRGB()


def image_transform_from_dict(data: List[dict]) -> ImagePreprocessor:
    transforms = []

    for t_dict in data:
        match t_dict["type"]:
            case "ResizeImage":
                transform = ResizeImage.from_dict(t_dict)
            case "CenterCropImage":
                transform = CenterCropImage.from_dict(t_dict)
            case "ConvertToRGB":
                transform = ConvertToRGB.from_dict()
            case _:
                raise ValueError(f"Unknown transform type: {t_dict['type']}")
        transforms.append(transform)

    return ImagePreprocessor(transforms)


def load_image_transform_config(preprocess_config_path: str) -> ImagePreprocessor:
    with open(preprocess_config_path, "r") as f:
        preprocess_config = json.load(f)
    return image_transform_from_dict(preprocess_config)
