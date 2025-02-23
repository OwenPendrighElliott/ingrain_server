from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import Compose, Resize
from PIL import Image

from typing import List


def convert_interpolation(interp: InterpolationMode) -> int:
    mapping = {
        InterpolationMode.NEAREST: Image.Resampling.NEAREST,
        InterpolationMode.NEAREST_EXACT: Image.Resampling.NEAREST,
        InterpolationMode.BILINEAR: Image.Resampling.BILINEAR,
        InterpolationMode.BICUBIC: Image.Resampling.BICUBIC,
        InterpolationMode.BOX: Image.Resampling.BOX,
        InterpolationMode.HAMMING: Image.Resampling.HAMMING,
        InterpolationMode.LANCZOS: Image.Resampling.LANCZOS,
    }
    return mapping[interp]


def image_transform_dict_from_torch_transforms(transforms: Compose) -> List[dict]:
    transform_dict = []
    for transform in transforms.transforms:
        if isinstance(transform, Resize):
            size = transform.size
            if isinstance(size, int):
                size = (size, size)
            transform_data = {
                "type": "ResizeImage",
                "size": size,
                "method": convert_interpolation(transform.interpolation),
            }
            transform_dict.append(transform_data)
        elif hasattr(transform, "__name__") and transform.__name__ == "_convert_to_rgb":
            transform_data = {"type": "ConvertToRGB"}
            transform_dict.append(transform_data)

    return transform_dict
