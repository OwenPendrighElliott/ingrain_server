from ingrain_inference.inference.preprocessors.image_preprocessor import (
    ImagePreprocessor,
    ResizeImage,
    ConvertToRGB,
    image_transform_from_dict,
)
from PIL import Image


def test_resize_image():
    image = Image.new("RGB", (100, 100), color="red")
    transform = ResizeImage((50, 50))
    resized_image = transform(image)
    assert resized_image.size == (50, 50)


def test_convert_to_rgb():
    image = Image.new("RGBA", (100, 100), color="red")
    transform = ConvertToRGB()
    rgb_image = transform(image)
    assert rgb_image.mode == "RGB"
    assert rgb_image.size == (100, 100)


def test_image_preprocessor():
    steps = [ConvertToRGB(), ResizeImage((50, 50))]

    preprocessor = ImagePreprocessor(steps)

    image = Image.new("RGBA", (100, 100), color="green")
    transformed_image_arr = preprocessor(image)

    transformed_image = Image.fromarray(
        (transformed_image_arr * 255).astype("uint8").transpose(1, 2, 0)
    )

    assert transformed_image.size == (50, 50)
    assert transformed_image.mode == "RGB"


def test_image_transform_from_dict():
    transform_data = [
        {"type": "ConvertToRGB"},
        {"type": "ResizeImage", "size": [50, 50]},
    ]

    preprocessor = image_transform_from_dict(transform_data)

    image = Image.new("RGBA", (100, 100), color="blue")
    transformed_image_arr = preprocessor(image)

    transformed_image = Image.fromarray(
        (transformed_image_arr * 255).astype("uint8").transpose(1, 2, 0)
    )

    assert transformed_image.size == (50, 50)
    assert transformed_image.mode == "RGB"
