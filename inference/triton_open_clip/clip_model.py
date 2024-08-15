import os
import open_clip
import validators
import numpy as np
import base64
import certifi
import pycurl
from io import BytesIO
import tritonclient.grpc as grpcclient
from PIL import Image
from ..common import get_text_image_model_names
from ..model_client import TritonModelClient
from .clip_converting import (
    script_open_clip_model,
    generate_image_clip_config,
    generate_text_clip_config,
)

from typing import Tuple, Union, List


def create_transforms(model_name: str, pretrained: Union[str, None]):
    """Create the model and transforms.

    Args:
        model_name (str): The model name.
        pretrained (Union[str, None]): The pretrained checkpoint.

    Returns:
        Tuple: The text and image model names, preprocess function, and tokenizer.
    """
    _, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    friendly_text_name, friendly_image_name = get_text_image_model_names(
        model_name, pretrained
    )
    return friendly_text_name, friendly_image_name, preprocess, tokenizer


def create_model_and_transforms_triton(
    model_name: str,
    pretrained: Union[str, None],
    triton_model_repository_path: str,
):
    """Create the model and transforms for Triton.

    Args:
        model_name (str): The model name.
        pretrained (Union[str, None]): The pretrained checkpoint.
        triton_model_repository_path (str): The path to the Triton model repository.

    Returns:
        Tuple: The text and image model names, preprocess function, and tokenizer.
    """
    config = open_clip.get_model_config(model_name)
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    friendly_text_name, friendly_image_name = get_text_image_model_names(
        model_name, pretrained
    )
    os.makedirs(
        os.path.join(triton_model_repository_path, friendly_text_name, "1"),
        exist_ok=True,
    )
    os.makedirs(
        os.path.join(triton_model_repository_path, friendly_image_name, "1"),
        exist_ok=True,
    )

    image_encoder_exists = os.path.exists(
        os.path.join(triton_model_repository_path, friendly_image_name, "1", "model.pt")
    )
    text_encoder_exists = os.path.exists(
        os.path.join(triton_model_repository_path, friendly_text_name, "1", "model.pt")
    )
    image_config_exists = os.path.exists(
        os.path.join(triton_model_repository_path, friendly_image_name, "config.pbtxt")
    )
    text_config_exists = os.path.exists(
        os.path.join(triton_model_repository_path, friendly_text_name, "config.pbtxt")
    )

    # exit early if the models and configs already exist
    if (
        image_encoder_exists
        and text_encoder_exists
        and image_config_exists
        and text_config_exists
    ):
        return friendly_text_name, friendly_image_name, preprocess, tokenizer

    image_encoder, text_encoder = script_open_clip_model(model)

    text_encoder.save(
        os.path.join(triton_model_repository_path, friendly_text_name, "1", "model.pt")
    )
    image_encoder.save(
        os.path.join(triton_model_repository_path, friendly_image_name, "1", "model.pt")
    )

    generate_text_clip_config(
        os.path.join(triton_model_repository_path, friendly_text_name),
        friendly_text_name,
        tokenizer.context_length,
        config["embed_dim"],
    )

    generate_image_clip_config(
        os.path.join(triton_model_repository_path, friendly_image_name),
        friendly_image_name,
        (3, *model.visual.preprocess_cfg["size"]),
        config["embed_dim"],
    )
    return friendly_text_name, friendly_image_name, preprocess, tokenizer


from concurrent.futures import ThreadPoolExecutor


class TritonCLIPClient(TritonModelClient):
    def __init__(
        self,
        triton_grpc_url: str,
        model: str,
        pretrained: Union[str, None],
        triton_model_repository_path: str,
    ):
        super().__init__(triton_grpc_url)
        self.text_model_name, self.image_model_name = get_text_image_model_names(
            model, pretrained
        )

        if not self.triton_client.is_model_ready(
            self.text_model_name
        ) or not self.triton_client.is_model_ready(self.image_model_name):
            (
                self.text_model_name,
                self.image_model_name,
                self.preprocess,
                self.tokenizer,
            ) = create_model_and_transforms_triton(
                model, pretrained, triton_model_repository_path
            )
            self.triton_client.load_model(self.text_model_name)
            self.triton_client.load_model(self.image_model_name)
        else:
            (
                self.text_model_name,
                self.image_model_name,
                self.preprocess,
                self.tokenizer,
            ) = create_transforms(model, pretrained)

        self.modalities = {"text", "image"}

    def encode_text(
        self, text: Union[str, List[str]], normalize: bool = True
    ) -> np.ndarray:
        tokens = self.tokenizer(text).numpy()
        text_inputs = grpcclient.InferInput("input__0", tokens.shape, "INT64")
        text_inputs.set_data_from_numpy(tokens)
        outputs = self.triton_client.infer(
            model_name=self.text_model_name, inputs=[text_inputs]
        ).as_numpy("output__0")

        if normalize:
            outputs = outputs / np.linalg.norm(outputs, axis=-1, keepdims=True)

        return outputs

    def download_image(self, url: str) -> Image.Image:
        buffer = BytesIO()
        c = pycurl.Curl()
        c.setopt(c.URL, url)
        c.setopt(c.WRITEDATA, buffer)
        c.setopt(pycurl.CAINFO, certifi.where())
        c.perform()
        c.close()
        buffer.seek(0)
        return Image.open(buffer)

    def decode_base64_image(self, base64_image: str) -> Image.Image:
        if base64_image.startswith("data:image"):
            base64_image = base64_image.split(",")[1]
        # Decode the base64 string to bytes
        image_data = base64.b64decode(base64_image)
        buffer = BytesIO()
        buffer.write(image_data)
        buffer.seek(0)
        return Image.open(buffer)

    def load_image(self, image: str) -> Image.Image:
        image_data = None
        if validators.url(image):
            image_data = self.download_image(image)
        elif isinstance(image, str):
            image_data = self.decode_base64_image(image)

        return image_data

    def load_images_parallel(self, images: List[str]) -> List[Image.Image]:
        with ThreadPoolExecutor(max_workers=10) as executor:
            return list(executor.map(self.load_image, images))

    def encode_image(
        self, image: Union[Image.Image, List[Image.Image]], normalize: bool = True
    ) -> np.ndarray:

        if isinstance(image, Image.Image):
            image = [image]
        processed_images = np.stack([self.preprocess(image).numpy() for image in image])

        image_inputs = grpcclient.InferInput("input__0", processed_images.shape, "FP32")
        image_inputs.set_data_from_numpy(processed_images)
        outputs = self.triton_client.infer(
            model_name=self.image_model_name, inputs=[image_inputs]
        ).as_numpy("output__0")

        if normalize:
            outputs = outputs / np.linalg.norm(outputs, axis=-1, keepdims=True)

        return outputs

    def load(self):
        self.triton_client.load_model(self.text_model_name)
        self.triton_client.load_model(self.image_model_name)

    def unload(self):
        self.triton_client.unload_model(self.text_model_name)
        self.triton_client.unload_model(self.image_model_name)

    def is_ready(self) -> bool:
        return self.triton_client.is_model_ready(
            self.text_model_name
        ) and self.triton_client.is_model_ready(self.image_model_name)
