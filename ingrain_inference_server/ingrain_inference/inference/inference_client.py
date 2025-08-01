import tritonclient.grpc as grpcclient
import numpy as np
from PIL import Image
import base64
import certifi
import pycurl
from io import BytesIO
import validators
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union, Optional


class TritonModelInferenceClient:
    def __init__(self, triton_grpc_url: str):
        self.triton_client = grpcclient.InferenceServerClient(
            url=triton_grpc_url, verbose=False
        )
        self.modalities = set()

    def encode_text(
        self,
        text: Union[str, List[str]],
        normalize: bool = True,
        n_dims: Optional[int] = None,
    ) -> np.ndarray:
        raise NotImplementedError

    def encode_image(
        self,
        image: Union[Image.Image, List[Image.Image]],
        normalize: bool = True,
        n_dims: Optional[int] = None,
    ) -> np.ndarray:
        raise NotImplementedError

    def unload(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def is_ready(self) -> bool:
        raise NotImplementedError

    def download_image(
        self, url: str, image_download_headers: Optional[dict] = None
    ) -> Image.Image:
        buffer = BytesIO()
        c = pycurl.Curl()
        c.setopt(c.URL, url)
        c.setopt(c.WRITEDATA, buffer)
        c.setopt(pycurl.CAINFO, certifi.where())
        if image_download_headers:
            c.setopt(
                c.HTTPHEADER, [f"{k}: {v}" for k, v in image_download_headers.items()]
            )
        else:
            c.setopt(c.HTTPHEADER, ["User-Agent: ingrain-server"])
        c.perform()
        c.close()
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    def decode_base64_image(self, base64_image: str) -> Image.Image:
        if base64_image.startswith("data:image"):
            base64_image = base64_image.split(",")[1]
        # Decode the base64 string to bytes
        image_data = base64.b64decode(base64_image)
        buffer = BytesIO()
        buffer.write(image_data)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    def load_image(
        self, image: str, image_download_headers: Optional[dict] = None
    ) -> Image.Image:
        image_data = None
        if validators.url(image):
            image_data = self.download_image(
                image, image_download_headers=image_download_headers
            )
        elif isinstance(image, str):
            image_data = self.decode_base64_image(image)

        return image_data

    def load_images_parallel(
        self, images: List[str], image_download_headers: Optional[dict] = None
    ) -> List[Image.Image]:
        def load_with_headers(image_url: str) -> Image.Image:
            return self.load_image(
                image_url, image_download_headers=image_download_headers
            )

        with ThreadPoolExecutor(max_workers=10) as executor:
            return list(executor.map(load_with_headers, images))
