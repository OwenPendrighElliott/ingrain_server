import tritonclient.grpc as grpcclient
import numpy as np
from PIL import Image
from typing import List, Union


class TritonModelClient:
    def __init__(self, triton_grpc_url: str):
        self.triton_client = grpcclient.InferenceServerClient(
            url=triton_grpc_url, verbose=False
        )
        self.modalities = set()

    def encode_text(
        self, text: Union[str, List[str]], normalize: bool = True
    ) -> np.ndarray:
        raise NotImplementedError

    def encode_image(
        self, image: Union[Image.Image, List[Image.Image]], normalize: bool = True
    ) -> np.ndarray:
        raise NotImplementedError

    def unload(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def is_ready(self) -> bool:
        raise NotImplementedError
