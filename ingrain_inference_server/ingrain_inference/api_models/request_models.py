from ingrain_inference.api_models.camel_model import CamelModel

from typing import List, Optional, Union


class EmbeddingRequest(CamelModel):
    name: str
    pretrained: Optional[str] = None
    text: Optional[Union[str, List[str]]] = None
    image: Optional[Union[str, List[str]]] = None
    normalize: Optional[bool] = True
    n_dims: Optional[int] = None
    image_download_headers: Optional[dict] = None


class TextEmbeddingRequest(CamelModel):
    name: str
    pretrained: Optional[str] = None
    text: Union[str, List[str]]
    normalize: Optional[bool] = True
    n_dims: Optional[int] = None


class ImageEmbeddingRequest(CamelModel):
    name: str
    pretrained: Optional[str] = None
    image: Union[str, List[str]]
    normalize: Optional[bool] = True
    n_dims: Optional[int] = None
    image_download_headers: Optional[dict] = None


class ImageClassificationRequest(CamelModel):
    name: str
    pretrained: Optional[str] = None
    image: Union[str, List[str]]
    image_download_headers: Optional[dict] = None
