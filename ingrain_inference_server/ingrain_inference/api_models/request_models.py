from pydantic import BaseModel

from typing import List, Optional, Union


class InferenceRequest(BaseModel):
    name: str
    pretrained: Optional[str] = None
    text: Optional[Union[str, List[str]]] = None
    image: Optional[Union[str, List[str]]] = None
    normalize: Optional[bool] = True
    n_dims: Optional[int] = None
    image_download_headers: Optional[dict] = None


class TextInferenceRequest(BaseModel):
    name: str
    pretrained: Optional[str] = None
    text: Union[str, List[str]]
    normalize: Optional[bool] = True
    n_dims: Optional[int] = None


class ImageInferenceRequest(BaseModel):
    name: str
    pretrained: Optional[str] = None
    image: Union[str, List[str]]
    normalize: Optional[bool] = True
    n_dims: Optional[int] = None
    image_download_headers: Optional[dict] = None
