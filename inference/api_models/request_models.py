from pydantic import BaseModel

from typing import List, Optional, Union, Literal


class InferenceRequest(BaseModel):
    name: str
    pretrained: Optional[str] = None
    text: Optional[Union[str, List[str]]] = None
    image: Optional[Union[str, List[str]]] = None
    normalize: Optional[bool] = True


class TextInferenceRequest(BaseModel):
    name: str
    pretrained: Optional[str] = None
    text: Union[str, List[str]]
    normalize: Optional[bool] = True


class ImageInferenceRequest(BaseModel):
    name: str
    pretrained: Optional[str] = None
    image: Union[str, List[str]]
    normalize: Optional[bool] = True


class PairwiseInferenceRequest(BaseModel):
    name: str
    pretrained: Optional[str] = None
    left_text: Union[str, List[str]]
    right_text: Union[str, List[str]]
    left_image: Union[str, List[str]]
    right_image: Union[str, List[str]]
    metric: Literal['cosine', 'euclidean', 'inner_product', 'manhattan']
    normalize: Optional[bool] = True


class GenericModelRequest(BaseModel):
    name: str
    pretrained: Optional[str] = None


class SentenceTransformerModelRequest(BaseModel):
    name: str


class OpenCLIPModelRequest(BaseModel):
    name: str
    pretrained: Optional[str] = None
