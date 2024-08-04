from pydantic import BaseModel

from typing import List, Optional, Union


class InferenceRequest(BaseModel):
    model_name: str
    pretrained: Optional[str] = None
    text: Optional[Union[str, List[str]]] = None
    image: Optional[Union[str, List[str]]] = None
    normalize: Optional[bool] = True


class TextInferenceRequest(BaseModel):
    model_name: str
    pretrained: Optional[str] = None
    text: Union[str, List[str]]
    normalize: Optional[bool] = True


class ImageInferenceRequest(BaseModel):
    model_name: str
    pretrained: Optional[str] = None
    image: Union[str, List[str]]
    normalize: Optional[bool] = True


class GenericModelRequest(BaseModel):
    model_name: str
    pretrained: Optional[str] = None


class SentenceTransformerModelRequest(BaseModel):
    model_name: str


class OpenCLIPModelRequest(BaseModel):
    model_name: str
    pretrained: Optional[str] = None
