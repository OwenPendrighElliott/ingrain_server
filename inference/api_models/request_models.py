from pydantic import BaseModel

from typing import List, Optional, Union, Literal


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


class GenericModelRequest(BaseModel):
    name: str
    pretrained: Optional[str] = None


class SentenceTransformerModelRequest(BaseModel):
    name: str


class TimmModelRequest(BaseModel):
    name: str
    pretrained: Optional[str | bool] = None


class OpenCLIPModelRequest(BaseModel):
    name: str
    pretrained: Optional[str] = None


class DownloadCustomModelRequest(BaseModel):
    url: str
    model_name: str
    model_url: str
    config_json_url: Optional[str]
    tokenizer_json_url: Optional[str]
    tokenizer_config_json_url: Optional[str]
    vocab_txt_url: Optional[str]
    special_tokens_map_json_url: Optional[str]
    pooling_config_json_url: Optional[str]
    sentence_bert_config_json_url: Optional[str] = None
