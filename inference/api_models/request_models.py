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
    library: Literal["open_clip", "sentence_transformers", "timm"]
    pretrained_name: str
    safetensors_url: str
    config_json_url: Optional[str] = None  # sentence_transformers
    tokenizer_json_url: Optional[str] = None  # sentence_transformers
    tokenizer_config_json_url: Optional[str] = None  # sentence_transformers
    vocab_txt_url: Optional[str] = None  # sentence_transformers
    special_tokens_map_json_url: Optional[str] = None  # sentence_transformers
    pooling_config_json_url: Optional[str] = None  # sentence_transformers
    sentence_bert_config_json_url: Optional[str] = None  # sentence_transformers
    modules_json_url: Optional[str] = None  # sentence_transformers
    mode: Optional[str] = None  # open_clip
    mean: Optional[List[float]] = None  # open_clip
    std: Optional[List[float]] = None  # open_clip
    interpolation: Optional[str] = None  # open_clip
    resize_mode: Optional[str] = None  # open_clip
    num_classes: Optional[int] = None  # timm
