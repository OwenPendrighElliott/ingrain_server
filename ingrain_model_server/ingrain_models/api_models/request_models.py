from ingrain_models.api_models.camel_model import CamelModel

from typing import List, Optional, Literal


class LoadModelRequest(CamelModel):
    name: str
    library: Literal["open_clip", "sentence_transformers", "timm"]


class UnloadModelRequest(CamelModel):
    name: str


class ModelMetadataRequest(CamelModel):
    name: str


class DownloadCustomModelRequest(CamelModel):
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
