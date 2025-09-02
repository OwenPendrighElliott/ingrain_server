from ingrain_models.api_models.camel_model import CamelModel

from typing import List, Optional


class LoadedModelData(CamelModel):
    name: str
    library: str


class LoadedModelResponse(CamelModel):
    models: List[LoadedModelData]


class RepositoryModel(CamelModel):
    name: str
    library: str
    state: Optional[str]


class RepositoryModelResponse(CamelModel):
    models: List[RepositoryModel]


class GenericMessageResponse(CamelModel):
    message: str


class ModelEmbeddingDimsResponse(CamelModel):
    embedding_size: int


class ModelClassificationLabelsResponse(CamelModel):
    labels: List[str]
