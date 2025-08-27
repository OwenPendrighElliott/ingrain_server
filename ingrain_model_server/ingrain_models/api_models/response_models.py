from ingrain_models.api_models.camel_model import CamelModel

from typing import List, Optional


class LoadedModelResponse(CamelModel):
    models: List[str]


class RepositoryModel(CamelModel):
    name: str
    state: Optional[str]


class RepositoryModelResponse(CamelModel):
    models: List[RepositoryModel]


class GenericMessageResponse(CamelModel):
    message: str


class ModelEmbeddingDimsResponse(CamelModel):
    embedding_size: int


class ModelClassificationLabelsResponse(CamelModel):
    labels: List[str]
