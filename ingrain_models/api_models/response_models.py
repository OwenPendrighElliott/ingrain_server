from pydantic import BaseModel

from typing import List, Optional, Dict


class LoadedModelResponse(BaseModel):
    models: List[str]


class RepositoryModel(BaseModel):
    name: str
    state: Optional[str]


class RepositoryModelResponse(BaseModel):
    models: List[RepositoryModel]


class GenericMessageResponse(BaseModel):
    message: str
