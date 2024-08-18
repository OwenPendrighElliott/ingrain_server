from pydantic import BaseModel

from typing import List, Optional, Dict


class InferenceResponse(BaseModel):
    textEmbeddings: Optional[List[List[float]]] = None
    imageEmbeddings: Optional[List[List[float]]] = None
    processingTimeMs: float


class TextInferenceResponse(BaseModel):
    embeddings: List[List[float]]
    processingTimeMs: float


class ImageInferenceResponse(BaseModel):
    embeddings: List[List[float]]
    processingTimeMs: float


class LoadedModelResponse(BaseModel):
    models: List[str]


class RepositoryModel(BaseModel):
    name: str
    state: Optional[str]


class RepositoryModelResponse(BaseModel):
    models: List[RepositoryModel]


class GenericMessageResponse(BaseModel):
    message: str


class InferenceStats(BaseModel):
    count: Optional[str] = None
    ns: Optional[str] = None


class BatchStats(BaseModel):
    batch_size: str
    compute_input: InferenceStats
    compute_infer: InferenceStats
    compute_output: InferenceStats


class ModelStats(BaseModel):
    name: str
    version: str
    last_inference: Optional[str] = None
    inference_count: Optional[str] = None
    execution_count: Optional[str] = None
    inference_stats: Dict[str, Optional[InferenceStats]]
    batch_stats: Optional[List[BatchStats]] = None


class MetricsResponse(BaseModel):
    modelStats: List[ModelStats]
