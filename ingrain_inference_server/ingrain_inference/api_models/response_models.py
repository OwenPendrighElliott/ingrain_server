from ingrain_inference.api_models.camel_model import CamelModel


from typing import List, Optional, Dict


class EmbeddingResponse(CamelModel):
    text_embeddings: Optional[List[List[float]]] = None
    image_embeddings: Optional[List[List[float]]] = None
    processing_time_ms: float


class TextEmbeddingResponse(CamelModel):
    embeddings: List[List[float]]
    processing_time_ms: float


class ImageEmbeddingResponse(CamelModel):
    embeddings: List[List[float]]
    processing_time_ms: float


class ImageClassificationResponse(CamelModel):
    probabilities: List[List[float]]
    processing_time_ms: float


class GenericMessageResponse(CamelModel):
    message: str


class InferenceStats(CamelModel):
    count: Optional[str] = None
    ns: Optional[str] = None


class BatchStats(CamelModel):
    batch_size: str
    compute_input: InferenceStats
    compute_infer: InferenceStats
    compute_output: InferenceStats


class ModelStats(CamelModel):
    name: str
    version: str
    last_inference: Optional[str] = None
    inference_count: Optional[str] = None
    execution_count: Optional[str] = None
    inference_stats: Dict[str, Optional[InferenceStats]]
    batch_stats: Optional[List[BatchStats]] = None


class MetricsResponse(CamelModel):
    model_stats: List[ModelStats]
