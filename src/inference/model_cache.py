import collections
from ..inference.triton_open_clip.clip_model import (
    TritonCLIPModelClient,
)
from ..inference.triton_timm.timm_model import (
    TritonTimmModelClient,
)

from ..inference.triton_open_clip.clip_inference import (
    TritonCLIPInferenceClient,
)
from ..inference.triton_timm.timm_inference import (
    TritonTimmInferenceClient,
)
from ..inference.triton_sentence_transformers.sentence_transformer_model import (
    TritonSentenceTransformersModelClient,
)
from ..inference.triton_sentence_transformers.sentence_transformer_inference import (
    TritonSentenceTransformersInferenceClient,
)
from typing import Union, Dict, Tuple


class LRUModelCache:
    def __init__(self, capacity: int):
        self.capacity: int = capacity
        self.data: Dict[
            Tuple[str, Union[str, None]],
            Union[
                TritonCLIPModelClient,
                TritonSentenceTransformersModelClient,
                TritonCLIPInferenceClient,
                TritonSentenceTransformersInferenceClient,
            ],
        ] = {}
        self.loaded: Dict[Tuple[str, Union[str, None]], bool] = {}
        self.usage_order = collections.OrderedDict()
        self.hits: int = 0
        self.misses: int = 0

    def __len__(self):
        return len(self.data)

    def keys(self):
        return self.data.keys()

    def remove(self, key: Tuple[str, Union[str, None]]) -> None:
        if key in self.data:
            self.data[key].unload()
            del self.data[key]
            del self.loaded[key]
            del self.usage_order[key]

    def get(self, key: Tuple[str, Union[str, None]]) -> Union[
        TritonCLIPModelClient,
        TritonSentenceTransformersModelClient,
        TritonCLIPInferenceClient,
        TritonSentenceTransformersInferenceClient,
        None,
    ]:
        if key not in self.data:
            self.misses += 1
            return None
        if not self.loaded[key]:
            self.misses += 1
            self.data[key].load()
            self.loaded[key] = True
        else:
            self.hits += 1
        if key not in self.usage_order:
            self.put(key, self.data[key])
        self.usage_order.move_to_end(key)
        return self.data[key]

    def put(
        self,
        key: Tuple[str, Union[str, None]],
        value: Union[
            TritonCLIPModelClient,
            TritonSentenceTransformersModelClient,
            TritonCLIPInferenceClient,
            TritonSentenceTransformersInferenceClient,
        ],
    ) -> None:
        if self.capacity <= 0:
            return

        if len(self.data) >= self.capacity:
            self._unload_least_recently_used()
        self.data[key] = value
        self.loaded[key] = True
        self.usage_order[key] = None

    def _unload_least_recently_used(self):
        oldest_key, _ = self.usage_order.popitem(last=False)
        self.data[oldest_key].unload()  # Unload the model
        self.loaded[oldest_key] = False
        del self.data[oldest_key]

    def get_stats(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "capacity": self.capacity,
            "current_size": len(self.data),
            "loaded_models": sum(self.loaded.values()),
        }
