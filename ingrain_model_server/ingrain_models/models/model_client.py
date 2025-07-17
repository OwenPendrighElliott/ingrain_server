import tritonclient.grpc as grpcclient
from ingrain_models.models.triton_open_clip.clip_model import (
    create_model_and_transforms_triton,
)
from ingrain_models.models.triton_sentence_transformers.sentence_transformer_model import (
    create_model as create_model_sentence_transformers,
)
from ingrain_models.models.triton_timm.timm_model import (
    create_model as create_model_timm,
)
from ingrain_common.common import get_model_name, get_text_image_model_names
from typing import Union

MODALITY_MAPPING = {
    "open_clip": {"text", "image"},
    "sentence_transformers": {"text"},
    "timm": {"image"},
}


class TritonModelLoadingClient:
    def __init__(
        self,
        triton_grpc_url: str,
        model: str,
        pretrained: Union[str, None],
        library_name: str,
        triton_model_repository_path: str,
        custom_model_dir: str,
    ):
        self.model = model
        self.pretrained = pretrained
        self.triton_model_repository_path = triton_model_repository_path
        self.custom_model_dir = custom_model_dir
        self.library_name = library_name
        self.triton_client = grpcclient.InferenceServerClient(
            url=triton_grpc_url, verbose=False
        )
        self.modalities = MODALITY_MAPPING[library_name]

        self.model_name = get_model_name(model, pretrained)
        self.image_model_name = None
        self.text_model_name = None

        if library_name == "open_clip":
            self.text_model_name, self.image_model_name = get_text_image_model_names(
                model, pretrained
            )

    def create_triton_model(self):
        if self.library_name == "open_clip":
            return self._create_triton_open_clip_model()
        elif self.library_name == "sentence_transformers":
            return self._create_triton_sentence_transformer_model()
        elif self.library_name == "timm":
            return self._create_triton_timm_model()
        else:
            raise ValueError(f"Unsupported library name: {self.library_name}")

    def _create_triton_open_clip_model(self):
        if not self.triton_client.is_model_ready(
            self.text_model_name
        ) or not self.triton_client.is_model_ready(self.image_model_name):
            self.text_model_name, self.image_model_name = (
                create_model_and_transforms_triton(
                    self.model,
                    self.pretrained,
                    self.triton_model_repository_path,
                    self.custom_model_dir,
                    self.text_model_name,
                    self.image_model_name,
                )
            )
            # self.triton_client.load_model(self.text_model_name)
            # self.triton_client.load_model(self.image_model_name)

    def _create_triton_sentence_transformer_model(self):
        if not self.triton_client.is_model_ready(self.model_name):
            create_model_sentence_transformers(
                self.model,
                self.triton_model_repository_path,
                self.custom_model_dir,
                self.model_name,
            )
            # self.triton_client.load_model(self.model_name)

    def _create_triton_timm_model(self):
        if not self.triton_client.is_model_ready(self.model_nice_name):
            create_model_timm(
                self.model,
                self.pretrained,
                self.triton_model_repository_path,
                self.custom_model_dir,
                self.model_name,
            )
            # self.triton_client.load_model(self.model_nice_name)

    def is_in_repository(self) -> bool:
        repository_index = self.triton_client.get_model_repository_index(as_json=True)
        if "models" not in repository_index:
            return False

        model_names = [model["name"] for model in repository_index["models"]]
        if self.library_name == "open_clip":
            return (
                self.text_model_name in model_names
                and self.image_model_name in model_names
            )
        else:
            return self.model_name in model_names

    def unload(self):
        if self.library_name == "open_clip":
            self.triton_client.unload_model(self.text_model_name)
            self.triton_client.unload_model(self.image_model_name)
        else:
            self.triton_client.unload_model(self.model_name)

    def load(self):
        if self.library_name == "open_clip":
            self.triton_client.load_model(self.text_model_name)
            self.triton_client.load_model(self.image_model_name)
        else:
            self.triton_client.load_model(self.model_name)

    def is_ready(self) -> bool:
        if self.library_name == "open_clip":
            return self.triton_client.is_model_ready(
                self.text_model_name
            ) and self.triton_client.is_model_ready(self.image_model_name)
        else:
            return self.triton_client.is_model_ready(self.model_name)
