
from typing import Union

def get_model_name(model_name: str, pretrained: Union[str, None] = None) -> str:
    name = model_name.replace("/", "_")
    if pretrained is not None:
        name += f"_{pretrained}"
    return name