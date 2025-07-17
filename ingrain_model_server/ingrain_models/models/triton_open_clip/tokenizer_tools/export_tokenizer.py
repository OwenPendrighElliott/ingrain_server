from open_clip.tokenizer import SimpleTokenizer, SigLipTokenizer, HFTokenizer
from tokenizers import Tokenizer
import os
import json
from typing import Union


def export_tokenizer(
    tokenizer: Union[SimpleTokenizer, SigLipTokenizer, HFTokenizer], folder_path: str
) -> None:

    context_length = tokenizer.context_length

    if isinstance(tokenizer, SimpleTokenizer):
        save_pretrained_simple_tokenizer(folder_path)
    else:
        tokenizer.save_pretrained(folder_path)

    with open(os.path.join(folder_path, "_tokenizer_context_length.json"), "w") as f:
        json.dump({"context_length": context_length}, f)


def save_pretrained_simple_tokenizer(folder_path: str) -> Tokenizer:
    """All default SimpleTokenizers are openai/clip-vit-large-patch14 at the moment"""
    t: Tokenizer = Tokenizer.from_pretrained("openai/clip-vit-large-patch14")
    os.makedirs(folder_path, exist_ok=True)
    t.save(os.path.join(folder_path, "tokenizer.json"))
