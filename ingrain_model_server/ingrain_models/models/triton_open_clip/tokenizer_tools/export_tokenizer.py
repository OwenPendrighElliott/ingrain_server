from open_clip.tokenizer import SimpleTokenizer, SigLipTokenizer, HFTokenizer
from tokenizers import Tokenizer
import os
import json
from typing import Union, Optional


def export_tokenizer(
    tokenizer: Union[SimpleTokenizer, SigLipTokenizer, HFTokenizer],
    folder_path: str,
    text_clean: Optional[str] = None,
) -> None:

    context_length = tokenizer.context_length

    if isinstance(tokenizer, SimpleTokenizer):
        save_pretrained_simple_tokenizer(folder_path)
    else:
        tokenizer.save_pretrained(folder_path)

    with open(os.path.join(folder_path, "_tokenizer_context_length.json"), "w") as f:
        json.dump({"context_length": context_length}, f)

    clean_fn = ""
    if text_clean is not None:
        clean_fn = text_clean

    with open(os.path.join(folder_path, "_tokenizer_meta.json"), "w") as f:
        json.dump(
            {"tokenizer_type": type(tokenizer).__name__, "text_clean": clean_fn}, f
        )


def save_pretrained_simple_tokenizer(folder_path: str) -> Tokenizer:
    """All default SimpleTokenizers are openai/clip-vit-large-patch14 at the moment"""
    t: Tokenizer = Tokenizer.from_pretrained("openai/clip-vit-large-patch14")
    os.makedirs(folder_path, exist_ok=True)
    t.save(os.path.join(folder_path, "tokenizer.json"))
