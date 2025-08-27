import ftfy
import html
import string

from typing import Optional, Callable, Literal

"""Tokenizer helpers from https://github.com/mlfoundations/open_clip"""


def basic_clean(text: str) -> str:
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text: str) -> str:
    text = " ".join(text.split())
    text = text.strip()
    return text


def canonicalize_text(
    text: str,
    *,
    keep_punctuation_exact_string: Optional[bool] = None,
    trans_punctuation: dict = str.maketrans("", "", string.punctuation),
) -> str:
    """Returns canonicalized `text` (lowercase and punctuation removed).

    From: https://github.com/google-research/big_vision/blob/53f18caf27a9419231bbf08d3388b07671616d3d/big_vision/evaluators/proj/image_text/prompt_engineering.py#L94

    Args:
      text: string to be canonicalized.
      keep_punctuation_exact_string: If provided, then this exact string kept.
        For example providing '{}' will keep any occurrences of '{}' (but will
        still remove '{' and '}' that appear separately).
    """
    text = text.replace("_", " ")
    if keep_punctuation_exact_string:
        text = keep_punctuation_exact_string.join(
            part.translate(trans_punctuation)
            for part in text.split(keep_punctuation_exact_string)
        )
    else:
        text = text.translate(trans_punctuation)
    text = text.lower()
    text = " ".join(text.split())
    return text.strip()


def _clean_canonicalize(x: str) -> str:
    # basic, remove whitespace, remove punctuation, lower case
    return canonicalize_text(basic_clean(x))


def _clean_lower(x: str) -> str:
    # basic, remove whitespace, lower case
    return whitespace_clean(basic_clean(x)).lower()


def _clean_whitespace(x: str) -> str:
    # basic, remove whitespace
    return whitespace_clean(basic_clean(x))


def noop(x: str) -> str:
    return x


def get_clean_fn(
    type: Literal["canonicalize", "lower", "whitespace"]
) -> Callable[[str], str]:
    if type == "canonicalize":
        return _clean_canonicalize
    elif type == "lower":
        return _clean_lower
    elif type == "whitespace":
        return _clean_whitespace

    raise ValueError(f"Unknown clean function type: {type}")
