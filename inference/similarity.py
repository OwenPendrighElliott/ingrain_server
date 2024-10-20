import numpy as np

from typing import Literal

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate the cosine similarity between two vectors.

    Args:
        a (np.ndarray): Vector a.
        b (np.ndarray): Vector b.

    Returns:
        float: The cosine similarity.
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

def euclidean_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate the Euclidean similarity between two vectors.

    Args:
        a (np.ndarray): Vector a.
        b (np.ndarray): Vector b.

    Returns:
        float: The Euclidean similarity.
    """
    return np.linalg.norm(a - b)

def inner_product_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate the inner product similarity between two vectors.

    Args:
        a (np.ndarray): Vector a.
        b (np.ndarray): Vector b.

    Returns:
        float: The inner product similarity.
    """
    return np.inner(a, b)

def manhattan_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate the Manhattan similarity between two vectors.

    Args:
        a (np.ndarray): Vector a.
        b (np.ndarray): Vector b.

    Returns:
        float: The Manhattan similarity.
    """
    return np.sum(np.abs(a - b))

def calculate_similarity(
    a: np.ndarray, b: np.ndarray, metric: Literal['cosine', 'euclidean', 'inner_product', 'manhattan'] = 'cosine'
) -> float:
    """Calculate the similarity between two vectors.

    Args:
        a (np.ndarray): Vector a.
        b (np.ndarray): Vector b.
        metric (Literal['cosine', 'euclidean', 'inner_product', 'manhattan'], optional): The similarity metric to use. Defaults to 'cosine'.

    Returns:
        float: The similarity value.
    """
    if metric == 'cosine':
        return cosine_similarity(a, b)
    elif metric == 'euclidean':
        return euclidean_similarity(a, b)
    elif metric == 'inner_product':
        return inner_product_similarity(a, b)
    elif metric == 'manhattan':
        return manhattan_similarity(a, b)
