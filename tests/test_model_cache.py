import pytest
from unittest.mock import MagicMock
from inference.triton_open_clip.clip_model import TritonCLIPClient
from inference.triton_sentence_transformers.sentence_transformer_model import TritonSentenceTransformersClient
from inference.model_cache import LRUModelCache

@pytest.fixture
def mock_clip_client():
    client = MagicMock(spec=TritonCLIPClient)
    client.load = MagicMock()
    client.unload = MagicMock()
    return client

@pytest.fixture
def mock_sentence_client():
    client = MagicMock(spec=TritonSentenceTransformersClient)
    client.load = MagicMock()
    client.unload = MagicMock()
    return client

def test_lru_model_cache_put_and_get(mock_clip_client, mock_sentence_client):
    cache = LRUModelCache(capacity=2)
    
    key1 = ("model1", "pretrained1")
    key2 = ("model2", None)
    
    cache.put(key1, mock_clip_client)
    assert cache.get(key1) == mock_clip_client
    assert cache.hits == 1
    assert cache.misses == 0
    
    cache.put(key2, mock_sentence_client)
    assert cache.get(key2) == mock_sentence_client
    assert cache.hits == 2
    assert cache.misses == 0
    
    assert cache.get(("non_existing_model", None)) is None
    assert cache.hits == 2
    assert cache.misses == 1

def test_lru_model_cache_eviction(mock_clip_client, mock_sentence_client):
    cache = LRUModelCache(capacity=1)
    
    key1 = ("model1", "pretrained1")
    key2 = ("model2", None)
    
    cache.put(key1, mock_clip_client)
    cache.put(key2, mock_sentence_client)
    
    assert cache.get(key1) is None
    assert cache.hits == 0
    assert cache.misses == 1
    mock_clip_client.unload.assert_called_once()
    
    assert cache.get(key2) == mock_sentence_client
    assert cache.hits == 1
    assert cache.misses == 1

def test_lru_model_cache_remove(mock_clip_client):
    cache = LRUModelCache(capacity=2)
    
    key1 = ("model1", "pretrained1")
    
    cache.put(key1, mock_clip_client)
    cache.remove(key1)
    
    assert cache.get(key1) is None
    assert cache.hits == 0
    assert cache.misses == 1
    mock_clip_client.unload.assert_called_once()

def test_lru_model_cache_get_stats(mock_clip_client, mock_sentence_client):
    cache = LRUModelCache(capacity=2)
    
    key1 = ("model1", "pretrained1")
    key2 = ("model2", None)
    
    cache.put(key1, mock_clip_client)
    cache.put(key2, mock_sentence_client)
    
    stats = cache.get_stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 0
    assert stats["capacity"] == 2
    assert stats["current_size"] == 2
    assert stats["loaded_models"] == 2

    cache.get(key1)
    cache.get(key2)
    
    stats = cache.get_stats()
    assert stats["hits"] == 2
    assert stats["misses"] == 0
    assert stats["current_size"] == 2
    assert stats["loaded_models"] == 2
