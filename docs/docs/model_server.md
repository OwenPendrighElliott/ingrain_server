# Model Server API Documentation

This section documents the endpoints available on the Model Server.

## Endpoints

### GET `/health`

Returns a JSON message confirming that the model server is running.

=== "cURL"
    ```bash
    curl -X GET "http://127.0.0.1:8687/health" -H "accept: application/json"
    ```

=== "Python"
    ```python
    import ingrain

    client = ingrain.Client()
    client.health()
    ```

### POST `/load_clip_model`

Loads a CLIP model.

**Request Body:**

- `name`: Model name.
- `pretrained`: Pretrained checkpoint identifier.

=== "cURL"
    ```bash
    curl -X POST "http://127.0.0.1:8687/load_clip_model" \
    -H "Content-Type: application/json" \
    -d '{"name": "MobileCLIP-S2", "pretrained": "datacompdr"}'
    ```

=== "Python"
    ```python
    import ingrain

    client = ingrain.Client()

    client.load_clip_model(name="MobileCLIP-S2", pretrained="datacompdr")
    ```

### POST `/load_sentence_transformer_model`

Loads a Sentence Transformer model.

**Request Body:**

- `name`: Model name.

=== "cURL"
    ```bash
    curl -X POST "http://127.0.0.1:8687/load_sentence_transformer_model" \
    -H "Content-Type: application/json" \
    -d '{"name": "intfloat/e5-small-v2"}'
    ```

=== "Python"
    ```python
    import ingrain

    client = ingrain.Client()
    client.load_sentence_transformer_model(name="intfloat/e5-small-v2")
    ```

### POST `/load_timm_model`

Loads a Timm model.

**Request Body:**

- `name`: Model name.
- `pretrained`: Pretrained, `true` or `false` - defaults to `true`. There is no clear use case for setting this to `false`.

=== "cURL"
    ```bash
    curl -X POST "http://127.0.0.1:8687/load_timm_model" \
    -H "Content-Type: application/json" \
    -d '{"name": "hf_hub:timm/mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k", "pretrained": true}'
    ```

=== "Python"
    ```python
    import ingrain

    client = ingrain.Client()
    client.load_timm_model(name="hf_hub:timm/mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k")
    ```

### POST `/unload_model`

Unloads a model from the server.

**Request Body:**

- `name`: Model name.
- `pretrained`: Pretrained checkpoint identifier. Not used for Sentence Transformers models.

=== "cURL"
    ```bash
    curl -X POST "http://127.0.0.1:8687/unload_model" \
    -H "Content-Type: application/json" \
    -d '{"name": "intfloat/e5-small-v2", "pretrained": null}'
    ```

=== "Python"
    ```python
    import ingrain

    client = ingrain.Client()
    client.unload_model(name="intfloat/e5-small-v2")
    ```

### DELETE `/delete_model`

Deletes a model from the repository.

**Request Body:**

- `name`: Model name.
- `pretrained`: Pretrained checkpoint identifier. Not used for Sentence Transformers models.

=== "cURL"
    ```bash
    curl -X DELETE "http://127.0.0.1:8687/delete_model" \
    -H "Content-Type: application/json" \
    -d '{"name": "intfloat/e5-small-v2", "pretrained": null}'
    ```

=== "Python"
    ```python
    import ingrain

    client = ingrain.Client()
    client.delete_model(name="intfloat/e5-small-v2")
    ```

### GET `/loaded_models`

Retrieves a list of currently loaded models.

=== "cURL"
    ```bash
    curl -X GET "http://127.0.0.1:8687/loaded_models" -H "accept: application/json"
    ```

=== "Python"
    ```python
    import ingrain

    client = ingrain.Client()
    resp = client.loaded_models()
    ```

### GET `/repository_models`

Retrieves a list of models in the repository.

=== "cURL"
    ```bash
    curl -X GET "http://127.0.0.1:8687/repository_models" -H "accept: application/json"
    ```

=== "Python"
    ```python
    import ingrain

    client = ingrain.Client()
    resp = client.repository_models()
    ```

### POST `/download_custom_model`

**Experimental, subject to change.**
Downloads a custom model.

**Request Body:**

- `library`: The model library type (`open_clip`, `sentence_transformers`, or `timm`).
- `pretrained_name`: The name of the pretrained model.
- `safetensors_url`: URL to the model file.
- Other parameters may be required based on the library.

=== "cURL"
    ```bash
    curl -X POST "http://127.0.0.1:8687/download_custom_model" \
    -H "Content-Type: application/json" \
    -d '{"library": "open_clip", "pretrained_name": "your_model", "safetensors_url": "http://example.com/model.safetensors"}'
    ```

=== "Python"
    ```python
    # N/A
    ```
