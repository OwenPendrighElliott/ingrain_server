# Inference Server API Documentation

This section documents the endpoints available on the Inference Server.

## Endpoints

### GET `/health`

Returns a JSON message confirming that the inference server is running.

=== "cURL"
    ```bash
    curl -X GET "http://127.0.0.1:8686/health" -H "accept: application/json"
    ```

=== "Python"
    ```python
    import ingrain

    client = ingrain.Client()
    client.health()
    ```

### POST `/infer_text`

Performs text inference using the specified model.

**Request Body:**

- `name`: Model name.
- `pretrained`: Pretrained checkpoint identifier. Not used for Sentence Transformers models.
- `text`: Text input for inference. Can be a list to infer a batch.
- `normalize`: Boolean flag for normalization.
- `n_dims`: Number of dimensions for the embedding. Only useful for MRL models.

=== "cURL"
    ```bash
    curl -X POST "http://127.0.0.1:8686/infer_text" \
    -H "Content-Type: application/json" \
    -d '{"name": "intfloat/e5-small-v2", "pretrained": null, "text": "sample text", "normalize": true, "n_dims": null}'
    ```

=== "Python"
    ```python
    import ingrain

    client = ingrain.Client()

    response = client.infer_text(
        name="intfloat/e5-small-v2", 
        text="sample text", 
    )
    ```

### POST `/infer_image`

Performs image inference using the specified model.

**Request Body:**

- `name`: Model name.
- `pretrained`: Pretrained checkpoint identifier.
- `image`: Image URL or base64 encoded image. Can be a list to infer a batch.
- `normalize`: Boolean flag for normalization.
- `n_dims`: Number of dimensions for the embedding. Only useful for MRL models.
- `image_download_headers`: Optional headers for image downloading requests.

=== "cURL"
    ```bash
    curl -X POST "http://127.0.0.1:8686/infer_image" \
    -H "Content-Type: application/json" \
    -d '{"name": "MobileCLIP-S2", "pretrained": "datacompdr", "image": "http://example.com/image.jpg", "normalize": true, "n_dims": null}'
    ```

=== "Python"
    ```python
    import ingrain

    client = ingrain.Client()
    response = client.infer_image(
        name="MobileCLIP-S2", 
        pretrained="datacompdr", 
        image="http://example.com/image.jpg"
    )
    ```

### POST `/infer`

Performs both text and image inference concurrently, can do either or both together.

**Request Body:**

- `name`: Model name.
- `pretrained`: Pretrained checkpoint identifier.
- `text`: Text input for inference. Can be a list to infer a batch.
- `image`: Image URL or base64 encoded image. Can be a list to infer a batch.
- `normalize`: Boolean flag for normalization.
- `n_dims`: Number of dimensions for the embedding. Only useful for MRL models.
- `image_download_headers`: Optional headers for image downloading.

=== "cURL"
    ```bash
    curl -X POST "http://127.0.0.1:8686/infer" \
    -H "Content-Type: application/json" \
    -d '{""name": "MobileCLIP-S2", "pretrained": "datacompdr", "text": "sample text", "image": "http://example.com/image.jpg", "normalize": true, "n_dims": 512}'
    ```

=== "Python"
    ```python
    import ingrain

    client = ingrain.Client()
    response = client.infer_image(
        name="MobileCLIP-S2", 
        pretrained="datacompdr", 
        image="http://example.com/image.jpg", 
        text="sample text"
    )
    ```

### GET `/metrics`

Retrieves server metrics.

=== "cURL"
    ```bash
    curl -X GET "http://127.0.0.1:8686/metrics" -H "accept: application/json"
    ```

=== "Python"
    ```python
    import ingrain

    client = ingrain.Client()

    metrics = client.metrics()
    ```
