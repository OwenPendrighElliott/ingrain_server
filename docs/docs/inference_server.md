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

### POST `/embed_text`

Performs text inference using the specified model.

**Request Body:**

- `name`: Model name.
- `text`: Text input for inference. Can be a list to infer a batch.
- `normalize`: Boolean flag for normalization.
- `nDims`: Number of dimensions for the embedding. Only useful for MRL models.

=== "cURL"
    ```bash
    curl -X POST "http://127.0.0.1:8686/embed_text" \
    -H "Content-Type: application/json" \
    -d '{"name": "intfloat/e5-small-v2" "text": "sample text", "normalize": true, "nDims": null}'
    ```

=== "Python"
    ```python
    import ingrain

    client = ingrain.Client()

    response = client.embed_text(
        name="intfloat/e5-small-v2", 
        text="sample text", 
    )
    ```

**Example Response:**

```json
{
    "embeddings": [[0.235, 0.3754, -0.24645, ...], ...],
    "processingTimeMs": 15
}
```

### POST `/embed_image`

Performs image inference using the specified model.

**Request Body:**

- `name`: Model name.
- `image`: Image URL or base64 encoded image. Can be a list to infer a batch.
- `normalize`: Boolean flag for normalization.
- `nDims`: Number of dimensions for the embedding. Only useful for MRL models.
- `image_download_headers`: Optional headers for image downloading requests.

=== "cURL"
    ```bash
    curl -X POST "http://127.0.0.1:8686/embed_image" \
    -H "Content-Type: application/json" \
    -d '{"name": "hf-hub:apple/MobileCLIP-S1-OpenCLIP", "image": "http://example.com/image.jpg", "normalize": true, "nDims": null}'
    ```

=== "Python"
    ```python
    import ingrain

    client = ingrain.Client()
    response = client.embed_image(
        name="hf-hub:apple/MobileCLIP-S1-OpenCLIP", 
        image="http://example.com/image.jpg"
    )
    ```

**Example Response:**

```json
{
    "embeddings": [[0.235, 0.3754, -0.24645, ...], ...],
    "processingTimeMs": 15
}
```

### POST `/infer`

Performs both text and image inference concurrently, can do either or both together.

**Request Body:**

- `name`: Model name.
- `text`: Text input for inference. Can be a list to infer a batch.
- `image`: Image URL or base64 encoded image. Can be a list to infer a batch.
- `normalize`: Boolean flag for normalization.
- `nDims`: Number of dimensions for the embedding. Only useful for MRL models.
- `image_download_headers`: Optional headers for image downloading.

=== "cURL"
    ```bash
    curl -X POST "http://127.0.0.1:8686/embed" \
    -H "Content-Type: application/json" \
    -d '{"name": "hf-hub:apple/MobileCLIP-S1-OpenCLIP", "text": "sample text", "image": "http://example.com/image.jpg", "normalize": true, "nDims": 512}'
    ```

=== "Python"
    ```python
    import ingrain

    client = ingrain.Client()
    response = client.embed(
        name="hf-hub:apple/MobileCLIP-S1-OpenCLIP", 
        image="http://example.com/image.jpg", 
        text="sample text"
    )
    ```

```json
{
    "textEmbeddings": [[0.235, 0.3754, -0.24645, ...], ...],
    "imageEmbeddings": [[0.25745, -0.73244, 0.8743, ...], ...],
    "processingTimeMs": 15
}
```

### Post `/classify_image`

Performs image classification using the specified model.

**Request Body:**

- `name`: Model name.
- `image`: Image URL or base64 encoded image. Can be a list to infer a batch.
- `image_download_headers`: Optional headers for image downloading requests.

=== "cURL"
    ```bash
    curl -X POST "http://127.0.0.1:8686/classify_image" \
    -H "Content-Type: application/json" \
    -d '{"name": "hf-hub:apple/MobileCLIP-S1-OpenCLIP", "image": "http://example.com/image.jpg",}'
    ```

=== "Python"
    ```python
    import ingrain

    client = ingrain.Client()
    response = client.classify_image(
        name="hf-hub:apple/MobileCLIP-S1-OpenCLIP", 
        image="http://example.com/image.jpg"
    )
    ```

```json
{
    "probabilities": [[0.896, 0.002, 0.00234, ...], ...],
    "processingTimeMs": 15
}
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

**Example Response:**

```json
{
  "modelStats": [
    {
      "name": "hf-hub:timm_ViT-B-16-SigLIP-i18n-256/image_encoder",
      "version": "1",
      "lastInference": null,
      "inferenceCount": null,
      "executionCount": null,
      "inferenceStats": {
        "success": {
          "count": null,
          "ns": null
        },
        "fail": {
          "count": null,
          "ns": null
        },
        "queue": {
          "count": null,
          "ns": null
        },
        "compute_input": {
          "count": null,
          "ns": null
        },
        "compute_infer": {
          "count": null,
          "ns": null
        },
        "compute_output": {
          "count": null,
          "ns": null
        },
        "cache_hit": {
          "count": null,
          "ns": null
        },
        "cache_miss": {
          "count": null,
          "ns": null
        }
      },
      "batchStats": null
    },
    {
      "name": "hf-hub:timm_ViT-B-16-SigLIP-i18n-256/text_encoder",
      "version": "1",
      "lastInference": "1756351592249",
      "inferenceCount": "48",
      "executionCount": "48",
      "inferenceStats": {
        "success": {
          "count": "48",
          "ns": "7090273130"
        },
        "fail": {
          "count": null,
          "ns": null
        },
        "queue": {
          "count": "48",
          "ns": "8507746"
        },
        "compute_input": {
          "count": "48",
          "ns": "3113130"
        },
        "compute_infer": {
          "count": "48",
          "ns": "7021648834"
        },
        "compute_output": {
          "count": "48",
          "ns": "2755455"
        },
        "cache_hit": {
          "count": null,
          "ns": null
        },
        "cache_miss": {
          "count": null,
          "ns": null
        }
      },
      "batchStats": [
        {
          "batchSize": "1",
          "computeInput": {
            "count": "48",
            "ns": "3113130"
          },
          "computeInfer": {
            "count": "48",
            "ns": "7021648834"
          },
          "computeOutput": {
            "count": "48",
            "ns": "2755455"
          }
        }
      ]
    },
  ]
}
```