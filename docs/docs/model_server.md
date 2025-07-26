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

### POST `/load_model`

Load a model into the server.

**Request Body:**

- `name`: Model name.
- `library`: Model library type, must be one of `open_clip`, `sentence_transformers`, or `timm`.

=== "cURL"
    ```bash
    curl -X POST "http://127.0.0.1:8687/load_model" \
    -H "Content-Type: application/json" \
    -d '{"name": "intfloat/e5-small-v2", "library": "sentence_transformers"}'
    ```

=== "Python"
    ```python
    import ingrain

    client = ingrain.Client()

    client.load_clip_model(name="intfloat/e5-small-v2", library="sentence_transformers")
    ```

### POST `/unload_model`

Unloads a model from the server.

**Request Body:**

- `name`: Model name.

=== "cURL"
    ```bash
    curl -X POST "http://127.0.0.1:8687/unload_model" \
    -H "Content-Type: application/json" \
    -d '{"name": "intfloat/e5-small-v2"}'
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

=== "cURL"
    ```bash
    curl -X DELETE "http://127.0.0.1:8687/delete_model" \
    -H "Content-Type: application/json" \
    -d '{"name": "intfloat/e5-small-v2"}'
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
