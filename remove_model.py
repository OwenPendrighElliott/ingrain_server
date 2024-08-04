import requests

model_name = "intfloat/e5-small-v2"

response = requests.post(
    "http://localhost:8686/unload_model",
    json={"model_name": model_name},
)
print(response.json())

# infer text
response = requests.post(
    "http://localhost:8686/infer_text", json={"model_name": model_name, "text": "a cat"}
)
print(response.json())

response = requests.post(
    "http://localhost:8686/infer_text", json={"model_name": model_name, "text": "a cat"}
)
print(response.json())

response = requests.post(
    "http://localhost:8686/infer_text", json={"model_name": model_name, "text": "a cat"}
)
print(response.json())

response = requests.post(
    "http://localhost:8686/infer_text", json={"model_name": model_name, "text": "a cat"}
)
print(response.json())

response = requests.post(
    "http://localhost:8686/infer_text", json={"model_name": model_name, "text": "a cat"}
)
print(response.json())

response = requests.post(
    "http://localhost:8686/infer_text", json={"model_name": model_name, "text": "a cat"}
)
print(response.json())

response = requests.post(
    "http://localhost:8686/infer_text", json={"model_name": model_name, "text": "a cat"}
)
print(response.json())

response = requests.post(
    "http://localhost:8686/infer_text", json={"model_name": model_name, "text": "a cat"}
)
print(response.json())
