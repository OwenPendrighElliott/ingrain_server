import requests


# load a model
model_name = "ViT-B-32"
pretrained = "laion2b_s34b_b79k"

response = requests.post(
    "http://localhost:8686/load_clip_model",
    json={"model_name": model_name, "pretrained": pretrained},
)
print(response.json())

# infer text
response = requests.post(
    "http://localhost:8686/infer_text",
    json={"model_name": model_name, "pretrained": pretrained, "text": "a cat"},
)
print(response.json())


# infer image
response = requests.post(
    "http://localhost:8686/infer_image",
    json={
        "model_name": model_name,
        "pretrained": pretrained,
        "image": "https://thumbor.forbes.com/thumbor/fit-in/900x510/https://www.forbes.com/advisor/wp-content/uploads/2023/07/top-20-small-dog-breeds.jpeg.jpg",
    },
)
print(response.json())

# infer
response = requests.post(
    "http://localhost:8686/infer",
    json={
        "model_name": model_name,
        "pretrained": pretrained,
        "texts": ["a cat", "a dog"],
        "images": [
            "https://thumbor.forbes.com/thumbor/fit-in/900x510/https://www.forbes.com/advisor/wp-content/uploads/2023/07/top-20-small-dog-breeds.jpeg.jpg",
            "https://thumbor.forbes.com/thumbor/fit-in/900x510/https://www.forbes.com/advisor/wp-content/uploads/2023/07/top-20-small-dog-breeds.jpeg.jpg",
        ],
    },
)
print(response.json().keys())

# loaded models
response = requests.get("http://localhost:8686/loaded_models")
print(response.json())

# metrics
response = requests.get("http://localhost:8686/metrics")
print(response.json())

# load a model
model_name = "intfloat/e5-small-v2"

response = requests.post(
    "http://localhost:8686/load_sentence_transformer_model",
    json={"model_name": model_name},
)
print(response.json())

# infer text
response = requests.post(
    "http://localhost:8686/infer_text", json={"model_name": model_name, "text": "a cat"}
)
print(response.json())


# infer
response = requests.post(
    "http://localhost:8686/infer",
    json={"model_name": model_name, "texts": ["a cat", "a dog"]},
)
print(response.json().keys())

# loaded models
response = requests.get("http://localhost:8686/loaded_models")
print(response.json())

# metrics
response = requests.get("http://localhost:8686/metrics")
print(response.json())
