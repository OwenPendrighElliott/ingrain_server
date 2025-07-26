import time
import ingrain
import os
import requests
from flask import Flask, request, render_template, jsonify, send_from_directory

# Constants
HNSWLIB_SERVER_URL = "http://localhost:8685"
IMAGE_DIR = "images"
INDEX_NAME = "image_search"
CLIP_MODEL_NAME = "hf-hub:timm/PE-Core-B-16"
MODEL_DIM = 512
K = 20

# Create Flask app
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/search", methods=["POST"])
def search():
    query_text = request.json["query_text"]

    client = ingrain.Client()
    response = client.infer_text(name=CLIP_MODEL_NAME, text=query_text)
    query_embedding = response["embeddings"][0]

    # Search the index for similar images
    start_time = time.time()
    search_response = requests.post(
        f"{HNSWLIB_SERVER_URL}/search",
        json={
            "indexName": INDEX_NAME,
            "queryVector": query_embedding,
            "k": K,
            "efSearch": 512,
            "filter": "",
            "returnMetadata": True,
        },
    )

    results = [
        os.path.join(IMAGE_DIR, h["source_image"])
        for h in search_response.json()["metadatas"]
    ]

    inference_time = round(response["processingTimeMs"], 4)
    search_time_ms = round((time.time() - start_time) * 1000, 4)

    return jsonify(
        {
            "results": results,
            "processing_time": search_time_ms,
            "inference_time": inference_time,
        }
    )


@app.route(f"/{IMAGE_DIR}/<path:filename>")
def serve_image(filename):
    return send_from_directory(IMAGE_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
