import os
import time
import numpy as np
import ingrain
import json
from flask import Flask, request, render_template, jsonify, send_from_directory
import hnswlib

# Constants
IMAGE_DIR = "images"
INDEX_FILE = "image_index.bin"
CLIP_MODEL_NAME = "MobileCLIP-S2"
CLIP_PRETRAINED = "datacompdr"
MODEL_DIM = 512
K = 20
INDEX = hnswlib.Index(space="cosine", dim=MODEL_DIM)
INDEX.load_index(INDEX_FILE)
INDEX.set_ef(256)

ID_TO_IMAGE_MAPPING = json.load(open("image_id_mapping.json"))

# Create Flask app
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/search", methods=["POST"])
def search():
    query_text = request.json["query_text"]

    client = ingrain.Client()
    response = client.infer_text(
        name=CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, text=query_text
    )
    query_embedding = np.array(response["embeddings"])

    # Search the index for similar images
    start_time = time.time()
    labels, distances = INDEX.knn_query(query_embedding, k=K)
    labels = labels.flatten()
    inference_time = round(response["processingTimeMs"], 4)
    search_time_ms = round((time.time() - start_time) * 1000, 4)

    # Prepare the image results
    results = []
    for label in labels:
        img_filename = ID_TO_IMAGE_MAPPING[str(label)]
        img_path = f"/{IMAGE_DIR}/{img_filename}"
        results.append({"img_path": img_path})

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
