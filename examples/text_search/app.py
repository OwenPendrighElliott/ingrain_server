import os
import json
import ingrain
import requests
import time
from flask import Flask, render_template, request, jsonify

# Constants
MODEL = "intfloat/e5-small-v2"
client = ingrain.Client()
client.load_model(MODEL, library="sentence_transformers")
MODEL_DIM = client.model_embedding_dims(MODEL).embedding_size
INDEX_NAME = "scidocs_index"
TOP_K = 50
HNSWLIB_SERVER_URL = "http://localhost:8685"


# Initialize Flask app
app = Flask(__name__)


# Load SciDocs corpus
CORPUS_FILE = os.path.join("data", "scidocs", "scidocs", "corpus.jsonl")
corpus_dict = {}
with open(CORPUS_FILE, "r") as f:
    for line in f:
        doc = json.loads(line)
        corpus_dict[doc["_id"]] = doc  # Create a dictionary for quick lookup by _id


# Search endpoint
@app.route("/search", methods=["POST"])
def search():
    query = request.form.get("query")
    if not query:
        return jsonify([])

    # Perform inference on query
    client = ingrain.Client(return_numpy=False)
    query = "query: " + query
    response = client.embed(name=MODEL, text=query)
    query_embedding = response.text_embeddings[0]
    embedding_time = response.processing_time_ms
    # Perform search on HNSWLib index
    start_search_time = time.time()
    search_response = requests.post(
        f"{HNSWLIB_SERVER_URL}/search",
        json={
            "indexName": INDEX_NAME,
            "queryVector": query_embedding,
            "k": TOP_K,
            "efSearch": 256,
            "filter": "",
            "returnMetadata": True,
        },
    ).json()
    search_time = (time.time() - start_search_time) * 1000  # Convert to milliseconds

    # Prepare search results
    results = []

    for i in range(len(search_response["hits"])):
        distance = search_response["distances"][i]
        title = search_response["metadatas"][i]["title"]
        text = search_response["metadatas"][i]["text"]

        results.append({"title": title, "text": text, "distance": distance})

    return jsonify(
        {
            "results": results,
            "embedding_time_ms": embedding_time,
            "search_time_ms": search_time,
        }
    )


# Home page
@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
