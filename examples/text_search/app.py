import os
import json
import hnswlib
import numpy as np
import ingrain
import time
from flask import Flask, render_template, request, jsonify

# Constants
MODEL = "intfloat/e5-base-v2"
INDEX_FILE = "scidocs_index.bin"
MAPPING_FILE = "scidocs_id_mapping.json"
MODEL_DIM = 768
TOP_K = 50

# Initialize Flask app
app = Flask(__name__)

# Load HNSWLib index
index = hnswlib.Index(space="cosine", dim=MODEL_DIM)
index.load_index(INDEX_FILE)
index.set_ef(256)

# Load ID to passage mapping
with open(MAPPING_FILE, "r") as f:
    id_to_passage_mapping = json.load(f)

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
    client = ingrain.Client(return_numpy=True)
    query = "query: " + query
    response = client.infer(name=MODEL, text=query)
    query_embedding = response["textEmbeddings"][0]
    embedding_time = response["processingTimeMs"]
    # Perform search on HNSWLib index
    start_search_time = time.time()
    labels, distances = index.knn_query(query_embedding, k=TOP_K)
    search_time = (time.time() - start_search_time) * 1000  # Convert to milliseconds

    # Prepare search results
    results = []
    for label, distance in zip(labels[0], distances[0]):
        doc_id = list(id_to_passage_mapping.keys())[
            int(label)
        ]  # Get _id from the mapping
        passage = corpus_dict.get(doc_id, None)  # Retrieve document from corpus

        if passage:
            results.append(
                {
                    "title": passage.get("title", "No Title"),
                    "text": passage.get("text", "No Text"),
                    "distance": float(
                        distance
                    ),  # Convert np.float32 to native Python float
                }
            )

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
