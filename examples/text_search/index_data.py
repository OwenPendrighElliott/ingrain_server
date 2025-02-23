import os
import ingrain
import json
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from typing import List

# Constants
HNSWLIB_SERVER_URL = "http://localhost:8685"
MODEL_NAME = "intfloat/e5-small-v2"
MODEL_DIM = 384
NUM_THREADS = 20
BATCH_SIZE = 4
INDEX_NAME = "scidocs_index"
PASSAGE_PREFIX = "passage: "
INDEXING_BATCH_SIZE = 512
CORPUS_FILE = os.path.join("data", "scidocs", "scidocs", "corpus.jsonl")

# Initialize ingrain client
client = ingrain.Client(return_numpy=False)
client.load_sentence_transformer_model(name=MODEL_NAME)

# Initialize HNSWLib index
response = requests.post(
    f"{HNSWLIB_SERVER_URL}/create_index",
    json={
        "indexName": INDEX_NAME,
        "dimension": MODEL_DIM,
        "indexType": "Approximate",
        "spaceType": "IP",
        "efConstruction": 512,
        "M": 16,
    },
)


# Function to process a batch of corpus data
def process_batch(corpus_batch: List[dict]):
    client = ingrain.Client(return_numpy=False)
    texts = [PASSAGE_PREFIX + doc["text"] for doc in corpus_batch]
    try:
        response = client.infer(name=MODEL_NAME, text=texts)
        embeddings = response["textEmbeddings"]
        ids = [doc["_id"] for doc in corpus_batch]
        return list(zip(ids, embeddings))
    except Exception as e:
        print(f"Error processing batch: {e}")
        return []


# Main Code
corpus = []
with open(CORPUS_FILE, "r") as f:
    for line in f:
        doc = json.loads(line)
        corpus.append(doc)

# Prepare index and mapping
all_passages = []
ids = []
source_texts = []
source_titles = []
source_ids = []
embeddings = []

# Split corpus into batches
batches = [corpus[i : i + BATCH_SIZE] for i in range(0, len(corpus), BATCH_SIZE)]

i = 0
# Use ThreadPoolExecutor to parallelize batch processing
with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
    # Wrap the map with tqdm for progress tracking
    for batch_results in tqdm(executor.map(process_batch, batches), total=len(batches), unit="doc", unit_scale=BATCH_SIZE, desc="Processing documents", ascii=True):
        if batch_results:
            for doc_id, embedding in batch_results:
                embeddings.append(embedding)
                ids.append(i)
                source_texts.append(corpus[i]["text"])
                source_titles.append(corpus[i]["title"])
                source_ids.append(doc_id)
                i += 1

print(len(source_texts), len(embeddings))
print(len(ids), len(source_ids))
for i in tqdm(range(0, len(embeddings), INDEXING_BATCH_SIZE)):
    metadatas = [
        {"source_id": source_ids[j], "text": source_texts[j], "title": source_titles[j]}
        for j in range(i, min(i + INDEXING_BATCH_SIZE, len(embeddings)))
    ]

    response = requests.post(
        f"{HNSWLIB_SERVER_URL}/add_documents",
        json={
            "indexName": INDEX_NAME,
            "ids": ids[i : i + INDEXING_BATCH_SIZE],
            "vectors": embeddings[i : i + INDEXING_BATCH_SIZE],
            "metadatas": metadatas,
        },
    )


response = requests.post(
    f"{HNSWLIB_SERVER_URL}/save_index",
    json={"indexName": INDEX_NAME},
)
