import os
import numpy as np
import hnswlib
import ingrain
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

# Constants
MODEL_NAME = "intfloat/e5-base-v2"
MODEL_DIM = 768
NUM_THREADS = 8
BATCH_SIZE = 8
INDEX_FILE = "scidocs_index.bin"
MAPPING_FILE = "scidocs_id_mapping.json"
PASSAGE_PREFIX = "passage: "
CORPUS_FILE = os.path.join("data", "scidocs", "scidocs", "corpus.jsonl")

# Initialize ingrain client
client = ingrain.Client(return_numpy=True)
client.load_sentence_transformer_model(name=MODEL_NAME)

# Initialize HNSWLib index
index = hnswlib.Index(space="cosine", dim=MODEL_DIM)


# Function to process a batch of corpus data
def process_batch(corpus_batch: List[dict]):
    client = ingrain.Client(return_numpy=True)
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
embeddings = []
id_to_passage_mapping = {}

index.init_index(max_elements=len(corpus), ef_construction=256, M=16)

# Split corpus into batches
batches = [corpus[i : i + BATCH_SIZE] for i in range(0, len(corpus), BATCH_SIZE)]

# Use ThreadPoolExecutor to parallelize batch processing
with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
    futures = {executor.submit(process_batch, batch): batch for batch in batches}
    for future in tqdm(as_completed(futures), total=len(futures)):
        batch_results = future.result()
        if batch_results:
            for doc_id, embedding in batch_results:
                embeddings.append(embedding)
                # Map document ID to its index in the embedding array
                id_to_passage_mapping[doc_id] = len(embeddings) - 1

embeddings = np.vstack(embeddings)
index.add_items(embeddings)
index.save_index(INDEX_FILE)

# Save the ID to passage mapping as JSON
with open(MAPPING_FILE, "w") as f:
    json.dump(id_to_passage_mapping, f)
