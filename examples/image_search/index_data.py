import os
import numpy as np
import hnswlib
import ingrain
import base64
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Constants
IMAGE_DIR = "images"
INDEX_FILE = "image_index.bin"
MAPPING_FILE = "image_id_mapping.json"
CLIP_MODEL_NAME = "MobileCLIP-S2"
CLIP_PRETRAINED = "datacompdr"
MODEL_DIM = 512
NUM_THREADS = 5  # Adjust the number of threads to your machine's capability
BATCH_SIZE = 4  # Number of images per batch

# Initialize ingrain client
client = ingrain.Client(return_numpy=True)
client.load_clip_model(name=CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED)

# Initialize HNSWLib index
index = hnswlib.Index(space="cosine", dim=MODEL_DIM)


# Function to process a batch of images and get their embeddings
def process_batch(filenames):
    client = ingrain.Client(return_numpy=True)
    image_datas = []
    for filename in filenames:
        img_path = os.path.join(IMAGE_DIR, filename)
        with open(img_path, "rb") as img_file:
            image_data = base64.b64encode(img_file.read()).decode("utf-8")
            data_type = img_path.split(".")[-1]
            image_data_uri = f"data:image/{data_type};base64,{image_data}"
            image_datas.append(image_data_uri)

    response = client.infer_image(
        name=CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, image=image_datas
    )
    embeddings = np.array(response["embeddings"])

    return list(zip(filenames, embeddings))


# Prepare index and mapping
all_images = []
embeddings = []
id_to_image_mapping = {}


image_files = [
    f for f in os.listdir(IMAGE_DIR) if f.endswith((".png", ".jpg", ".jpeg", ".webp"))
]
index.init_index(max_elements=len(image_files), ef_construction=256, M=16)

# Split image_files into batches
batches = [
    image_files[i : i + BATCH_SIZE] for i in range(0, len(image_files), BATCH_SIZE)
]

# Use ThreadPoolExecutor to parallelize batch processing
with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
    futures = {executor.submit(process_batch, batch): batch for batch in batches}
    for future in tqdm(as_completed(futures), total=len(futures)):
        batch_results = future.result()
        for filename, embedding in batch_results:
            embeddings.append(embedding)
            all_images.append(filename)
            # Map filename to its index (use index length as ID)
            id_to_image_mapping[len(id_to_image_mapping)] = filename

embeddings = np.vstack(embeddings)
index.add_items(embeddings)
index.save_index(INDEX_FILE)

# Save the ID to image mapping as JSON
with open(MAPPING_FILE, "w") as f:
    json.dump(id_to_image_mapping, f)
