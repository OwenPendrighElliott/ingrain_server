import os
import requests
import ingrain
import base64
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


# Constants
HNSWLIB_SERVER_URL = "http://localhost:8685"
IMAGE_DIR = "images"
INDEX_NAME = "image_search"
CLIP_MODEL_NAME = "MobileCLIP-S2"
CLIP_PRETRAINED = "datacompdr"
MODEL_DIM = 512
INDEXING_BATCH_SIZE = 512
NUM_THREADS = 10
BATCH_SIZE = 1


# Function to process a batch of images and get their embeddings
def process_batch(filenames):
    client = ingrain.Client(return_numpy=False)
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
    embeddings = response["embeddings"]

    return list(zip(filenames, embeddings))


def main():
    # Initialize ingrain client
    client = ingrain.Client(return_numpy=False)
    client.load_clip_model(name=CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED)

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

    all_images = []
    embeddings = []
    ids = []

    image_files = [
        f
        for f in os.listdir(IMAGE_DIR)
        if f.endswith((".png", ".jpg", ".jpeg", ".webp"))
    ]

    # Split image_files into batches
    batches = [
        image_files[i : i + BATCH_SIZE] for i in range(0, len(image_files), BATCH_SIZE)
    ]

    i = 0
    # Use ThreadPoolExecutor to parallelize batch processing
    with ProcessPoolExecutor(max_workers=NUM_THREADS) as executor:
        for batch_results in tqdm(
            executor.map(process_batch, batches), total=len(batches)
        ):
            for filename, embedding in batch_results:
                embeddings.append(embedding)
                all_images.append(filename)
                ids.append(i)
                i += 1

    for i in tqdm(range(0, len(embeddings), INDEXING_BATCH_SIZE)):
        response = requests.post(
            f"{HNSWLIB_SERVER_URL}/add_documents",
            json={
                "indexName": INDEX_NAME,
                "ids": ids[i : i + INDEXING_BATCH_SIZE],
                "vectors": embeddings[i : i + INDEXING_BATCH_SIZE],
                "metadatas": [
                    {"source_image": im}
                    for im in all_images[i : i + INDEXING_BATCH_SIZE]
                ],
            },
        )

    response = requests.post(
        f"{HNSWLIB_SERVER_URL}/save_index",
        json={"indexName": INDEX_NAME},
    )


if __name__ == "__main__":
    main()
