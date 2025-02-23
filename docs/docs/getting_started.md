# Getting Started

Ingrain is a scalable wrapper for [Triton Inference Server](https://developer.nvidia.com/triton-inference-server) that makes using it with sentence transformers, timm, and open CLIP models easy.

The server handles all model conversion, loading, and preprocessing. It manages tokenizers, image downloading, image preprocessing, and other complexities of running a model. 

## Setup

To run Ingrain, you need to have Docker installed. You can run the server with the following command:

```bash
docker run --name ingrain_server -p 8686:8686 -p 8687:8687 --gpus all owenpelliott/ingrain-server:latest
```

To run without a GPU, remove the `--gpus all` flag.

You can also install a python client to get started quickly:

```bash
pip install ingrain
```

## Example Usage

Ingrain has two main components: the model server and the inference server. The model server handles all model loading and unloading operations whereas the inference server handles all inference requests on loaded models.

### Loading Models

Models are accessable via the same naming conventions as their underlying libraries (`open_clip_torch`, `sentence_transformers`, or `timm`). For example to load a MobileCLIP model in `open_clip` you would use the following code:

```python
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms('MobileCLIP-S2', pretrained='datacompdr')
model.eval()
tokenizer = open_clip.get_tokenizer('MobileCLIP-S2')
```

To do the same in Ingrain, you would use the following code:

```python
import ingrain

client = ingrain.Client()
client.load_clip_model(name="MobileCLIP-S2", pretrained="datacompdr")
```

A similar pattern holds for `sentence_transformers` models:

Sentence Transformers:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('intfloat/e5-small-v2')
```

Ingrain:

```python
import ingrain

client = ingrain.Client()
client.load_sentence_transformer_model(name="intfloat/e5-small-v2")
```

### Inference

Inference requests are made to the Inference Server. The server can handle both text and image inputs. Images can be URLs to image assets or base64 encoded images. Requests can include single items or batches of items - the server will also dynamically batch requests to optimize performance.

For example to encode a list of sentences using the `intfloat/e5-small-v2` model:
```python
import ingrain

client = ingrain.Client()
client.load_sentence_transformer_model(name="intfloat/e5-small-v2")

response = client.infer(
    name="intfloat/e5-small-v2", 
    text=["I am a sentence.", "I am another sentence.", "I am a third sentence."]
)
```

This will return a dictionary like so:

```python
{
    'textEmbeddings': [
        [-0.028349684551358223, 0.03690376505255699, ...],
        [-0.657342546546734223, 0.46323376505255699, ...],
        [-0.035745636365643294, 0.95440376505255699, ...]
    ], 
    'imageEmbeddings': None, 
    'processingTimeMs': 17.091903000618913
}
```