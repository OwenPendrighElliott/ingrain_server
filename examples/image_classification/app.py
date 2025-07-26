import flask
import ingrain
from flask import request, jsonify, render_template

app = flask.Flask(__name__)

MODEL = "hf_hub:timm/mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k"

with open("in1kclasses.txt", "r") as f:
    CLASSES = f.read().splitlines()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return "Classifcation Demo is up and running!"


@app.route("/classify", methods=["POST"])
def classify():
    client = ingrain.Client(return_numpy=True)
    data = request.json
    image_b64 = data["image"]
    response = client.infer_image(MODEL, image=[image_b64])
    logits = response["embeddings"][0]
    top5 = sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)[:5]
    results = [{"class": CLASSES[i], "score": logits[i]} for i in top5]
    return jsonify(results)


if __name__ == "__main__":
    client = ingrain.Client()
    client.load_model(MODEL, library="timm")
    app.run(port=5000, debug=True)
