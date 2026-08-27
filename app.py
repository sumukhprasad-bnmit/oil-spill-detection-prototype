import base64
import io
import sys
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, render_template_string, request
from PIL import Image


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make sure "src" (sibling of this file) is importable, same as the notebook does
# with sys.path.append(str(Path.cwd().parent)) — here we just add our own directory.
sys.path.append(str(Path(__file__).resolve().parent))

from src.model import UNet  

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Model setup (loaded once at startup)
# ---------------------------------------------------------------------------
MODEL_PATH = Path(__file__).resolve().parent / "models" / "epoch20_best_model.pt"

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model = UNet().to(device)
model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
    )
)
model.eval()


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def predict(model, image, device, threshold=0.5):
    model.eval()

    with torch.no_grad():
        image = image.unsqueeze(0).to(device)

        logits = model(image)

        probability = torch.sigmoid(logits)

        prediction = probability >= threshold

    return (
        probability.squeeze().cpu().numpy(),
        prediction.squeeze().cpu().numpy(),
    )


# ---------------------------------------------------------------------------
# Preprocessing —
# ---------------------------------------------------------------------------
def preprocess_image(file_stream):
    # Load grayscale image.
    image = np.array(
        Image.open(file_stream).convert("L"),
        dtype=np.float32,
    )

    # Normalize SAR image from [0, 255] --> [0, 1].
    image /= 255.0

    # Add channel dimension: [H, W] --> [1, H, W]
    image = torch.from_numpy(image).unsqueeze(0)

    return image


def array_to_base64_png(array, is_bool=False):
    """Convert a 2D numpy array (probability in [0,1], or boolean mask) to a base64 PNG."""
    if is_bool:
        pixels = (array.astype(np.uint8)) * 255
    else:
        pixels = (np.clip(array, 0.0, 1.0) * 255).astype(np.uint8)

    img = Image.fromarray(pixels, mode="L")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

def create_overlay_image(image, prediction):
    fig, ax = plt.subplots(figsize=(image.shape[1] / 100, image.shape[0] / 100))

    # request image
    ax.imshow(image)

    # prediction mask overlay
    ax.imshow(
        prediction,
        alpha=0.45,
    )

    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        bbox_inches="tight",
        pad_inches=0,
        transparent=False,
    )
    plt.close(fig)

    buffer.seek(0)
    return "data:image/png;base64,"+base64.b64encode(buffer.read()).decode("utf-8")


@app.route("/", methods=["GET"])
def home():
    with open("template.html") as f:
        return render_template_string(f.read())


@app.route("/predict", methods=["POST"])
def predict_route():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files["image"]
    if image_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        image_file.stream.seek(0)
        original_image = Image.open(image_file.stream).convert("RGB")
        original_image = np.array(original_image)
	   
        # reset stream because preprocess_image will read it
        image_file.stream.seek(0)
        image_tensor = preprocess_image(image_file.stream)

        probability, prediction = predict(
            model,
            image_tensor,
            device,
        )

        spill_percentage = float(prediction.mean()) * 100.0
	   
        overlay_image = create_overlay_image(original_image, prediction)
        image_file.stream.seek(0)

        return (
            jsonify(
                {
                    "status": "Success",
                    "spill_percentage": spill_percentage,
                    "original_image": "data:image/png;base64,"+base64.b64encode(image_file.read()).decode("utf-8"),
                    "probability_image": array_to_base64_png(probability, is_bool=False),
                    "prediction_image": array_to_base64_png(prediction, is_bool=True),
                    "overlay_image": overlay_image,
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"status": "Failed", "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)