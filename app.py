import base64
import io
import sys
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, render_template_string, request
from PIL import Image

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


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="viewport" content="width=device-width, initial-scale=1.0">
    <title>Oil Spill Detection Dashboard</title>
    <style>
        body { font-family: system-ui, sans-serif; max-width: 700px; margin: 50px auto; padding: 20px; line-height: 1.6; background: #f9f9f9; }
        .card { border: 1px solid #ddd; padding: 25px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); background: white; }
        input[type="file"] { margin: 20px 0; display: block; width: 100%; }
        button { background: #0070f3; color: white; border: none; padding: 12px 20px; border-radius: 6px; cursor: pointer; font-size: 16px; font-weight: 500; width: 100%; }
        button:hover { background: #0051a8; }
        .status { margin-top: 20px; font-family: monospace; font-size: 13px; white-space: pre-wrap; }
        .status.error { color: #d33; }
        .status.ok { color: #0a0; }
        .results { display: none; margin-top: 25px; }
        .results.visible { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
        .results img { width: 100%; border-radius: 8px; border: 1px solid #ddd; }
        .results figcaption { text-align: center; font-size: 13px; color: #555; margin-top: 6px; }
        .spill-pct { margin-top: 15px; font-size: 15px; text-align: center; }
    </style>
</head>
<body>
    <div class="card">
        <center><h2>Oil Spill Detection</h2></center>
        <p>Upload a grayscale SAR .png image. It will be run through the UNet model to detect oil spill regions.</p>

        <form id="uploadForm">
            <input type="file" name="image" accept="image/png" required>
            <button type="submit">Upload and Predict</button>
        </form>

        <div id="statusBox" class="status"></div>

        <div id="results" class="results">
            <figure>
                <img id="probabilityImg" alt="Probability map">
                <figcaption>Probability</figcaption>
            </figure>
            <figure>
                <img id="predictionImg" alt="Predicted mask">
                <figcaption>Predicted</figcaption>
            </figure>
        </div>
        <div id="spillPct" class="spill-pct"></div>
    </div>

    <script>
        document.getElementById('uploadForm').onsubmit = async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const statusBox = document.getElementById('statusBox');
            const results = document.getElementById('results');
            const spillPct = document.getElementById('spillPct');

            results.classList.remove('visible');
            spillPct.textContent = '';
            statusBox.className = 'status';
            statusBox.textContent = 'Processing...';

            try {
                const res = await fetch('/predict', { method: 'POST', body: formData });
                const data = await res.json();

                if (!res.ok) {
                    statusBox.className = 'status error';
                    statusBox.textContent = 'Error: ' + (data.error || 'Unknown error');
                    return;
                }

                document.getElementById('probabilityImg').src = data.probability_image;
                document.getElementById('predictionImg').src = data.prediction_image;
                spillPct.textContent = `Predicted spill coverage: ${data.spill_percentage.toFixed(2)}%`;

                results.classList.add('visible');
                statusBox.className = 'status ok';
                statusBox.textContent = 'Done.';
            } catch (err) {
                statusBox.className = 'status error';
                statusBox.textContent = 'Error: ' + err.message;
            }
        };
    </script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE)


@app.route("/predict", methods=["POST"])
def predict_route():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files["image"]
    if image_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        image_tensor = preprocess_image(image_file.stream)

        probability, prediction = predict(
            model,
            image_tensor,
            device,
        )

        spill_percentage = float(prediction.mean()) * 100.0

        return (
            jsonify(
                {
                    "status": "Success",
                    "spill_percentage": spill_percentage,
                    "probability_image": array_to_base64_png(probability, is_bool=False),
                    "prediction_image": array_to_base64_png(prediction, is_bool=True),
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"status": "Failed", "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)