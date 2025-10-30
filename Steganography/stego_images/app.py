import os
import uuid
import logging
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from Crypto.Random import get_random_bytes
import cv2  # Import OpenCV

# Import your steganography classes
from AES_LSB import UniversalSteganography as LsbStego
from hugo import HugoSteganography
from wow import WowSteganography


# ===============================================
# 1. SRNet Model Definition (Copied from your training script)
# ===============================================
class SRNet(nn.Module):
    def __init__(self):
        super(SRNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.layer3 = self._make_res_block(16, 16)
        self.layer4 = self._make_res_block(16, 64)
        self.layer5 = self._make_res_block(64, 128)
        self.layer6 = self._make_res_block(128, 256)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(256, 1)
        )

    def _make_res_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.fc(x)
        return x


# Configure logging and Flask App
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StegoApp")
app = Flask(__name__)

# --- Configuration ---
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, "uploads")
STATIC_FOLDER = os.path.join(APP_ROOT, "static")
GENERATED_FOLDER = os.path.join(STATIC_FOLDER, "generated")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GENERATED_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SECRET_KEY"] = os.urandom(24)

# ===============================================
# 2. Load PyTorch Detection Model from .pth file
# ===============================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.path.join(APP_ROOT, "best_srnet_from_scratch_changed.pth")
detection_model = None

try:
    detection_model = SRNet().to(DEVICE)
    detection_model.load_state_dict(
        torch.load(MODEL_PATH, map_location=torch.device(DEVICE))
    )
    detection_model.eval()
    logger.info(
        f"PyTorch detection model loaded successfully from '{MODEL_PATH}' on device '{DEVICE}'."
    )
except FileNotFoundError:
    logger.error(
        f"FATAL: Model file not found at '{MODEL_PATH}'. Detection API will be disabled."
    )
    detection_model = None
except Exception as e:
    logger.error(f"FATAL: An error occurred while loading the PyTorch model: {e}")
    detection_model = None

# --- Steganography Algorithm Mapping ---
ALGORITHMS = {
    "lsb": LsbStego(payload=0.3),
    "hugo": HugoSteganography(payload=0.3),
    "wow": WowSteganography(payload=0.3),
}
ENCRYPTION_KEY = get_random_bytes(32)


# === Page Routes ===
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/embed")
def embed_page():
    return render_template("embed.html")


@app.route("/detect")
def detect_page():
    return render_template("detect.html")


@app.route("/extract")
def extract_page():
    return render_template("extract.html")


# === API Routes ===
@app.route("/perform_embed", methods=["POST"])
def perform_embed():
    if (
        "image" not in request.files
        or "message" not in request.form
        or "algorithm" not in request.form
    ):
        return jsonify({"error": "Missing form data."}), 400

    file = request.files["image"]
    message = request.form["message"]
    algorithm_name = request.form["algorithm"]

    if file.filename == "" or not message or not algorithm_name:
        return jsonify({"error": "All fields are required."}), 400
    if algorithm_name not in ALGORITHMS:
        return jsonify({"error": "Invalid algorithm."}), 400

    try:
        filename = secure_filename(file.filename)
        unique_id = uuid.uuid4().hex

        # Save original file to a temporary location
        cover_path = os.path.join(
            app.config["UPLOAD_FOLDER"], f"{unique_id}_{filename}"
        )
        file.save(cover_path)

        # Define output paths
        stego_filename = f"stego_{unique_id}_{os.path.splitext(filename)[0]}.jpg"
        output_path = os.path.join(GENERATED_FOLDER, stego_filename)

        # Perform embedding
        stego_processor = ALGORITHMS[algorithm_name]
        success = stego_processor.embed_file(
            cover_path=cover_path,
            output_path=output_path,
            data=message,
            key=ENCRYPTION_KEY,
        )
        if not success:
            raise Exception("Embedding process failed.")

        # --- Visual Distortion Map Generation ---
        # Load the original and stego images
        original_image = cv2.imread(cover_path)
        stego_image = cv2.imread(output_path)

        # Ensure images are the same size
        if original_image.shape != stego_image.shape:
            stego_image = cv2.resize(
                stego_image, (original_image.shape[1], original_image.shape[0])
            )

        # Calculate the absolute difference
        difference = cv2.absdiff(original_image, stego_image)

        # Convert to grayscale and create a binary map of changes
        gray_diff = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
        _, binary_map = cv2.threshold(gray_diff, 1, 255, cv2.THRESH_BINARY)

        # Save the distortion map
        distortion_filename = f"distort_{unique_id}_{os.path.splitext(filename)[0]}.jpg"
        distortion_path = os.path.join(GENERATED_FOLDER, distortion_filename)
        cv2.imwrite(distortion_path, binary_map)

        # For frontend comparison, save a copy of the original cover image to the generated folder
        cover_display_filename = (
            f"cover_{unique_id}_{os.path.splitext(filename)[0]}.jpg"
        )
        cover_display_path = os.path.join(GENERATED_FOLDER, cover_display_filename)
        cv2.imwrite(cover_display_path, original_image)

        return jsonify(
            {
                "success": True,
                "coverUrl": f"/static/generated/{cover_display_filename}",
                "stegoUrl": f"/static/generated/{stego_filename}",
                "distortionUrl": f"/static/generated/{distortion_filename}",
            }
        )

    except Exception as e:
        logger.error(f"An error occurred during embedding: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up the uploaded cover image
        if "cover_path" in locals() and os.path.exists(cover_path):
            os.remove(cover_path)


@app.route("/perform_detect", methods=["POST"])
def perform_detect():
    if detection_model is None:
        return (
            jsonify(
                {
                    "error": "Detection model is not loaded on the server. Check server logs."
                }
            ),
            500,
        )
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No image selected."}), 400

    temp_path = None
    try:
        filename = secure_filename(file.filename)
        temp_path = os.path.join(
            app.config["UPLOAD_FOLDER"], f"detect_{uuid.uuid4().hex}_{filename}"
        )
        file.save(temp_path)

        IMG_SIZE = 256
        transform = transforms.Compose(
            [
                transforms.Resize(IMG_SIZE),
                transforms.CenterCrop(IMG_SIZE),
                transforms.ToTensor(),
            ]
        )
        image = Image.open(temp_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = detection_model(image_tensor).squeeze(1)
            probability = torch.sigmoid(logits).item()

        is_stego = probability >= 0.5
        result_text = "Stego Image" if is_stego else "Cover Image"
        confidence = probability * 100 if is_stego else (1 - probability) * 100

        return jsonify(
            {
                "success": True,
                "prediction": result_text,
                "confidence": f"{confidence:.2f}%",
            }
        )
    except Exception as e:
        logger.error(f"An error occurred during detection: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    app.run(debug=True)
