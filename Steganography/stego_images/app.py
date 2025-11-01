import os
import uuid
import logging
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from Crypto.Random import get_random_bytes
import cv2  # Import OpenCV
import io
import base64
import math

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


# ===============================================
# 2. Image-in-Image Steganography Functions
# ===============================================
def write_image_data(img, data, filename):
    """
    Embeds image data into a cover image using LSB steganography.
    Format: [filename(12 bytes)][filesize(4 bytes)][image data]
    """
    byte_array = bytearray()
    # Pad filename to 12 characters
    byte_array.extend(bytes(filename.rjust(12, '0'), 'utf-8'))
    # Add file size as 4 bytes (little endian)
    byte_array.extend(len(data).to_bytes(4, 'little'))
    # Add the actual image data
    byte_array.extend(data)
    
    height, width, channels = img.shape
    data_size = len(byte_array)
    byte_num = 0
    nib_num = 0
    
    for i in range(0, height):
        for j in range(0, width):
            for c in range(0, channels):
                # Clear last 2 bits and insert 2 bits from data
                img[i, j, c] = img[i, j, c] & 0xFC | (byte_array[byte_num] >> nib_num * 2) & 0x03
                nib_num += 1
                nib_num %= 4
                if nib_num == 0:
                    byte_num += 1
                    if byte_num >= data_size:
                        return img
    return img


def extract_image_data(img):
    """
    Extracts hidden image data from a stego image.
    Returns: (image_data_bytes, original_filename)
    """
    filename_byte_array = bytearray()
    filesize_byte_array = bytearray()
    byte_array = bytearray()
    height, width, channels = img.shape
    data_size = 16
    byte = 0
    nib = 0
    byte_dat = 0
    
    for i in range(0, height):
        for j in range(0, width):
            for c in range(0, channels):
                # Extract 2 bits from pixel
                byte_dat = byte_dat | (img[i, j, c] & 0x03) << nib * 2
                nib += 1
                nib %= 4
                if nib == 0:
                    if byte < 12:
                        filename_byte_array.append(byte_dat)
                    elif byte < 16:
                        filesize_byte_array.append(byte_dat)
                        if byte == 15:
                            # Calculate data size from little endian bytes
                            data_size = int.from_bytes(filesize_byte_array, 'little') + 16
                    else:
                        byte_array.append(byte_dat)
                    byte_dat = 0
                    byte += 1
                if byte >= data_size:
                    # Decode filename
                    filename = filename_byte_array.decode().replace('0', '', filename_byte_array.decode().count('0') - filename_byte_array.decode()[::-1].find('0') + 1)
                    filename = filename.lstrip('0')  # Remove leading zeros
                    return bytes(byte_array), filename
    
    return None, None


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
# 3. Load PyTorch Detection Model from .pth file
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


# ===============================================
# NEW: Image-in-Image Steganography Routes
# ===============================================
@app.route("/image-stego")
def image_stego_page():
    """Page for image-in-image steganography"""
    return render_template("image_stego.html")


@app.route("/embed_image", methods=["POST"])
def embed_image():
    """
    API endpoint to embed a secret image inside a cover image.
    Expects: 'cover' and 'secret' image files
    Returns: Base64 encoded stego image
    """
    try:
        if 'cover' not in request.files or 'secret' not in request.files:
            return jsonify({'success': False, 'error': 'Both cover and secret images are required'}), 400
        
        cover_file = request.files['cover']
        secret_file = request.files['secret']
        
        if cover_file.filename == '' or secret_file.filename == '':
            return jsonify({'success': False, 'error': 'Both images must be selected'}), 400
        
        # Check filename length (max 12 characters including extension)
        if len(secret_file.filename) > 12:
            return jsonify({'success': False, 'error': 'Secret image filename must be 12 characters or less (including extension)'}), 400
        
        # Load cover image
        cover_img = Image.open(cover_file).convert('RGB')
        cover_array = np.array(cover_img)
        
        # Calculate maximum capacity (2 bits per pixel channel)
        max_bytes = math.floor(cover_array.shape[0] * cover_array.shape[1] * 3 * 2 / 8)
        
        # Load secret image and convert to PNG bytes
        secret_img = Image.open(secret_file)
        secret_bytes_io = io.BytesIO()
        secret_img.save(secret_bytes_io, format='PNG')
        secret_data = secret_bytes_io.getvalue()
        
        # Check if secret image fits in cover image
        if len(secret_data) > max_bytes - 16:
            return jsonify({
                'success': False,
                'error': f'Secret image is too large! Maximum size: {(max_bytes - 16) // 1024}KB, Your image: {len(secret_data) // 1024}KB'
            }), 400
        
        # Embed secret image into cover image
        stego_array = write_image_data(cover_array, secret_data, secret_file.filename)
        
        # Convert back to PIL Image
        stego_img = Image.fromarray(stego_array)
        
        # Convert to base64 for response
        output = io.BytesIO()
        stego_img.save(output, format='PNG')
        output.seek(0)
        img_base64 = base64.b64encode(output.getvalue()).decode()
        
        logger.info(f"Successfully embedded '{secret_file.filename}' into cover image")
        
        return jsonify({
            'success': True,
            'image': img_base64,
            'message': f'Successfully embedded {len(secret_data) // 1024}KB secret image'
        })
        
    except Exception as e:
        logger.error(f"Error during image embedding: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route("/extract_image", methods=["POST"])
def extract_image():
    """
    API endpoint to extract a hidden image from a stego image.
    Expects: 'stego' image file
    Returns: Base64 encoded extracted image and original filename
    """
    try:
        if 'stego' not in request.files:
            return jsonify({'success': False, 'error': 'Stego image is required'}), 400
        
        stego_file = request.files['stego']
        
        if stego_file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Load stego image
        stego_img = Image.open(stego_file).convert('RGB')
        stego_array = np.array(stego_img)
        
        # Extract hidden data
        extracted_data, original_filename = extract_image_data(stego_array)
        
        if extracted_data is None:
            return jsonify({'success': False, 'error': 'Failed to extract data from image. This may not be a valid stego image.'}), 400
        
        # Convert extracted data to base64
        img_base64 = base64.b64encode(extracted_data).decode()
        
        logger.info(f"Successfully extracted '{original_filename}' from stego image")
        
        return jsonify({
            'success': True,
            'image': img_base64,
            'filename': original_filename,
            'message': f'Successfully extracted {len(extracted_data) // 1024}KB image'
        })
        
    except Exception as e:
        logger.error(f"Error during image extraction: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# === Existing API Routes (UNCHANGED) ===
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